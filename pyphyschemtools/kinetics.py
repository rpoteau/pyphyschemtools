import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import r2_score as r2
from .visualID_Eng import fg, hl, bg, color
from .colab_utils import colab_upload
import os
import re
import unicodedata
import copy as _copy

# =====================================================================
#  FR — kbasic : recherche de la constante de vitesse k par minimisation
#  du RMSE. Version pédagogique, simple et autonome.
#
#  Une classe avec trois opérations, dans l'ordre :
#     1) kb.read(fichier)  : lire le fichier Excel (temps, Y) + récapitulatif
#     2) kb.fit(ordre)     : faire le fit (trouver le meilleur k)
#     3) kb.plot()         : tracer le résultat (comparer modèle et données)
#
#  Idée : on a des points expérimentaux (temps, grandeur mesurée Y) et un
#  modèle théorique qui dépend de k. On cherche le k qui MINIMISE l'écart
#  entre modèle et données. Cet écart est le RMSE.
#
#  EN — kbasic: pedagogical, self-contained search for the rate constant
#  k by minimizing the RMSE.
#
#  A class with three operations, in order:
#     1) kb.read(file)   : read the Excel file (time, Y) + summary table
#     2) kb.fit(order)   : do the fit (find the best k)
#     3) kb.plot()       : plot the result (compare model and data)
#
#  Idea: we have experimental points (time, measured quantity Y) and a
#  theoretical model that depends on k. We look for the k that MINIMIZES
#  the gap between model and data. That gap is the RMSE.
# =====================================================================

# numpy, pandas and matplotlib.pyplot are already imported at the top of this
# file (used by KORD too); the one new import kbasic needs is minimize:
from scipy.optimize import minimize     # FR: outil qui cherche le minimum d'une fonction / EN: tool that finds the minimum of a function


class kbasic:
    """
    FR — Recherche pédagogique de la constante de vitesse k.

    Utilisation typique ::

        kb = kbasic()
        kb.read("mesures.xlsx")   # 1) lire les données
        kb.fit(ordre=1)           # 2) chercher le meilleur k
        kb.plot()                 # 3) tracer le résultat

    La grandeur mesurée Y décroît avec le temps depuis sa valeur initiale
    Y0. La forme de Y(t) dépend de l'ordre de la réaction :

    - ordre 0 : décroissance LINÉAIRE
    - ordre 1 : décroissance EXPONENTIELLE (le cas le plus courant)
    - ordre 2 : décroissance HYPERBOLIQUE

    EN — Pedagogical search for the rate constant k.

    Typical usage ::

        kb = kbasic()
        kb.read("measurements.xlsx")  # 1) read the data
        kb.fit(order=1)               # 2) look for the best k
        kb.plot()                     # 3) plot the result

    The measured quantity Y decreases with time from its initial value
    Y0. The shape of Y(t) depends on the reaction order:

    - order 0: LINEAR decay
    - order 1: EXPONENTIAL decay (the most common case)
    - order 2: HYPERBOLIC decay
    """

    def __init__(self):
        # Les données expérimentales (remplies par read).
        self.t_exp = None      # temps mesurés
        self.Y_exp = None      # grandeurs mesurées
        # Les en-têtes lus dans le fichier (remplis par read), pour les axes.
        self.label_t = "Temps" # nom de l'axe des temps (par défaut)
        self.label_Y = "Y"     # nom de l'axe de la grandeur (par défaut)
        # Les résultats du fit (remplis par fit).
        self.Y0 = None         # valeur de Y à t=0
        self.ordre = None      # ordre choisi pour le fit
        self.k_opt = None      # meilleure valeur de k trouvée
        self.rmse_min = None   # RMSE au minimum
        self.t_half = None     # temps de demi-réaction t_1/2
        self._initialized = True   # voir __setattr__ ci-dessous

    def __setattr__(self, name, value):
        """
        FR — Protection contre les fautes de frappe sur un nom d'attribut
        (p. ex. `kb.Y_initial = ...` au lieu de `kb.Y0 = ...`) : une fois
        l'objet construit, on ne peut plus CRÉER un nouvel attribut — seuls
        ceux déjà définis dans __init__ peuvent être modifiés. Sans ça,
        Python créerait silencieusement l'attribut mal orthographié, et le
        reste du code continuerait d'utiliser l'ancienne valeur sans jamais
        signaler l'erreur.

        EN — Guards against attribute-name typos (e.g. `kb.Y_initial = ...`
        instead of `kb.Y0 = ...`): once the object is built, no NEW
        attribute can be created — only ones already defined in __init__
        can be changed. Without this, Python would silently create the
        misspelled attribute, and the rest of the code would keep using the
        old value with no error at all.
        """
        if getattr(self, '_initialized', False) and name not in self.__dict__:
            existing = ', '.join(sorted(k for k in self.__dict__ if not k.startswith('_')))
            raise AttributeError(
                f"kbasic has no attribute '{name}' (possible typo?). "
                f"Existing attributes: {existing}"
            )
        object.__setattr__(self, name, value)

    # -----------------------------------------------------------------
    #  Les trois modèles théoriques.
    #  Chacun ne dépend que du temps t, de la constante k et de Y0.
    # -----------------------------------------------------------------
    def _modele(self, t, k, Y0, ordre):
        """Renvoie Y(t) prédit par le modèle de l'ordre demandé."""
        if ordre == 0:
            # Ordre 0 : droite qui descend, bornée à zéro une fois le
            # réactif épuisé (np.maximum empêche de passer sous zéro).
            return np.maximum(Y0 - k * t, 0)
        elif ordre == 1:
            # Ordre 1 : exponentielle. À t=0, Y=Y0 ; à t grand, Y->0.
            return Y0 * np.exp(-k * t)
        elif ordre == 2:
            # Ordre 2 : hyperbole. À t=0, Y=Y0.
            return Y0 / (1 + k * t)
        else:
            raise ValueError("ordre doit valoir 0, 1 ou 2")

    # -----------------------------------------------------------------
    #  Le RMSE : la mesure de l'écart entre données et modèle.
    #  RMSE = racine de la moyenne des carrés des écarts.
    #  Plus il est petit, mieux le modèle colle aux données.
    # -----------------------------------------------------------------
    def _rmse(self, k, ordre, Y0):
        """Calcule le RMSE pour une valeur de k donnée."""
        # ATTENTION : ce n'est pas nous qui appelons cette fonction, c'est
        # minimize (voir la méthode fit). Or minimize travaille toujours avec
        # un TABLEAU de paramètres, jamais avec un simple nombre. Comme on lui
        # donne x0=[k_depart] (une liste), il rappelle _rmse en passant k sous
        # forme de tableau à un élément, par exemple [0.1] puis [0.093]...
        # La VALEUR change à chaque itération, mais la forme reste [un_element].
        # On extrait donc ce nombre unique avec k[0] pour pouvoir l'utiliser
        # dans le modèle. np.ndim(k) vaut 0 pour un nombre, 1 pour un tableau :
        # le test rend la fonction utilisable aussi avec un simple nombre.
        if np.ndim(k) > 0:
            k = k[0]
        Y_modele = self._modele(self.t_exp, k, Y0, ordre)
        return np.sqrt(np.mean((self.Y_exp - Y_modele) ** 2))

    # =================================================================
    #  1) LIRE LES DONNÉES
    #
    #  Fichier Excel à deux colonnes : temps à gauche, Y à droite.
    #  La première ligne PEUT être un en-tête (texte) ou déjà des
    #  données (nombres). On regarde la première cellule :
    #     - texte  -> c'est un en-tête, on saute la 1ère ligne
    #     - nombre -> ce sont déjà des données, on garde tout
    # =================================================================
    def read(self, fichier=None, colab=False):
        """
        FR — Lit un fichier Excel à deux colonnes (temps, Y) et affiche un
        récapitulatif sous forme de DataFrame pandas.

        Détecte automatiquement si la première ligne est un en-tête. Si oui,
        les noms de colonnes serviront de labels pour les axes du tracé.

        Retourne le DataFrame des données lues.

        Arguments :
            fichier : chemin (ou URL) du fichier Excel à lire.
                      Inutile si colab=True (on choisit le fichier à l'upload).
            colab   : si True, ouvre le bouton d'upload de Google Colab
                      et lit le fichier choisi. Défaut False.

        EN — Reads a two-column Excel file (time, Y) and displays a
        summary as a pandas DataFrame.

        Automatically detects whether the first row is a header. If so,
        the column names are used as axis labels for the plot.

        Returns the DataFrame of the data read.

        Args:
            fichier: path (or URL) of the Excel file to read.
                     Unused if colab=True (the file is picked via the upload
                     widget).
            colab: if True, opens the Google Colab upload button and reads
                   the chosen file. Default False.
        """
        # Mode Google Colab : on ouvre le sélecteur de fichier, l'utilisateur
        # choisit son .xlsx, et on récupère son nom pour le lire ensuite.
        if colab:
            fichier = colab_upload()   # bouton "Parcourir" Colab, voir colab_utils.py

        # On lit d'abord SANS supposer d'en-tête (header=None), pour
        # examiner nous-mêmes la première cellule.
        brut = pd.read_excel(fichier, header=None)

        # isinstance(x, str) répond à : "est-ce que x est du texte ?"
        if isinstance(brut.iloc[0, 0], str):
            # En-tête présent : on garde les deux noms de colonnes pour
            # les utiliser comme labels d'axes dans plot(), puis les vraies
            # données commencent à la ligne 1.
            self.label_t = str(brut.iloc[0, 0])
            self.label_Y = str(brut.iloc[0, 1])
            donnees = brut.iloc[1:]
        else:
            # Pas d'en-tête : tout le tableau est déjà des données, et on
            # garde les labels par défaut ("Temps" et "Y").
            donnees = brut

        # Colonne 0 = temps, colonne 1 = Y. On force en nombres (float).
        self.t_exp = donnees.iloc[:, 0].astype(float).values
        self.Y_exp = donnees.iloc[:, 1].astype(float).values

        # Récapitulatif propre sous forme de DataFrame, qu'on affiche.
        # On y réutilise les en-têtes lus (ou les labels par défaut).
        recap = pd.DataFrame({self.label_t: self.t_exp, self.label_Y: self.Y_exp})
        print(f"{len(self.t_exp)} points lus.")
        try:
            # display() rend un joli tableau dans un notebook Jupyter.
            from IPython.display import display
            display(recap)
        except ImportError:
            # En dehors d'un notebook, un simple print suffit.
            print(recap)

        return recap

    # -----------------------------------------------------------------
    #  Estimation d'un point de départ raisonnable pour k, à partir des
    #  données elles-mêmes (plutôt qu'une constante arbitraire comme 0.1,
    #  qui peut être très loin du bon ordre de grandeur et faire échouer
    #  ou ralentir la convergence de minimize).
    # -----------------------------------------------------------------
    def _k_depart_auto(self, ordre, Y0):
        """
        FR — Estime un k de départ en inversant analytiquement le modèle
        de l'ordre choisi, à partir d'un seul point expérimental (ni le
        tout premier, ni le tout dernier) pris comme référence :

        - ordre 0 : Y = Y0 - k t       =>  k = (Y0 - Y_n) / t_n
        - ordre 1 : Y = Y0 exp(-k t)   =>  k = -ln(Y_n / Y0) / t_n
        - ordre 2 : Y = Y0 / (1 + k t) =>  k = (Y0 / Y_n - 1) / t_n

        Si l'estimation échoue ou donne un k négatif ou nul (données trop
        bruitées, point de référence mal choisi...), on revient à 0.1.

        EN — Estimates a starting k by analytically inverting the chosen
        order's model at a single reference data point (neither the very
        first nor the very last). Falls back to 0.1 if the estimate fails
        or comes out non-positive.
        """
        # Point de référence : ~10% des données, jamais le premier point
        # (t=0 ne donne aucune information sur k).
        n = max(1, len(self.t_exp) // 10)
        t_n, Y_n = self.t_exp[n], self.Y_exp[n]

        if t_n == 0:
            return 0.1

        try:
            if ordre == 0:
                k0 = (Y0 - Y_n) / t_n
            elif ordre == 1:
                if Y_n <= 0:
                    return 0.1
                k0 = -np.log(Y_n / Y0) / t_n
            elif ordre == 2:
                if Y_n == 0:
                    return 0.1
                k0 = (Y0 / Y_n - 1) / t_n
            else:
                return 0.1
        except (ValueError, ZeroDivisionError):
            return 0.1

        return k0 if k0 > 0 else 0.1

    # =================================================================
    #  2) FAIRE LE FIT : chercher le meilleur k (minimiser le RMSE)
    # =================================================================
    def fit(self, ordre=1, Y0=None, k_depart=None):
        """
        FR — Trouve la valeur de k qui minimise le RMSE pour l'ordre choisi.

        Si Y0 n'est pas donné, on le prend égal à la première valeur
        mesurée. Sinon, on utilise la valeur fournie par l'utilisateur.

        Si k_depart n'est pas donné (défaut), il est estimé automatiquement
        à partir des données par _k_depart_auto() (voir cette méthode) au
        lieu d'une constante arbitraire — utile si k est très différent de
        0.1 en ordre de grandeur, ce qui pouvait auparavant ralentir ou
        faire échouer la convergence. Vous pouvez toujours imposer votre
        propre valeur de départ en passant k_depart explicitement.

        Calcule aussi le temps de demi-réaction t_1/2 (stocké dans
        self.t_half et affiché), selon la formule propre à l'ordre choisi
        (voir section 3.4 "Theory" de Kinetics.md) :
          - ordre 0 : t_1/2 = Y0 / (2*k)
          - ordre 1 : t_1/2 = ln(2) / k
          - ordre 2 : t_1/2 = 1 / k

        Retourne : (k_optimal, rmse_au_minimum)

        EN — Finds the value of k that minimizes the RMSE for the chosen
        order.

        If Y0 is not given, it is taken equal to the first measured value.
        Otherwise, the user-provided value is used.

        If k_depart is not given (default), it is estimated automatically
        from the data by _k_depart_auto() instead of an arbitrary constant —
        useful when k is far from 0.1 in order of magnitude, which could
        previously slow down or fail convergence. You can still force your
        own starting value by passing k_depart explicitly.

        Also computes the half-life t_1/2 (stored in self.t_half and
        printed), using the order-specific formula (see section 3.4
        "Theory" in Kinetics.md):
          - order 0: t_1/2 = Y0 / (2*k)
          - order 1: t_1/2 = ln(2) / k
          - order 2: t_1/2 = 1 / k

        Returns: (k_optimal, rmse_at_minimum)
        """
        # Valeur par défaut de Y0 : le premier point mesuré.
        if Y0 is None:
            Y0 = self.Y_exp[0]

        # Valeur de départ de k : estimée à partir des données si non fournie.
        if k_depart is None:
            k_depart = self._k_depart_auto(ordre, Y0)

        # minimize cherche le k qui rend le RMSE le plus petit possible.
        #   - self._rmse    : la fonction à minimiser
        #   - x0=[k_depart] : valeur de départ de la recherche
        #   - args=(...)    : les autres arguments fixes passés à _rmse
        #   - bounds        : on impose k >= 0 (une vitesse n'est pas négative)
        #
        # Pourquoi x0=[k_depart] avec des crochets, et pas juste k_depart ?
        # minimize est conçu pour optimiser plusieurs paramètres à la fois ; il
        # attend donc TOUJOURS une LISTE de valeurs de départ, même quand il n'y
        # a qu'un seul paramètre (ici k). On écrit donc [k_depart], c.-à-d.
        # [0.1], et non 0.1. C'est aussi pour cela que _rmse reçoit k sous forme
        # de tableau et fait k = k[0] (voir le commentaire dans _rmse).
        resultat = minimize(
            self._rmse,
            x0=[k_depart],
            args=(ordre, Y0),
            bounds=[(0, None)],
        )

        # On mémorise tout dans l'objet, pour que plot() puisse s'en servir.
        self.ordre = ordre
        self.Y0 = Y0
        self.k_opt = resultat.x[0]   # meilleure valeur de k
        self.rmse_min = resultat.fun # RMSE correspondant (le plus petit)

        # Temps de demi-réaction t_1/2, calculé à partir de k_opt et de la
        # formule propre à chaque ordre (voir section 3.4 "Theory" de Kinetics.md).
        if ordre == 0:
            self.t_half = self.Y0 / (2 * self.k_opt)
        elif ordre == 1:
            self.t_half = np.log(2) / self.k_opt
        elif ordre == 2:
            self.t_half = 1 / self.k_opt

        print(f"Ordre {ordre} : k optimal = {self.k_opt:.4e}"
              f"  (RMSE = {self.rmse_min:.4e})"
              f"  (t1/2 = {self.t_half:.4e})")
        return self.k_opt, self.rmse_min

    # =================================================================
    #  3) TRACER LE RÉSULTAT : comparer données et modèle ajusté
    #
    #  Deux graphiques côte à côte :
    #    - à gauche : les données et la courbe du modèle ajusté
    #    - à droite : le "paysage" du RMSE en fonction de k, qui montre
    #                 la vallée dont minimize a trouvé le fond.
    # =================================================================
    def plot(self):
        """FR — Trace la comparaison données / modèle et le paysage du RMSE.
        EN — Plots the data/model comparison and the RMSE landscape."""
        # On a besoin du résultat du fit. Si fit() n'a pas encore été appelé,
        # k_opt vaut None : on prévient l'utilisateur au lieu de planter.
        if self.k_opt is None:
            raise RuntimeError("Faites d'abord kb.fit(...) avant kb.plot().")

        figure, (gauche, droite) = plt.subplots(1, 2, figsize=(13, 5))

        # ---- Graphique de gauche : données + modèle ----
        # Grille de temps fine et régulière pour une courbe bien lisse.
        t_lisse = np.linspace(self.t_exp.min(), self.t_exp.max(), 400)
        Y_lisse = self._modele(t_lisse, self.k_opt, self.Y0, self.ordre)

        gauche.scatter(self.t_exp, self.Y_exp, color="black", marker="x",
                       label="Données expérimentales")
        gauche.plot(t_lisse, Y_lisse, color="red",
                    label=f"Modèle ordre {self.ordre} (k = {self.k_opt:.4e})")
        # On utilise les en-têtes lus dans le fichier comme labels d'axes
        # (ou les valeurs par défaut "Temps" et "Y" s'il n'y avait pas d'en-tête).
        gauche.set_xlabel(self.label_t)
        gauche.set_ylabel(self.label_Y)
        gauche.legend()
        gauche.grid(True)
        gauche.set_title('Données et modèle ajusté ("fitté")')

        # ---- Graphique de droite : le paysage du RMSE ----
        # On essaie plein de valeurs de k autour de l'optimum et on calcule
        # le RMSE pour chacune, afin de "voir" la vallée.
        k_essais = np.linspace(max(self.k_opt * 0.1, 1e-9), self.k_opt * 3, 300)
        rmse_essais = [self._rmse(k, self.ordre, self.Y0) for k in k_essais]

        droite.plot(k_essais, rmse_essais, color="blue", label="RMSE(k)")
        # On marque le minimum trouvé par minimize d'un gros point rouge.
        droite.scatter([self.k_opt], [self.rmse_min], color="red", zorder=5,
                       label=f"minimum (k = {self.k_opt:.4e})")
        droite.axvline(self.k_opt, color="red", linestyle="--", alpha=0.5)
        droite.set_xlabel("k")
        droite.set_ylabel("RMSE")
        droite.legend()
        droite.grid(True)
        droite.set_title("Le paysage du RMSE : minimize en cherche le 'fond'")

        figure.tight_layout()
        plt.show()

class KORD:
    """
    Initialize the kinetic study with experimental data.
    Reaction: alpha A = beta B
    
    Args:
        - t_exp (array-like): Time values.
        - Y_exp (array-like): Measured physical quantity (Absorbance, Conductivity, etc.).

    Fixed Parameters:
        - alpha (float): Stoichiometric coefficient for A reactant (smallest positive integer). Default is 1.0.
        - beta (float): Stoichiometric coefficient for B product (smallest positive integer). Default is 1.0.
        - a0 (float): Initial concentration. Note: Y_theo is independent of a0 for Order 1. Default is 1.0.

    Adjustable Variables (Initial Guesses):
        - k_guess (float, optional): Initial estimate for the rate constant. Default is 0.01.
        - Y0_guess (float, optional): Initial estimate for the initial measured value (Y at t=0). if None, it will be initialized as Y_exp[0]
        - Yinf_guess (float, optional): Initial estimate for the final measured value (Y at infinity). if None, it will be initialized as Y_exp[-1]
        - binf_guess (float, optional): Initial estimate for the final concentration of B ([B] at infinity). if None, it will be initialized as [A](t=0), i.e. a0
        
    Other Args:
        - t_simul_max (float): the maximum time duration for theoretical simulations.
          It defines the range of the time vector [0, t_simul_max}] used to simulate and plot kinetic curves with simulate_plot
        - verbose (bool): If True defaulk), enables detailed optimization logs (recommended).
        - headers (tuple of strings): headers of the t_exp and Y_exp arrays read in the excel file

    """
    
    def __init__(self, t_exp=None, Y_exp=None, headers=("Time / s", "Y"), a0=1.0, alpha=1.0, beta=1.0, k_guess=None,
                 Y0_guess = None, Yinf_guess = None, binf_guess=None, t_simul_max = 15, verbose=True):

        # Force conversion to float arrays to prevent 'object' dtype issues.
        # This ensures [0, 0.5, ...] (objects) become [0.0, 0.5, ...] (floats),
        # allowing NumPy mathematical functions (like np.exp) to operate correctly.
        # It is optional if a user wants to simulate a spectrum
        self.t_exp = np.array(t_exp, dtype=float) if t_exp is not None else None
        self.Y_exp = np.array(Y_exp, dtype=float) if Y_exp is not None else None
        self.headers = headers
        # Ensure fixed parameters are treated as native floats.
        # This prevents errors during optimization if values are passed as strings or tuples.
        self.a0 = float(a0)
        self.alpha = float(alpha)
        self.beta = float(beta)

        self.t_simul_max = t_simul_max

        # Logic for Guesses: Priority to provided values, then data, then defaults
        if Y0_guess is not None:
            self.Y0_guess = Y0_guess
        elif self.Y_exp is not None:
            self.Y0_guess = self.Y_exp[0]
        else:
            self.Y0_guess = 0.0 # Default fallback if no data and no guess (this is for a simulation plot)

        if Yinf_guess is not None:
            self.Yinf_guess = Yinf_guess
        elif self.Y_exp is not None:
            self.Yinf_guess = self.Y_exp[-1]
        else:
            self.Yinf_guess = 1.0 # Default fallback (this is for a simulation plot)
            
        self.binf_guess = binf_guess if binf_guess is not None else self.a0*self.beta/self.alpha

        self.k_guess = k_guess
        
        self.results = {}
        self.verbose = verbose
        # Color mapping for orders
        self.order_colors = {0: 'blue', 1: 'red', 2: 'green'}
        self.ansi_colors = {0: "\033[94m", 1: "\033[91m", 2: "\033[92m"}
        self.reset = "\033[0m"
        # t_fin = self.a0 / (self.alpha * self.k_guess)
        # print(f"{t_fin=}")
        self._initialized = True   # voir __setattr__ ci-dessous

    def __setattr__(self, name, value):
        """
        FR — Protection contre les fautes de frappe sur un nom d'attribut
        (p. ex. `study.Yinfguess = ...` au lieu de `study.Yinf_guess = ...`) :
        une fois l'objet construit, on ne peut plus CRÉER un nouvel
        attribut — seuls ceux déjà définis dans __init__ peuvent être
        modifiés. Sans cette protection, Python créerait silencieusement
        l'attribut mal orthographié ; fit() continuerait de lire l'ancienne
        valeur de Yinf_guess sans jamais signaler l'erreur.

        EN — Guards against attribute-name typos (e.g.
        `study.Yinfguess = ...` instead of `study.Yinf_guess = ...`): once
        the object is built, no NEW attribute can be created — only ones
        already defined in __init__ can be changed. Without this, Python
        would silently create the misspelled attribute, and fit() would
        keep reading the old value of Yinf_guess with no error at all.
        """
        if getattr(self, '_initialized', False) and name not in self.__dict__:
            existing = ', '.join(sorted(k for k in self.__dict__ if not k.startswith('_')))
            raise AttributeError(
                f"KORD has no attribute '{name}' (possible typo?). "
                f"Existing attributes: {existing}"
            )
        object.__setattr__(self, name, value)

    def copy(self):
        """
        FR — Retourne une copie indépendante de cette étude KORD : les
        données, les guesses et les résultats déjà en cache (self.results)
        sont dupliqués, pas partagés. Un fit() ultérieur sur la copie (ou
        sur l'original) ne modifie pas l'autre.

        Utile pour comparer, par exemple, un ajustement avec Y0 figé et un
        ajustement libre sur les mêmes données, sans que l'un écrase le
        cache de l'autre (voir superpose_plots()) :

        >>> study_free = KORD.from_spreadsheet(file, exp_number=1)
        >>> study_fixed = study_free.copy()
        >>> study_fixed.fit(order=1, fix_Y0=True)
        >>> superpose_plots([study_free, study_fixed], order=1)

        EN — Returns an independent copy of this KORD study: the data, the
        guesses, and any results already cached (self.results) are
        duplicated, not shared. A later fit() on the copy (or on the
        original) does not affect the other.

        Useful to compare, e.g., a fit with Y0 frozen against a free fit of
        the same data, without either one overwriting the other's cache
        (see superpose_plots()):

        >>> study_free = KORD.from_spreadsheet(file, exp_number=1)
        >>> study_fixed = study_free.copy()
        >>> study_fixed.fit(order=1, fix_Y0=True)
        >>> superpose_plots([study_free, study_fixed], order=1)
        """
        return _copy.deepcopy(self)

    @staticmethod
    def load_from_spreadsheet(file_path=None, exp_number=1, sheet_name=0, show_data=True, colab=False):
        """
        Static method to extract data from a spreadsheet file (.xlsx, .xls, .ods, ...
        anything pandas.read_excel can open — a local path, or any path/URL
        pandas.read_excel accepts).
        Selects the pair of columns (t, Y) corresponding to the experiment number.
        Also loads parameters (a0, alpha, beta)

        Format:
            - Row 1: Headers for t and Y
            - Row 2: [A]0 value (in the Y column)
            - Row 3: alpha value (in the Y column)
            - Row 4: beta value (in the Y column)
            - Row 5+: [t, Y] data points

        Args:
            file_path: path (or URL) of the spreadsheet to read. Not needed if
                colab=True (the file is picked via the Colab upload widget).
            colab: if True, opens the Google Colab upload button and reads the
                chosen file instead of file_path. Default False.
        """
        # 0. Google Colab mode: open the file picker, read the uploaded file's name.
        if colab:
            file_path = colab_upload()   # Colab upload button, see colab_utils.py

        # 1. Check if file exists
        if not os.path.exists(file_path):
            print(f"❌ Error: The file '{file_path}' was not found.")
            return None

        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        except Exception as e:
            print(f"❌ Error while reading the spreadsheet: {e}")
            return None

        idx_t, idx_Y = 2*(exp_number-1), 2*(exp_number-1)+1

        # --- Parameter Extraction (Now looking in the Y column: idx_Y) ---
        def parse_param(val):
            if isinstance(val, str):
                match = re.search(r"(\d+\.?\d*)", val)
                return float(match.group(1)) if match else 1.0
            return float(val) if pd.notnull(val) else 1.0

        total_cols = len(df.columns)
        num_experiments = total_cols // 2
        print(f"Experiments detected: {num_experiments}")

        # Parameters are expected in rows 2, 3, and 4 of Excel (indices 0, 1, 2)
        a0    = parse_param(df.iloc[0, idx_Y])
        alpha = parse_param(df.iloc[1, idx_Y])
        beta  = parse_param(df.iloc[2, idx_Y])

        # --- Format sanity check: rows 2-4 of the t column should be empty or
        # text labels (see Format above) — a genuine number there usually means
        # the data doesn't follow the expected layout (e.g. real t,Y data was
        # put starting at row 2 instead of row 5).
        t_col_rows_2_4 = df.iloc[0:3, idx_t]
        suspicious = t_col_rows_2_4.apply(lambda v: pd.notnull(v) and not isinstance(v, str))
        if suspicious.any():
            print(f"⚠️  Warning: rows 2-4 of column '{df.columns[idx_t]}' (the t column) "
                  f"contain numeric value(s): {list(t_col_rows_2_4[suspicious])}. Expected "
                  f"format is row 1 = headers, rows 2-4 = [A]0/alpha/beta (in the Y column "
                  f"only — the t column should be blank or a text label there), row 5+ = "
                  f"data. It does not seem to be the case here; check your data.")

        # --- Data Extraction (From index 3 onwards) ---
        data = df.iloc[3:, [idx_t, idx_Y]].dropna()
        
        label_t = KORD._clean_pandas_suffix(df.columns[idx_t])
        label_Y = KORD._clean_pandas_suffix(df.columns[idx_Y])
        data.columns = [label_t, label_Y]

        print(f"✅ Loaded: {label_Y} (Exp {exp_number})")
        print(f"   [Parameters from {label_Y}] a0: {a0:.4e} mol.L-1 | alpha: {alpha} | beta: {beta}\n")
        
        if show_data:
            from IPython.display import display
            display(data)
        
        return data.iloc[:, 0].values, data.iloc[:, 1].values, (label_t, label_Y), (a0, alpha, beta)

    @staticmethod
    def load_from_excel(file_path=None, exp_number=1, sheet_name=0, show_data=True, colab=False):
        """
        Deprecated alias for :meth:`load_from_spreadsheet` (kept so existing
        scripts/notebooks that call ``load_from_excel`` keep working — the
        method always read any spreadsheet format pandas.read_excel supports,
        including .ods, so the old name was misleading). Prefer
        ``load_from_spreadsheet`` in new code.
        """
        print("⚠️  KORD.load_from_excel is deprecated, use KORD.load_from_spreadsheet instead "
              "(same behaviour — it never was Excel-only).")
        return KORD.load_from_spreadsheet(file_path, exp_number, sheet_name=sheet_name,
                                           show_data=show_data, colab=colab)

    @classmethod
    def from_spreadsheet(cls, file_path=None, exp_number=1, sheet_name=0, show_data=True,
                          colab=False, **kwargs):
        """
        Convenience constructor: reads one experiment from a spreadsheet with
        load_from_spreadsheet(), then builds and returns a KORD instance from it,
        in a single call.

        Equivalent to::

            t, Y, headers, (a0, alpha, beta) = KORD.load_from_spreadsheet(
                file_path, exp_number, sheet_name, show_data, colab)
            study = KORD(t, Y, headers, a0=a0, alpha=alpha, beta=beta, **kwargs)

        Any extra keyword argument (k_guess, Y0_guess, Yinf_guess,
        binf_guess, t_simul_max, verbose, ...) is forwarded to KORD.__init__.
        colab=True opens the Google Colab upload widget instead of reading
        file_path (same as in load_from_spreadsheet / kbasic.read).

        Use load_from_spreadsheet() directly instead when you need to
        inspect or modify the loaded data/parameters before creating the
        KORD instance.
        """
        t, Y, headers, (a0, alpha, beta) = cls.load_from_spreadsheet(
            file_path, exp_number, sheet_name=sheet_name, show_data=show_data, colab=colab)
        return cls(t, Y, headers, a0=a0, alpha=alpha, beta=beta, **kwargs)

    @staticmethod
    def _clean_pandas_suffix(name):
        """
        Safely removes the '.1', '.2' suffixes added by Pandas for duplicate names.
        Only matches a dot followed by digits at the VERY END of the string.
        Example: 'mol.L-1.1' -> 'mol.L-1'
        """
        return re.sub(r'\.\d+$', '', str(name))
        
    def simulate_plot(self, order=None, show_experimental=True, save_img=None):
        """
        Plots the initial guesses for one or all three kinetic orders (0, 1,
        2), for example to visualize the starting point of a fit before
        calling fit().

        Args:
            order (int or None, optional): which order's guess to plot. None
                (the default) plots all three (0, 1, 2), as before. Pass 0,
                1 or 2 to show only that one order's guess curve.
            show_experimental (bool, optional): if True (the default),
                overlays the measured data (self.t_exp, self.Y_exp) as
                scatter points, so the guess can be checked visually against
                the actual data instead of only against the other guesses.
                Set to False to recover the old guesses-only behaviour.

        If save_img is provided, saves the plot (png, svg, jpg, pdf according to the extension).
        Vectorial svg is recommended
        """
        plt.figure(figsize=(10, 6))

        # 2. Generate smooth time vector for simulation
        t_sim = np.linspace(0, self.t_simul_max, 500)

        # 3. Define the models to loop through
        models = {0: self.Y0_theo, 1: self.Y1_theo, 2: self.Y2_theo}
        orders_to_plot = [0, 1, 2] if order is None else [order]

        # Overlay the experimental data first, so the guess curve(s) are
        # drawn on top of it.
        if show_experimental and self.t_exp is not None:
            plt.scatter(self.t_exp, self.Y_exp, label="Experimental",
                         color='black', marker="x", s=35, alpha=0.7, zorder=5)

        for o in orders_to_plot:
            func = models[o]
            # Use the color defined in your __init__
            color = self.order_colors[o]

            # Simulate using the guess values from __init__
            if o != 1:
                Y_sim = func(t_sim,
                             k=self.k_guess,
                             Y0=self.Y0_guess,
                             Yinf=self.Yinf_guess,
                             binf=self.binf_guess)
            else:
                Y_sim = func(t_sim,
                             k=self.k_guess,
                             Y0=self.Y0_guess,
                             Yinf=self.Yinf_guess)


            plt.plot(t_sim, Y_sim,
                     label=f"Guess Order {o}",
                     color=color, lw=2.5, linestyle='-')

        # Formatting
        plt.xlabel(self.headers[0])
        plt.ylabel(self.headers[1])
        plt.title(f"Initial Kinetic Guesses for {self.headers[1]}")
        
        # Add a text box with the guess values for clarity
        guess_text = (f"Guesses:\n"
                      f"k: {self.k_guess:.2e}\n"
                      f"Y0: {self.Y0_guess:.4f}\n"
                      f"Yinf: {self.Yinf_guess:.4f}\n"
                      f"binf: {self.binf_guess:.2e}")
        
        plt.gca().text(0.05, 0.95, guess_text, transform=plt.gca().transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        plt.legend(loc='lower right')
        plt.grid(True, linestyle=':', alpha=0.6)
        if save_img:
            # Check if the file already exists to prevent accidental overwrite
            if os.path.exists(save_img):
                # Prompt the user for confirmation
                print(f"{color.RED}{hl.BOLD}⚠️ File '{save_img}' already exists. Overwrite? (y/n): {color.OFF}", end="")
                # Then call input
                response = input().lower().strip()
                if response != 'y':
                    print("💾 Save cancelled. Plot not saved.")
                    plt.show() # Display anyway without saving
                    return # Exit the save logic

            # Security: ensure the target directory exists (e.g., if save_img="export/graph.svg")
            directory = os.path.dirname(save_img)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            # Universal save (supports SVG, PNG, PDF, etc. based on extension)
            # bbox_inches='tight' prevents labels or secondary axes from being cropped
            plt.savefig(save_img, dpi=300, bbox_inches='tight')
            print(f"💾 Plot saved as: {hl.BOLD}{save_img}{hl.bold}")
            
        plt.show()        
        
    def Y0_theo(self, t, k, Y0, Yinf, binf):
        """
        Model for Order 0 kinetics
        """
        return np.where(t <= binf/(self.beta*k), Y0 + self.beta * k*t * (Yinf-Y0) / binf, Yinf)
        
    def Y1_theo(self, t, k, Y0, Yinf):
        """Model for Order 1 kinetics"""
        return Yinf + np.exp(-self.alpha * k * t) * (Y0 - Yinf)

    def Y2_theo(self, t, k, Y0, Yinf, binf):
        """Model for Order 2 kinetics"""
        # return np.where(t !=0, Y0 + (Yinf - Y0)/(1 + self.beta/(binf*self.alpha**2*k*t)), Y0)
        tterm = binf*self.alpha**2*k*t
        return Y0 + (Yinf-Y0)*tterm/(tterm+self.beta)

    def concentrations(self,order,t,binf,k):
        """
        Calculate the time-dependent concentrations of reactant A and product B.

        This method derives the initial concentration (a0) from the fitted 
        final product concentration (binf) using the stoichiometric ratio alpha/beta.
        It then computes the concentration profiles based on the integrated 
        rate laws for the specified kinetic order.

        Args:
            order (int): Kinetic order (0, 1, or 2).
            t (array-like): Time vector for the calculation.
            binf (float): Final concentration of product B at t=infinity.
            k (float): Rate constant.

        Returns:
            tuple: (a, b) where 'a' is the concentration array of reactant A 
                   and 'b' is the concentration array of product B.

        Notes:
            - Order 0: Linear decay/formation until reactant exhaustion.
            - Order 1: Exponential decay/formation.
            - Order 2: Hyperbolic progression based on the 1/[A] linear relationship.
        """
        import numpy as np
        a0 = binf*self.alpha/self.beta
        if order == 0:
            a = np.where(t <= a0/(self.alpha*k), a0 - self.alpha*k*t, 0)
            b = np.where(t <= binf/(self.beta*k), self.beta*k*t, binf)
        elif order == 1:
            a = a0 * np.exp(-self.alpha*k*t)
            b = binf * (1-np.exp(-self.alpha*k*t))
        elif order == 2:
            a = 1 / (1/a0 + self.alpha*k*t)
            b = (binf*self.alpha)**2 * k*t/(self.beta + binf*self.alpha**2 * k*t)
        return a, b
        
    def halflife(self,order, binf, k):
        """
        Calculate the half-life (t1/2) for the reaction based on the kinetic order.

        The half-life represents the time required for the reactant concentration 
        to reach half of its initial value (a0/2). 

        Args:
            order (int): Kinetic order (0, 1, or 2).
            k (float): The optimized or guessed rate constant.

        Returns:
            float: The calculated half-life value.

        Notes:
            - Order 0: t1/2 = a0 / (2 * alpha * k) -> Dependent on a0.
            - Order 1: t1/2 = ln(2) / (alpha * k) -> Independent of a0.
            - Order 2: t1/2 = 1 / (alpha * k * a0) -> Inversely proportional to a0.
        """
        import numpy as np
        a0 = binf*self.alpha/self.beta
        if (order == 0):
            return a0/(2*self.alpha*k)
        elif (order == 1):
            return np.log(2)/(self.alpha*k)
        elif (order == 2):
            return 1/(self.alpha*k*a0)
    
    def fit(self, order=1, fix_Y0=False, fix_Yinf=False):
        """
        Fits the selected kinetic model (Order 0, 1, or 2) to the experimental data.

        This method uses non-linear least squares (scipy.optimize.curve_fit) to
        optimize the rate constant (k), and - unless fixed, see below - the initial
        and final physical property values (Y0, Yinf). The final product
        concentration (binf) is never a free fit parameter: it is always derived
        from mass balance as binf = a0*beta/alpha (see the Notes below for why).
        Physical constraints are applied via parameter bounds to ensure k remains
        positive.

        It includes automated 'Smart Guessing' for k, based on initial rates and computes
        information criteria (AIC/BIC) for model selection.

        The starting guesses (k_guess, Y0_guess, Yinf_guess) are NOT
        passed here: they are read from the attributes set when the KORD instance
        was created (self.k_guess, self.Y0_guess, self.Yinf_guess).
        To change a guess, set it on the instance (or pass it to KORD(...) /
        KORD.from_spreadsheet(...) in the first place) before calling fit(), e.g.
        ``study.Y0_guess = 1.40`` then ``study.fit(order=1, fix_Y0=True)``.

        Args:
            order (int, optional): Kinetic order of the reaction (0, 1, or 2). Defaults to 1.
            fix_Y0 (bool, optional): If True, Y0 is held fixed at Y0_guess (normally the
                first measured value) instead of being optimized. Use this when you trust
                the measured starting point and want to stop the fit from drifting it away
                to improve the apparent fit. Defaults to False (Y0 is optimized, as before).
            fix_Yinf (bool, optional): If True, Yinf is held fixed at Yinf_guess instead of
                being optimized. Use this when the final value is known on physical/chemical
                grounds (e.g. a non-absorbing product, Yinf = 0) and you don't want the
                optimizer to report a spurious non-zero plateau. Defaults to False (Yinf is
                optimized, as before).

        Returns:
            dict: A dictionary containing the optimized parameters ('k', 'Y0',
                  'Yinf'), the derived final product concentration ('binf'), and
                  statistical metrics ('RMSE', 'R2', 't_half'). Returns None if
                  optimization fails to converge.

        Notes:
            - binf is never fitted: it is always derived as binf = a0*beta/alpha,
              for every order, the same formula Order 1 already relied on. This
              is not an approximation to simplify the model — it is required for
              k and binf to be individually meaningful. For Orders 0 and 2, if k
              AND binf were both left free, the model's shape would depend on
              them only through one combined ratio (beta*k/binf for order 0,
              beta/(binf*alpha**2*k) for order 2): infinitely many (k, binf)
              pairs then give an identical curve and an identical RMSE, so
              curve_fit would converge to an arbitrary point on that family
              (whichever one is closest to binf_guess) rather than to a unique,
              physically meaningful answer. Deriving binf from mass balance
              removes that degeneracy.
            - fix_Y0 / fix_Yinf let you turn off the optimization of Y0 and/or Yinf
              independently of each other, for any order. AIC/BIC account for the
              resulting (smaller) number of free parameters, so model comparison
              across orders remains fair even when parameters are fixed.
        """
        models = {0: self.Y0_theo, 1: self.Y1_theo, 2: self.Y2_theo}
        func = models[order]

        # smart guess for k
        n = max(2, len(self.t_exp) // 10)
        delta_G = abs(self.Y_exp[n] - self.Y_exp[0])
        delta_t = self.t_exp[n] - self.t_exp[0]
        initial_slope = delta_G / delta_t if delta_t > 0 else 1e-5

        # 2. Apply the smart guess logic
        delta_G_total = abs(self.Yinf_guess - self.Y0_guess)
        if delta_G_total == 0: delta_G_total = 1.0

        norm_rate = initial_slope / delta_G_total

        if order == 0:
            k_start = (norm_rate * self.a0) / self.alpha
        elif order == 1:
            k_start = norm_rate / self.alpha
        elif order == 2:
            k_start = norm_rate / (self.alpha * self.a0)

        k_start = max(k_start, 1e-15) # Safety floor

        # Use the smart guess if k_guess wasn't explicitly provided by user
        final_k_guess = self.k_guess if self.k_guess is not None else k_start

        # --- Build the list of parameters, splitting it into the ones that
        # curve_fit is allowed to optimize ("free") and the ones held constant
        # ("fixed": Y0 and/or Yinf, on request). k is always free. binf is
        # NEVER a free fit parameter, for any order: mass balance already
        # fixes it at a0*beta/alpha (same formula used for order 1), so
        # letting it float too (as orders 0/2 used to) only adds a phantom,
        # non-identifiable degree of freedom — for a given k, the fit curve
        # depends on binf only through one combined ratio, so curve_fit can
        # trade k against binf along a whole family of equally-good pairs
        # without changing RMSE at all. Deriving binf everywhere removes
        # that degeneracy and keeps p (used in AIC/BIC) an honest count of
        # 3 free parameters for every order.
        binf_fixed = self.a0 * self.beta / self.alpha

        param_specs = [
            ('k',    final_k_guess,    False,    (0, np.inf)),
            ('Y0',   self.Y0_guess,    fix_Y0,   (-np.inf, np.inf)),
            ('Yinf', self.Yinf_guess,  fix_Yinf, (-np.inf, np.inf)),
        ]

        fixed_values = {name: guess for name, guess, fixed, _ in param_specs if fixed}
        free_specs = [(name, guess, bounds) for name, guess, fixed, bounds in param_specs if not fixed]
        free_names = [name for name, _, _ in free_specs]
        p0 = [guess for _, guess, _ in free_specs]
        lower_bounds = [b[0] for _, _, b in free_specs]
        upper_bounds = [b[1] for _, _, b in free_specs]

        def wrapped(t, *free_params):
            kwargs = dict(zip(free_names, free_params))
            kwargs.update(fixed_values)
            if order != 1:
                return func(t, k=kwargs['k'], Y0=kwargs['Y0'], Yinf=kwargs['Yinf'], binf=binf_fixed)
            return func(t, k=kwargs['k'], Y0=kwargs['Y0'], Yinf=kwargs['Yinf'])

        if self.verbose:
            c = self.ansi_colors[order]
            print(f"{c}{hl.BOLD}--- Order {order} ---{self.reset}")
            tags = {'Y0': ' (fixed)' if fix_Y0 else '', 'Yinf': ' (fixed)' if fix_Yinf else ''}

            # Bold the parameters that are actually optimized (free); leave
            # the ones held fixed (fix_Y0/fix_Yinf) in normal weight, so a
            # glance at GUESS/OPTIM shows which numbers are allowed to move.
            def _field(label, value, spec, frozen):
                text = f"{label}: {value:{spec}}{tags.get(label, '')}"
                return text if frozen else f"{hl.BOLD}{text}{self.reset}"

            if order != 1:
                print(f"  GUESS: {_field('k', final_k_guess, '.2e', False)} | "
                      f"{_field('Y0', self.Y0_guess, '.4f', fix_Y0)} | "
                      f"{_field('Yinf', self.Yinf_guess, '.4f', fix_Yinf)} | "
                      f"{_field('binf', binf_fixed, '.3e', True)}")
            else:
                print(f"  GUESS: {_field('k', final_k_guess, '.2e', False)} | "
                      f"{_field('Y0', self.Y0_guess, '.4f', fix_Y0)} | "
                      f"{_field('Yinf', self.Yinf_guess, '.4f', fix_Yinf)}")

        try:
            if free_specs:
                popt_free, _ = curve_fit(wrapped, self.t_exp, self.Y_exp, p0=p0,
                                          bounds=(lower_bounds, upper_bounds))
            else:
                popt_free = np.array([])

            full = dict(zip(free_names, popt_free))
            full.update(fixed_values)

            k_opt, Y0_opt, Yinf_opt = full['k'], full['Y0'], full['Yinf']
            binf_opt = binf_fixed

            if order != 1:
                Y_theo = func(self.t_exp, k=k_opt, Y0=Y0_opt, Yinf=Yinf_opt, binf=binf_opt)
            else:
                Y_theo = func(self.t_exp, k=k_opt, Y0=Y0_opt, Yinf=Yinf_opt)

            RMSE = rmse(self.Y_exp, Y_theo)
            R2 = r2(self.Y_exp, Y_theo)

            # t1/2 calculation
            t_half = self.halflife(order, binf_opt, k_opt)

            # AIC & BIC calculations
            # n: number of data points, p: number of FREE (actually optimized) parameters
            n = len(self.Y_exp)
            p = len(popt_free)

            # Using the simplified formula for AIC/BIC (assuming Gaussian errors)
            # RSS (Residual Sum of Squares) = RMSE^2 * n
            aic = n * np.log(RMSE**2) + 2 * p
            bic = n * np.log(RMSE**2) + p * np.log(n)

            if self.verbose:
                # Aligned exactly with the GUESS print for easy comparison;
                # same bold-if-free convention as GUESS above.
                if order != 1:
                    print(f"  OPTIM: {_field('k', k_opt, '.2e', False)} | "
                          f"{_field('Y0', Y0_opt, '.4f', fix_Y0)} | "
                          f"{_field('Yinf', Yinf_opt, '.4f', fix_Yinf)} | "
                          f"{_field('binf', binf_opt, '.3e', True)}")
                else:
                    print(f"  OPTIM: {_field('k', k_opt, '.2e', False)} | "
                          f"{_field('Y0', Y0_opt, '.4f', fix_Y0)} | "
                          f"{_field('Yinf', Yinf_opt, '.4f', fix_Yinf)}")
                print(f"  ✅ RMSE: {RMSE:.2e}")
                print(f"       R2: {R2:.2f}")
                print(f"      AIC: {aic:.2f} (lower is better)")
                print(f"      BIC: {bic:.2f} (more strict on complexity)")

            self.results[order] = {
                'k': k_opt, 'Y0': Y0_opt, 'Yinf': Yinf_opt, 'binf': binf_opt,
                'RMSE': RMSE, 'R2': R2, 'AIC': aic, 'BIC': bic,
                't_half': t_half,
                'fix_Y0': fix_Y0, 'fix_Yinf': fix_Yinf,
            }
            return self.results[order]
        except Exception as e:
            print(f"Could not fit order {order}: {e}")
            return None

    def plot_all_fits(self, save_img=None, fix_Y0=False, fix_Yinf=False):
        """
        Plots experimental data and all three kinetic models for visual comparison.
        If save_img is provided, saves the plot (png, svg, jpg, pdf according to the extension).
        Vectorial svg is recommended.

        fix_Y0 / fix_Yinf are forwarded to fit() for any order not already present
        in self.results, so you can get all three comparison plots fitted with Y0
        and/or Yinf held fixed (see fit()'s docstring).

        An order already present in self.results (from an earlier fit() call)
        is NOT re-fitted here, even if fix_Y0/fix_Yinf passed to this plot
        differ from the ones that produced the cached fit — your earlier,
        explicit choice is respected rather than silently overwritten. If
        that creates a mismatch (some orders fitted under different
        fix_Y0/fix_Yinf settings than the others), a warning table is
        printed, since comparing orders fitted under different constraints
        can be misleading.
        """
        mosaic = [['fit','o0'],
                  ['fit','o1'],
                  ['fit','o2']]
        cgraph = ['o0', 'o1', 'o2']
        background_graph = ['#dbeeff', '#ffe4e4', '#d2ffdb']
        Fig, Graph = plt.subplot_mosaic(mosaic, figsize=(14,10), gridspec_kw=dict(width_ratios = [1.4, 1]))

        # --- Check, before fitting/plotting anything, whether every order
        # will end up under the same fix_Y0/fix_Yinf conditions. An order
        # already cached in self.results keeps whatever conditions it was
        # originally fitted with (it is not re-fitted below); only orders
        # missing from self.results get fitted now, with the fix_Y0/fix_Yinf
        # passed to this call.
        conditions = []
        for order in [0, 1, 2]:
            if order in self.results:
                conditions.append((order, self.results[order]['fix_Y0'],
                                    self.results[order]['fix_Yinf'], 'cached'))
            else:
                conditions.append((order, fix_Y0, fix_Yinf, 'fitted now'))

        mismatched = [c for c in conditions
                      if c[3] == 'cached' and (c[1] != fix_Y0 or c[2] != fix_Yinf)]
        if mismatched:
            print(f"⚠️  Warning: not every order is fitted under the same fix_Y0/fix_Yinf "
                  f"conditions (requested here: fix_Y0={fix_Y0}, fix_Yinf={fix_Yinf}) — "
                  f"cached fit(s) were kept as-is, not re-fitted. Comparing orders on "
                  f"this plot may be misleading:\n")
            print(f"  {'Order':<7}{'fix_Y0':<9}{'fix_Yinf':<10}{'Source'}")
            print(f"  {'-----':<7}{'------':<9}{'--------':<10}{'------'}")
            for order, g0, ginf, src in conditions:
                print(f"  {order:<7}{str(g0):<9}{str(ginf):<10}{src}")
            print()

        ## Fit ###############
        Graph['fit'].scatter(self.t_exp, self.Y_exp, label=f"Experimental ($a_0$ = {self.a0:.2e})", color='black', marker="x", s=35, alpha=0.7)

        t_smooth = np.linspace(self.t_exp.min(), self.t_exp.max(), 500)
        models = {0: self.Y0_theo, 1: self.Y1_theo, 2: self.Y2_theo}

        for order in [0, 1, 2]:
            if order not in self.results:
                self.fit(order, fix_Y0=fix_Y0, fix_Yinf=fix_Yinf)

            res = self.results[order]

            if order != 1:
                Y_smooth = models[order](t_smooth, res['k'], res['Y0'], res['Yinf'], res['binf'])
            else:
                Y_smooth = models[order](t_smooth, res['k'], res['Y0'], res['Yinf'])
            
            Graph['fit'].plot(t_smooth, Y_smooth, 
                     label=f"Order {order}. $k_\mathrm{{app}}$ = {res['k']:.4e} (RMSE: {res['RMSE']:.2e})", 
                     color=self.order_colors[order], lw=2)
            a, b = self.concentrations(order,t_smooth,res['binf'],res['k'])
            Graph[cgraph[order]].plot(t_smooth,a,linestyle="-",marker="",markersize=12,label="$a(t)$",linewidth=2)
            Graph[cgraph[order]].plot(t_smooth,b,linestyle="-",marker="",markersize=12,label="$b(t)$",linewidth=2)
            Graph[cgraph[order]].set_xlim(left=t_smooth[0], right=None)
            binf = res['binf']
            Graph[cgraph[order]].set_title(f"order {order}. $t_{{1/2}}$ = {res['t_half']:.1f}. $b_{{\infty}}$ = {binf:.2e} mol.L$^{-1}$",fontweight="bold",color='blue',fontsize=10)
            Graph[cgraph[order]].legend(fontsize=12)
            Graph[cgraph[order]].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
            Graph[cgraph[order]].set_facecolor(background_graph[order])
            # Add a vertical line at t1/2 on each concentration graph
            Graph[cgraph[order]].axvline(res['t_half'], color=self.order_colors[order], linestyle='--', alpha=0.6)
            

        Graph['o1'].set_ylabel("C / mol.L$^{-1}$",size=12,weight='bold',color='b')
        Graph['o2'].set_xlabel(f"Time",size=12,weight='bold',color='b')

        # Add horizontal lines for the BEST model on the "fit" graph
        best_order = self.get_best_order(verbose=False)
        best_res = self.results[best_order]
        best_color = self.order_colors[best_order]
        
        Graph['fit'].axhline(best_res['Y0'], color=best_color, linestyle='--', alpha=0.6)
        Graph['fit'].axhline(best_res['Yinf'], color=best_color, linestyle='--', alpha=0.6)
        Graph['fit'].yaxis.grid(
            True, 
            which='major', 
            linestyle='--',   # Ligne hachurée
            linewidth=0.8,    # Épaisseur
            color='gray',     # Couleur discrète
            alpha=0.5         # Transparence pour ne pas surcharger
        )
        # Désactiver la grille verticale (optionnel, pour garder le focus sur Y)
        Graph['fit'].xaxis.grid(False)
        
        # Add the second axis for the fitted values
        ax2 = Graph['fit'].twinx()
        ax2.set_ylim(Graph['fit'].get_ylim()) # Keep scales aligned
        ax2.set_yticks([best_res['Y0'], best_res['Yinf']])
        ax2.set_yticklabels([f"Y0_fit={best_res['Y0']:.3f}", f"Yinf_fit={best_res['Yinf']:.3f}"])
        ax2.tick_params(axis='y', labelcolor=best_color)
        
        Graph['fit'].set_xlabel("Time",size=12,fontweight='bold',color='b')
        Graph['fit'].set_ylabel("Y property",size=12,fontweight='bold',color='b')
        Graph['fit'].set_title(f"KORD Kinetic Models Comparison (0, 1, 2). Label exp = {self.headers[1]}")
        
        plt.setp(Graph['fit'].get_xticklabels(), fontsize=12, fontweight='bold')
        plt.setp(Graph['fit'].get_yticklabels(), fontsize=12, fontweight='bold')
        Graph['fit'].legend()

        plt.tight_layout()
        
        if save_img:
            # Check if the file already exists to prevent accidental overwrite
            if os.path.exists(save_img):
                # Prompt the user for confirmation
                print(f"{color.RED}{hl.BOLD}⚠️ File '{save_img}' already exists. Overwrite? (y/n): {color.OFF}", end="")
                # Then call input
                response = input().lower().strip()
                if response != 'y':
                    print("💾 Save cancelled. Plot not saved.")
                    plt.show() # Display anyway without saving
                    return # Exit the save logic

            # Security: ensure the target directory exists (e.g., if save_img="export/graph.svg")
            directory = os.path.dirname(save_img)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            # Universal save (supports SVG, PNG, PDF, etc. based on extension)
            # bbox_inches='tight' prevents labels or secondary axes from being cropped
            plt.savefig(save_img, dpi=300, bbox_inches='tight')
            print(f"💾 Plot saved as: {hl.BOLD}{save_img}{hl.bold}")
        plt.show()
        
    def get_best_order(self, verbose=True):
        """Determines and prints the best model based on the lowest RMSE."""
        for i in [0, 1, 2]:
            if i not in self.results: self.fit(i)
            
        best_order = min(self.results, key=lambda x: self.results[x]['RMSE'])
        res = self.results[best_order]
        
        if verbose:
            # ANSI Escape sequences for color in terminal/notebook
            reset = self.reset
            color = self.ansi_colors[best_order]
    
            print(f"{hl.BOLD}{color}--- KORD CONCLUSION ---{hl.bold}")
            print(f"{hl.BOLD}Best model: ORDER {best_order}{hl.bold}")
            print(f"Initial concentration a0: {self.a0:.3e} mol.L-1")        
            
            if best_order != 1:
                print(f"Fitted final concentration (b_inf): {res['binf']:.3e} mol.L-1")
                print(f"a0 after binf = [B](inf): {res['binf']*self.alpha/self.beta:.3e} mol.L-1")
            print(f"alpha: {self.alpha}")
            print(f"beta: {self.beta}")
            print()
            print(f"{hl.BOLD}metrics{hl.bold}")
            print(f"RMSE: {res['RMSE']:.3f}")
            print(f"  R2: {res['R2']:.3f}")
            print()
            print(f"Y_exp(t=0): {self.Y_exp[0]:.3e}")
            print(f"Y_fit(t=0): {res['Y0']:.3e}")
            print(f"Y_fit(t=inf): {res['Yinf']:.3e}")
            print()
            print(f"k_fit: {res['k']:.3e}")
            print(f"t1/2: {res['t_half']:.3f}")
            print(f"------------------------{reset}")
            
            # Sort results by BIC to find the top two candidates
            sorted_orders = sorted(self.results.keys(), key=lambda x: self.results[x]['BIC'])
            best = sorted_orders[0]
            second = sorted_orders[1]
            
            delta_bic = self.results[second]['BIC'] - self.results[best]['BIC']
            
            # Scientific interpretation of Delta BIC
            if delta_bic < 2:
                verdict = "Weak/Inconclusive"
            elif delta_bic < 6:
                verdict = "Positive"
            elif delta_bic < 10:
                verdict = "Strong"
            else:
                verdict = "Decisive"
    
            if verbose:
                print(f"\n{hl.BOLD}--- Final Statistical Verdict ---{hl.bold}")
                print(f"Model Selection Confidence: {verdict}")
                print(f"ΔBIC (Best [order {best}] vs 2nd Best [order {second}]): {delta_bic:.2f}")
                print(f"--------------------------------{self.reset}")
                return
        else:
            return best_order


# =====================================================================
#  Comparing several KORD studies on the same graph
# =====================================================================
def superpose_plots(studies, order=1, show_experimental=True, labels=None,
                     fix_Y0=False, fix_Yinf=False, save_img=None):
    """
    FR — Superpose, sur un même graphe, la courbe **ajustée** (fit, pas
    guess) d'un seul ordre pour plusieurs instances KORD — typiquement la
    même expérience fittée dans des conditions différentes (p.ex. une étude
    avec Y0 figé, une autre où Y0 reste libre), pour voir directement
    l'effet de ce choix sur la courbe ajustée.

    Pour chaque étude, si `order` n'est pas déjà présent dans
    study.results, l'ajustement est fait maintenant avec fix_Y0/fix_Yinf
    (mêmes réglages que plot_all_fits()). Si `order` est déjà en cache
    (study.results) avec des réglages fix_Y0/fix_Yinf différents de ceux
    demandés ici, le cache n'est PAS écrasé — exactement le comportement de
    plot_all_fits() — mais un avertissement est affiché pour cette étude.
    La légende indique automatiquement, pour chaque étude, si Y0 et/ou Yinf
    ont été figés dans l'ajustement effectivement tracé.

    Args :
        studies (list[KORD]) : les instances KORD à comparer (peuvent être
            des fits de la même donnée, sous des réglages différents).
        order (int, optionnel) : l'ordre à tracer pour toutes les études
            (0, 1 ou 2). Par défaut 1.
        show_experimental (bool, optionnel) : si True (par défaut), superpose
            aussi les données mesurées de chaque étude.
        labels (list[str] ou None, optionnel) : légendes de base pour chaque
            étude ; par défaut "Study 1", "Study 2", ... (le statut
            fix_Y0/fix_Yinf est ajouté automatiquement).
        fix_Y0, fix_Yinf (bool, optionnel) : transmis à fit() pour toute
            étude dont `order` n'est pas déjà dans son .results.
        save_img : comme pour simulate_plot()/plot_all_fits().

    EN — Overlays, on a single graph, the **fitted** (not guess) curve of
    one chosen order for several KORD instances — typically the same
    experiment fitted under different conditions (e.g. one study with Y0
    frozen, another with Y0 left free), to see directly how that choice
    changes the fitted curve.

    For each study, if `order` is not already present in study.results, it
    is fitted now with fix_Y0/fix_Yinf (same settings as plot_all_fits()).
    If `order` is already cached in study.results under different
    fix_Y0/fix_Yinf settings than requested here, the cache is NOT
    overwritten — exactly like plot_all_fits() — but a warning is printed
    for that study. The legend automatically shows, for each study, whether
    Y0 and/or Yinf were held fixed in the fit actually plotted.

    Args:
        studies (list[KORD]): the KORD instances to compare (can be fits of
            the very same data, under different settings).
        order (int, optional): the order to plot for every study. Default 1.
        show_experimental (bool, optional): if True (the default), also
            overlay each study's measured data.
        labels (list[str] or None, optional): base legend labels; default
            "Study 1", "Study 2", ... (the fix_Y0/fix_Yinf status is added
            automatically).
        fix_Y0, fix_Yinf (bool, optional): forwarded to fit() for any study
            whose `order` isn't already in its .results.
        save_img: as in simulate_plot()/plot_all_fits().
    """
    plt.figure(figsize=(10, 6))
    models = {0: 'Y0_theo', 1: 'Y1_theo', 2: 'Y2_theo'}

    for i, study in enumerate(studies):
        base_label = labels[i] if labels else f"Study {i + 1}"

        if order in study.results:
            cached_fix_Y0 = study.results[order]['fix_Y0']
            cached_fix_Yinf = study.results[order]['fix_Yinf']
            if cached_fix_Y0 != fix_Y0 or cached_fix_Yinf != fix_Yinf:
                print(f"⚠️  {base_label}: using the cached order-{order} fit "
                      f"(fix_Y0={cached_fix_Y0}, fix_Yinf={cached_fix_Yinf}), "
                      f"not the fix_Y0={fix_Y0}, fix_Yinf={fix_Yinf} requested here.")
        else:
            study.fit(order, fix_Y0=fix_Y0, fix_Yinf=fix_Yinf)

        res = study.results[order]
        func = getattr(study, models[order])
        t_sim = np.linspace(study.t_exp.min(), study.t_exp.max(), 500)

        if order != 1:
            Y_sim = func(t_sim, k=res['k'], Y0=res['Y0'], Yinf=res['Yinf'], binf=res['binf'])
        else:
            Y_sim = func(t_sim, k=res['k'], Y0=res['Y0'], Yinf=res['Yinf'])

        tags = [tag for tag, fixed in (("Y0 fixed", res['fix_Y0']),
                                        ("Yinf fixed", res['fix_Yinf'])) if fixed]
        tag_str = f" ({', '.join(tags)})" if tags else " (free fit)"

        label = f"{base_label}{tag_str} — Order {order}"
        line, = plt.plot(t_sim, Y_sim, lw=2.5, label=label)

        if show_experimental and study.t_exp is not None:
            plt.scatter(study.t_exp, study.Y_exp, marker="x", s=35, alpha=0.7,
                        color=line.get_color(), label=f"{base_label} — Experimental")

    plt.xlabel("Time")
    plt.ylabel("Y property")
    plt.title(f"Order {order} — fitted curves compared across studies")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    if save_img:
        if os.path.exists(save_img):
            print(f"{color.RED}{hl.BOLD}⚠️ File '{save_img}' already exists. Overwrite? (y/n): {color.OFF}", end="")
            response = input().lower().strip()
            if response != 'y':
                print("💾 Save cancelled. Plot not saved.")
                plt.show()
                return
        directory = os.path.dirname(save_img)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(save_img, dpi=300, bbox_inches='tight')
        print(f"💾 Plot saved as: {hl.BOLD}{save_img}{hl.bold}")

    plt.show()


# =====================================================================
#  ReactionNetwork: general, numerically-integrated chemical kinetics.
#
#  kbasic and KORD both rely on closed-form integrated rate laws, valid
#  only for a single reactant -> single product transformation of order
#  0, 1 or 2. ReactionNetwork drops that restriction: it builds the
#  system of ODEs for an *arbitrary* set of (ir)reversible elementary
#  reactions straight from mass-action kinetics, integrates it
#  numerically (scipy.integrate.solve_ivp), and fits rate constants
#  against experimental data the same way KORD does -- a measured
#  signal Y_EXP(t) as a linear combination of concentrations,
#  Y_EXP(t) = sum_i eta_i * C_i(t) (see Kinetics.md, 4.2.1).
#
#  This subsumes parallel reactions (A->B, A->C), consecutive reactions
#  (A->B->C, the Bateman equations), reversible reactions (A<=>B), and
#  anything built out of those as a network -- and lets the quasi-steady-
#  state approximation be checked numerically against the exact solution
#  instead of only illustrated on a closed-form special case.
# =====================================================================

def _normalize_species_name(name):
    """Canonicalizes a parsed species name so that the same compound is
    recognized as the same species no matter how it was typed across
    different `add_reaction()` calls: accents are stripped (NFKD
    decomposition, dropping the combining marks -- "e'" -> "e", etc.,
    covering e/a/o/i/u with any accent) and, for a purely-alphabetic
    name, only the first letter is kept uppercase and the rest is
    lowercased ("ETHYLENE", "ethylene", "Ethylene" all become
    "Ethylene"). A name containing digits is treated as a chemical
    formula rather than a spelled-out word (e.g. "O3", "CO2", "H2O"):
    only its first character is forced to uppercase, the rest is left
    exactly as typed, so formula case is never mangled."""
    decomposed = unicodedata.normalize('NFKD', name)
    stripped = ''.join(c for c in decomposed if not unicodedata.combining(c))
    if not stripped:
        return stripped
    if stripped.isalpha():
        return stripped.capitalize()
    return stripped[0].upper() + stripped[1:]


def _parse_reaction_side(side, equation):
    """Parses one side of a reaction equation (left or right of the
    arrow) into a {species: stoichiometric_coefficient} dict.
    Terms are separated by '+'; each term is an optional integer
    coefficient followed by a species name, e.g. "2 B", "H2O", or a
    single-word Unicode name such as "ethylene" (accents included,
    e.g. "cyclobutane"). A species name cannot contain a space -- alias
    a multi-word compound to a short symbol first (A = ethylene, B =
    butadiene, then "2 A -> B"), exactly as on paper with A, B, C...

    Every name is run through `_normalize_species_name()`, so "2
    ethylene -> cyclobutane" and "2 Ethylene -> Cyclobutane" (or even
    "2 ETHYLENE -> CYCLOBUTANE") are recognized as the exact same
    reaction, and species are always stored/displayed as "Ethylene",
    "Cyclobutane", etc.
    """
    terms = {}
    for term in side.split("+"):
        term = term.strip()
        if not term:
            continue
        # First character: anything but a digit or whitespace (so a
        # leading coefficient is unambiguous); rest: any Unicode "word"
        # character (\w matches accented letters too for str patterns).
        m = re.match(r'^(\d+)?\s*([^\d\s]\w*)$', term, flags=re.UNICODE)
        if not m:
            raise ValueError(
                f"Could not parse term '{term}' in reaction '{equation}'. "
                f"Expected e.g. 'A', '2 B', 'H2O', 'ethylene' (a single "
                f"word -- no space -- optionally preceded by an integer "
                f"coefficient). For a multi-word compound, alias it to a "
                f"symbol first, e.g. A = ethylene, then write '2 A -> B'."
            )
        coeff = int(m.group(1)) if m.group(1) else 1
        species = _normalize_species_name(m.group(2))
        terms[species] = terms.get(species, 0) + coeff
    if not terms:
        raise ValueError(f"Empty side in reaction '{equation}'.")
    return terms


def _parse_reaction(equation):
    """Parses a reaction equation such as "A + 2 B -> C" (irreversible)
    or "A <=> B" (reversible) into (reactants, products, reversible)."""
    if "<=>" in equation:
        left, right = equation.split("<=>")
        reversible = True
    elif "->" in equation:
        left, right = equation.split("->")
        reversible = False
    else:
        raise ValueError(
            f"Reaction '{equation}' has no arrow: use '->' (irreversible) "
            f"or '<=>' (reversible)."
        )
    reactants = _parse_reaction_side(left, equation)
    products = _parse_reaction_side(right, equation)
    return reactants, products, reversible


class ReactionNetwork:
    """
    A general chemical kinetics tool: build an arbitrary network of
    (ir)reversible elementary reactions, simulate the concentration of
    every species over time by numerically integrating the mass-action
    ODE system, and fit the rate constant(s) of each reaction against
    experimental data.

    Unlike `kbasic`/`KORD` (closed-form integrated rate laws for a
    single A -> B transformation of order 0, 1 or 2), ReactionNetwork
    has no restriction on the number of species or reactions: parallel
    reactions (A->B, A->C), consecutive reactions (A->B->C), reversible
    reactions (A<=>B), and any combination of those, are all just a
    network of elementary steps.

    Workflow (three steps, in order):
        1) net.add_reaction("A -> B", k_guess=...)   # one call per step
        2) net.set_initial_conditions({"A": a0, ...})
        3) net.fit(t_exp, Y_exp, weights={...})      # or net.simulate(t)

    Example (consecutive reactions, A -> B -> C)::

        net = ReactionNetwork()
        net.add_reaction("A -> B", k_guess=0.5)
        net.add_reaction("B -> C", k_guess=0.1)
        net.set_initial_conditions({"A": 1.0, "B": 0.0, "C": 0.0})

        # If only B is measured (e.g. it's the colored/absorbing species):
        net.fit(t_exp, Y_exp, observed_species="B")
        net.plot(t_exp, Y_exp)

    Example (reversible reaction, A <=> B)::

        net = ReactionNetwork()
        net.add_reaction("A <=> B", kf_guess=1.0, kr_guess=0.2)
        net.set_initial_conditions({"A": 1.0, "B": 0.0})
        net.fit(t_exp, Y_exp, observed_species="B")

    Notes:
        - Rate law: mass action by default -- a reaction's forward (and,
          if reversible, backward) rate is k times the product of its
          reactants' (products') concentrations, each raised to its own
          stoichiometric coefficient. This is the standard assumption
          for an *elementary* step; it is not automatically valid for an
          overall multi-step reaction written as a single line (see
          Kinetics.md 4.2's discussion of pseudo-order and the QSSA).
        - `fit()` requires `set_initial_conditions()` to have been called
          first, and either `weights={...}` (general linear combination
          of concentrations) or the `observed_species="X"` shortcut
          (equivalent to `weights={"X": 1.0}`).
        - All rate constants are constrained to be >= 0 (physical
          consistency), exactly like k/binf in KORD.fit().
    """

    def __init__(self, verbose=True):
        self.species = []
        self.reactions = []
        self.C0 = None
        self.k_fit = None
        self.results = None
        self.verbose = verbose
        self._idx_of = {}
        self._initialized = True   # see __setattr__ below

    def __setattr__(self, name, value):
        """Same typo guard as kbasic/KORD: once built, no new attribute
        can be created, only existing ones can be changed -- catches a
        misspelled attribute instead of silently creating it."""
        if getattr(self, '_initialized', False) and name not in self.__dict__:
            existing = ', '.join(sorted(k for k in self.__dict__ if not k.startswith('_')))
            raise AttributeError(
                f"ReactionNetwork has no attribute '{name}' (possible typo?). "
                f"Existing attributes: {existing}"
            )
        object.__setattr__(self, name, value)

    def copy(self):
        """
        Returns an independent deep copy of this ReactionNetwork, so the
        same reaction list/initial conditions can be fitted more than
        once (different starting guesses, different `weights`, a
        different subset of the data...) without one `fit()` call
        overwriting another's result -- the same rationale as
        `KORD.copy()`. Mainly a convenience once a network has several
        reactions, so they don't have to be declared twice::

            net_B_only = ReactionNetwork()
            net_B_only.add_reaction("A -> B", k_guess=0.5)
            net_B_only.add_reaction("B -> C", k_guess=0.1)
            net_B_only.set_initial_conditions({"A": 1.0, "B": 0.0, "C": 0.0})

            net_mixed = net_B_only.copy()           # same reactions/initial conditions
            net_B_only.fit(t_exp, Y_exp, observed_species="B")
            net_mixed.fit(t_exp, Y_exp, weights={"B": 0.8, "C": 0.2})
            # net_B_only.results and net_mixed.results can now be compared
        """
        return _copy.deepcopy(self)

    def add_reaction(self, equation, k_guess=None, kf_guess=None, kr_guess=None,
                      order=None, order_back=None):
        """
        Adds one elementary reaction to the network, e.g.
        ``net.add_reaction("A + 2 B -> C", k_guess=0.5)`` or
        ``net.add_reaction("A <=> B", kf_guess=1.0, kr_guess=0.2)``.

        Args:
            equation (str): reaction equation. Species are single-word
                names (letters/digits/underscore, Unicode accents
                allowed, e.g. "A", "H2O", "ethylene"); terms are
                separated by '+', with an optional leading integer
                stoichiometric coefficient (e.g. "2 B"). Use '->' for an
                irreversible reaction, '<=>' for a reversible one.
            k_guess (float, optional): starting guess for the (single)
                rate constant of an irreversible reaction. Defaults to
                1.0 if not given.
            kf_guess, kr_guess (float, optional): starting guesses for
                the forward/backward rate constants of a reversible
                reaction. Each defaults to 1.0 if not given (kf_guess
                also accepts the value passed as k_guess, for
                convenience).
            order (dict, optional): forward rate law exponents, as
                {species: exponent}, overriding the default assumption
                that the kinetic order in each reactant equals its
                stoichiometric coefficient. Only needed when the
                mechanism isn't a textbook elementary step (e.g. a
                measured/pseudo order that differs from the written
                stoichiometry) -- see Kinetics.md 4.2's discussion of
                pseudo-order kinetics for the same caveat on `kbasic`/
                `KORD`. Keys must be a subset of this reaction's
                reactants.
            order_back (dict, optional): same as `order`, but for the
                backward rate of a reversible reaction; keys must be a
                subset of this reaction's products. Ignored for an
                irreversible reaction.

        Any species not seen in a previous call is added to the network
        automatically, in order of first appearance.
        """
        reactants, products, reversible = _parse_reaction(equation)

        if order is not None and not set(order) <= set(reactants):
            raise ValueError(
                f"order={order} refers to species not among this reaction's "
                f"reactants {list(reactants)} (equation: '{equation}')."
            )
        if order_back is not None:
            if not reversible:
                raise ValueError("order_back only applies to a reversible ('<=>') reaction.")
            if not set(order_back) <= set(products):
                raise ValueError(
                    f"order_back={order_back} refers to species not among this "
                    f"reaction's products {list(products)} (equation: '{equation}')."
                )

        for sp in list(reactants) + list(products):
            if sp not in self.species:
                self.species.append(sp)
        self._idx_of = {sp: i for i, sp in enumerate(self.species)}

        # Default kinetic order = stoichiometric coefficient (elementary
        # mass action); an explicit `order`/`order_back` overrides it,
        # species by species.
        fwd_order = dict(reactants)
        fwd_order.update(order or {})
        back_order = dict(products)
        back_order.update(order_back or {})

        idx = len(self.reactions) + 1
        if reversible:
            kf_guess = kf_guess if kf_guess is not None else (k_guess if k_guess is not None else 1.0)
            kr_guess = kr_guess if kr_guess is not None else 1.0
            reaction = {
                'equation': equation, 'reactants': reactants, 'products': products,
                'reversible': True, 'order': fwd_order, 'order_back': back_order,
                'kf_name': f"k{idx}", 'kr_name': f"k-{idx}",
                'kf_guess': kf_guess, 'kr_guess': kr_guess,
            }
            if self.verbose:
                print(f"  Reaction {idx} (reversible): {equation}  "
                      f"[kf_guess={kf_guess:.3g}, kr_guess={kr_guess:.3g}]")
        else:
            k_guess = k_guess if k_guess is not None else 1.0
            reaction = {
                'equation': equation, 'reactants': reactants, 'products': products,
                'reversible': False, 'order': fwd_order,
                'k_name': f"k{idx}", 'k_guess': k_guess,
            }
            if self.verbose:
                print(f"  Reaction {idx}: {equation}  [k_guess={k_guess:.3g}]")

        self.reactions.append(reaction)

    def overall_equation(self):
        """
        Sums the stoichiometry of every declared reaction (each counted
        once, forward direction as written) and cancels out whatever
        appears on both sides -- typically the reaction intermediates --
        to get the overall/net equation of the mechanism. Returns a
        string such as "A -> C" for the mechanism "A -> B", "B -> C"
        (B is an intermediate and cancels).

        This is a useful sanity check on a multi-step mechanism: the
        overall equation it implies should match the one you started
        from (e.g. the balanced equation given in the problem).

        Note: a reversible reaction contributes its forward stoichiometry
        only -- this is about the mechanism's bookkeeping (which atoms go
        where across the whole sequence), not about net flux at a given
        time, which depends on the actual rate constants and
        concentrations (see `qssa_check()` for that).
        """
        net = {sp: 0 for sp in self.species}
        for r in self.reactions:
            for sp, c in r['reactants'].items():
                net[sp] -= c
            for sp, c in r['products'].items():
                net[sp] += c

        def fmt(sp, c):
            return sp if c == 1 else f"{c} {sp}"

        left = [fmt(sp, -net[sp]) for sp in self.species if net[sp] < 0]
        right = [fmt(sp, net[sp]) for sp in self.species if net[sp] > 0]
        if not left or not right:
            return ("(every species cancels out -- the declared steps sum "
                     "to no net change at all)")
        return f"{' + '.join(left)} -> {' + '.join(right)}"

    def set_initial_conditions(self, conditions):
        """
        Sets each species' concentration at t=0, e.g.
        ``net.set_initial_conditions({"A": 1.0, "B": 0.0})``.

        Any species already in the network but missing from
        ``conditions`` defaults to 0.0 (with a warning if ``verbose``).
        """
        missing = [sp for sp in self.species if sp not in conditions]
        if missing and self.verbose:
            print(f"⚠️  No initial concentration given for {missing}; "
                  f"defaulting to 0.0.")
        self.C0 = np.array([conditions.get(sp, 0.0) for sp in self.species], dtype=float)

    def _param_specs(self):
        """Ordered list of (parameter_name, starting_guess) across every
        reaction -- one entry for an irreversible reaction's k, two
        (kf, kr) for a reversible one."""
        specs = []
        for r in self.reactions:
            if r['reversible']:
                specs.append((r['kf_name'], r['kf_guess']))
                specs.append((r['kr_name'], r['kr_guess']))
            else:
                specs.append((r['k_name'], r['k_guess']))
        return specs

    def _rhs(self, t, C, k_map):
        """Right-hand side of the mass-action ODE system, dC/dt, given
        the current concentration vector C and a {param_name: value}
        mapping of rate constants.

        The stoichiometric coefficients (reactants/products) always
        drive the mass-balance update of dC; the kinetic order used in
        the rate law itself comes from each reaction's 'order'/
        'order_back' dict, which defaults to those same stoichiometric
        coefficients (elementary mass action) but can be overridden
        per-species via add_reaction(order=..., order_back=...)."""
        dC = np.zeros(len(self.species))
        for r in self.reactions:
            rate_f = k_map[r['kf_name'] if r['reversible'] else r['k_name']]
            for sp, exp in r['order'].items():
                rate_f *= C[self._idx_of[sp]] ** exp
            rate_net = rate_f
            if r['reversible']:
                rate_b = k_map[r['kr_name']]
                for sp, exp in r['order_back'].items():
                    rate_b *= C[self._idx_of[sp]] ** exp
                rate_net = rate_f - rate_b
            for sp, coeff in r['reactants'].items():
                dC[self._idx_of[sp]] -= coeff * rate_net
            for sp, coeff in r['products'].items():
                dC[self._idx_of[sp]] += coeff * rate_net
        return dC

    def _gross_rhs(self, t, C, k_map):
        """Like _rhs, but also returns, for each species, the sum of the
        *absolute* contribution of every reaction touching it (gross
        flux, counting production and consumption as positive) --
        used by qssa_check() to judge whether net change is small
        compared to how much is actually flowing through a species."""
        dC = np.zeros(len(self.species))
        gross = np.zeros(len(self.species))
        for r in self.reactions:
            rate_f = k_map[r['kf_name'] if r['reversible'] else r['k_name']]
            for sp, exp in r['order'].items():
                rate_f *= C[self._idx_of[sp]] ** exp
            rate_net = rate_f
            if r['reversible']:
                rate_b = k_map[r['kr_name']]
                for sp, exp in r['order_back'].items():
                    rate_b *= C[self._idx_of[sp]] ** exp
                rate_net = rate_f - rate_b
            for sp, coeff in r['reactants'].items():
                dC[self._idx_of[sp]] -= coeff * rate_net
                gross[self._idx_of[sp]] += abs(coeff * rate_net)
            for sp, coeff in r['products'].items():
                dC[self._idx_of[sp]] += coeff * rate_net
                gross[self._idx_of[sp]] += abs(coeff * rate_net)
        return dC, gross

    def simulate(self, t, k_values=None):
        """
        Integrates the ODE system and returns each species' concentration
        at the requested time points.

        Args:
            t: array-like of time points (need not be sorted).
            k_values: rate constants to use -- a {param_name: value} dict,
                a flat list/array in the same order as the reactions were
                added (k or kf,kr per reaction), or None (default) to use
                the fitted values from `fit()` if available, otherwise the
                guesses passed to `add_reaction()`.

        Returns:
            dict: {species_name: numpy array of concentrations at `t`}.
        """
        if self.C0 is None:
            raise RuntimeError("Call set_initial_conditions() before simulate().")
        specs = self._param_specs()
        if k_values is None:
            k_map = {name: (self.k_fit[name] if self.k_fit is not None else guess)
                      for name, guess in specs}
        elif isinstance(k_values, dict):
            k_map = k_values
        else:
            k_map = {name: val for (name, _), val in zip(specs, k_values)}

        t = np.asarray(t, dtype=float)
        order = np.argsort(t)
        t_sorted = t[order]
        t0 = min(0.0, t_sorted[0])
        sol = solve_ivp(lambda tt, C: self._rhs(tt, C, k_map),
                         (t0, t_sorted[-1]), self.C0, t_eval=t_sorted,
                         method="LSODA", rtol=1e-8, atol=1e-12)
        if not sol.success:
            raise RuntimeError(f"ODE integration failed: {sol.message}")

        inv_order = np.argsort(order)
        return {sp: sol.y[i][inv_order] for i, sp in enumerate(self.species)}

    def qssa_check(self, species, t, k_values=None):
        """
        Checks, numerically, how good the quasi-steady-state
        approximation (QSSA/QSSA) is for ``species`` at each time in
        ``t`` -- the classic question for an intermediate in a
        consecutive mechanism (e.g. B in A -> B -> C): is d[B]/dt really
        negligible compared to how fast B is being produced and
        consumed, or only "small enough" after some induction period?

        Integrates the exact ODE system (same as `simulate()`), then at
        every requested time compares the *net* rate of change of
        ``species`` to its *gross* rate -- the sum of the absolute value
        of every reaction's individual contribution to it (production
        and consumption both counted as positive, so they cannot cancel
        out the way they do in the net rate):

        $$\\text{QSSA ratio}(t) = \\frac{|d[\\text{species}]/dt|}{\\sum_j |\\text{contribution}_j(t)|}$$

        A ratio close to 0 means production and consumption are nearly
        balanced (net change negligible compared to the flux passing
        through) -- the QSSA holds; a ratio close to 1 means the species
        is still mostly accumulating or mostly depleting on its own, not
        yet (or no longer) in a quasi-steady state.

        Args:
            species (str): the species to check (typically a reactive
                intermediate).
            t: array-like of time points to evaluate the ratio at.
            k_values: same as `simulate()` -- rate constants to use, or
                None to use the fitted/guessed values.

        Returns:
            numpy array: the QSSA ratio at each time in ``t`` (same
            order as given, not necessarily sorted).
        """
        if species not in self.species:
            raise ValueError(f"Unknown species '{species}'; known: {self.species}")
        if self.C0 is None:
            raise RuntimeError("Call set_initial_conditions() before qssa_check().")

        specs = self._param_specs()
        if k_values is None:
            k_map = {name: (self.k_fit[name] if self.k_fit is not None else guess)
                      for name, guess in specs}
        elif isinstance(k_values, dict):
            k_map = k_values
        else:
            k_map = {name: val for (name, _), val in zip(specs, k_values)}

        t = np.asarray(t, dtype=float)
        C_at_t = self.simulate(t, k_values=k_map)
        idx = self._idx_of[species]

        ratio = np.empty(len(t))
        for i in range(len(t)):
            C_vec = np.array([C_at_t[sp][i] for sp in self.species])
            dC, gross = self._gross_rhs(t[i], C_vec, k_map)
            ratio[i] = abs(dC[idx]) / gross[idx] if gross[idx] > 0 else 0.0
        return ratio

    def fit(self, t_exp, Y_exp, weights=None, observed_species=None):
        """
        Fits every reaction's rate constant(s) against experimental data,
        assuming the measured signal is a linear combination of
        concentrations, Y_EXP(t) = sum_i eta_i * C_i(t) (see Kinetics.md,
        4.2.1 -- the same assumption KORD relies on).

        Args:
            t_exp, Y_exp: experimental time points and measured values.
            weights (dict, optional): {species: eta_i} linear-combination
                weights. Use this when the signal mixes several species.
            observed_species (str, optional): shortcut for
                ``weights={observed_species: 1.0}``, when the signal is
                one single species' concentration directly.

        Exactly one of `weights`/`observed_species` must be given.

        Returns:
            dict: {'k': {param_name: fitted_value}, 'RMSE', 'R2', 'AIC',
            'BIC', 'weights'}. Also stored in ``self.results``, and the
            fitted rate constants in ``self.k_fit`` (so a later
            ``simulate()``/``plot()`` call with no ``k_values`` argument
            uses them automatically).
        """
        if observed_species is not None:
            if weights is not None:
                raise ValueError("Pass only one of weights= or observed_species=.")
            weights = {observed_species: 1.0}
        if weights is None:
            raise ValueError("Specify either weights={...} or observed_species='X'.")

        specs = self._param_specs()
        names = [name for name, _ in specs]
        p0 = [guess for _, guess in specs]

        t_exp = np.asarray(t_exp, dtype=float)
        Y_exp = np.asarray(Y_exp, dtype=float)

        def model(t, *k_vals):
            k_map = dict(zip(names, k_vals))
            C = self.simulate(t, k_values=k_map)
            return sum(w * C[sp] for sp, w in weights.items())

        popt, _ = curve_fit(model, t_exp, Y_exp, p0=p0, bounds=(0, np.inf))
        k_opt = dict(zip(names, popt))
        Y_theo = model(t_exp, *popt)

        RMSE_ = rmse(Y_exp, Y_theo)
        R2_ = r2(Y_exp, Y_theo)
        n, p = len(Y_exp), len(popt)
        aic = n * np.log(RMSE_**2) + 2 * p
        bic = n * np.log(RMSE_**2) + p * np.log(n)

        self.k_fit = k_opt
        self.results = {'k': k_opt, 'RMSE': RMSE_, 'R2': R2_, 'AIC': aic,
                         'BIC': bic, 'weights': weights}

        if self.verbose:
            k_str = ", ".join(f"{name}={val:.4g}" for name, val in k_opt.items())
            print(f"  FIT: {k_str}")
            print(f"  ✅ RMSE: {RMSE_:.2e}")
            print(f"       R2: {R2_:.2f}")
            print(f"      AIC: {aic:.2f}")
            print(f"      BIC: {bic:.2f}")

        return self.results

    def plot(self, t_exp=None, Y_exp=None, t_max=None, log_scale=False,
             time_unit=None, species=None, save_img=None):
        """
        Plots simulated concentration vs. time (using the fitted rate
        constants if `fit()` was called, otherwise the guesses),
        optionally overlaying the experimental ``(t_exp, Y_exp)`` points
        on the fitted observed linear combination.

        ``species``: optional list of species names to plot (e.g.
        ``net.plot(['A', 'I'])`` to zoom in on a reactant and a reaction
        intermediate). Defaults to every species in the network. Each
        name is checked against ``self.species`` first, with a clear
        error listing the known species if one doesn't exist (a likely
        typo, or a name that was never declared via `add_reaction()`).

        ``log_scale=True`` switches the y-axis to log scale -- useful
        whenever concentrations span several orders of magnitude, e.g. a
        trace reactive intermediate alongside a bulk reactant/product
        (species that reach exactly 0 are simply left off the log plot
        for the points where they're non-positive, as matplotlib does).

        ``time_unit``: optional string purely for the x-axis label, e.g.
        ``time_unit="s"`` labels the axis "Time / s". `ReactionNetwork`
        itself is unit-agnostic -- `t`, `C0` and the rate constants just
        have to be in a mutually consistent set of units; nothing here
        converts between units or checks them, this only labels the plot.

        ``save_img``: optional file path to save the figure to (png,
        pdf, svg, ... -- whatever matplotlib's `savefig` infers from the
        extension), in addition to showing it. Asks before overwriting
        an existing file.
        """
        if species is None:
            species = self.species
        else:
            unknown = [sp for sp in species if sp not in self.species]
            if unknown:
                raise ValueError(f"Unknown species {unknown}; known species: {self.species}")

        if t_max is None:
            if t_exp is not None:
                t_max = float(np.max(t_exp))
            else:
                raise ValueError("Pass t_max, or t_exp to infer it from.")

        t_grid = np.linspace(0.0, t_max, 300)
        C = self.simulate(t_grid)

        plt.figure(figsize=(8, 5))
        for sp in species:
            plt.plot(t_grid, C[sp], lw=2.5, label=sp)

        if t_exp is not None and Y_exp is not None:
            if self.results is not None:
                label = "Experimental (" + " + ".join(
                    f"{w:.2g}·{sp}" if w != 1 else sp
                    for sp, w in self.results['weights'].items()) + ")"
            else:
                label = "Experimental"
            plt.scatter(t_exp, Y_exp, color='black', marker='x', s=35,
                        alpha=0.7, zorder=5, label=label)

        plt.xlabel(f"Time / {time_unit}" if time_unit else "Time")
        plt.ylabel("Concentration")
        if log_scale:
            plt.yscale("log")
        plt.title("Reaction network simulation")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)

        if save_img:
            if os.path.exists(save_img):
                print(f"{color.RED}{hl.BOLD}⚠️ File '{save_img}' already exists. Overwrite? (y/n): {color.OFF}", end="")
                response = input().lower().strip()
                if response != 'y':
                    print("💾 Save cancelled. Plot not saved.")
                    plt.show()
                    return
            directory = os.path.dirname(save_img)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(save_img, dpi=300, bbox_inches='tight')
            print(f"💾 Plot saved as: {hl.BOLD}{save_img}{hl.bold}")

        plt.show()

    def plot_qssa(self, species, t, threshold=0.1, time_unit=None, save_img=None):
        """
        Two-panel diagnostic plot for the quasi-steady-state approximation
        on ``species`` (see `qssa_check()`): concentrations on top, the
        QSSA ratio on the bottom, with the region under ``threshold``
        shaded to mark where the approximation is a good one.

        ``time_unit``: optional string purely for the (shared) x-axis
        label, e.g. ``time_unit="s"`` -- see `plot()`'s docstring; this
        does not convert or check units, only labels the axis.

        ``save_img``: optional file path to save the figure to, in
        addition to showing it -- see `plot()`'s docstring.
        """
        t = np.asarray(t, dtype=float)
        order = np.argsort(t)
        t_sorted = t[order]

        C = self.simulate(t_sorted)
        ratio = self.qssa_check(species, t_sorted)

        fig, (ax_C, ax_r) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                           gridspec_kw={'height_ratios': [1.3, 1]})

        for sp in self.species:
            lw = 3.0 if sp == species else 1.5
            alpha = 1.0 if sp == species else 0.5
            ax_C.plot(t_sorted, C[sp], lw=lw, alpha=alpha, label=sp)
        ax_C.set_ylabel("Concentration")
        ax_C.set_title(f"Quasi-steady-state check for '{species}'")
        ax_C.legend()
        ax_C.grid(True, linestyle=':', alpha=0.6)

        ax_r.axhspan(0, threshold, color='green', alpha=0.15,
                     label=f"QSSA valid (ratio < {threshold:g})")
        ax_r.plot(t_sorted, ratio, color='black', lw=2.0)
        ax_r.set_xlabel(f"Time / {time_unit}" if time_unit else "Time")
        ax_r.set_ylabel(f"QSSA ratio for '{species}'")
        ax_r.set_ylim(0, max(1.0, float(np.max(ratio)) * 1.05))
        ax_r.legend()
        ax_r.grid(True, linestyle=':', alpha=0.6)

        if save_img:
            if os.path.exists(save_img):
                print(f"{color.RED}{hl.BOLD}⚠️ File '{save_img}' already exists. Overwrite? (y/n): {color.OFF}", end="")
                response = input().lower().strip()
                if response != 'y':
                    print("💾 Save cancelled. Plot not saved.")
                    plt.show()
                    return
            directory = os.path.dirname(save_img)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(save_img, dpi=300, bbox_inches='tight')
            print(f"💾 Plot saved as: {hl.BOLD}{save_img}{hl.bold}")

        plt.show()

    def summary(self):
        """Prints each reaction with its (fitted, if available, else
        guessed) rate constant(s)."""
        for i, r in enumerate(self.reactions, start=1):
            if r['reversible']:
                kf = self.k_fit[r['kf_name']] if self.k_fit else r['kf_guess']
                kr = self.k_fit[r['kr_name']] if self.k_fit else r['kr_guess']
                tag = "fitted" if self.k_fit else "guess"
                print(f"  {i}. {r['equation']}   {r['kf_name']}={kf:.4g} ({tag}), "
                      f"{r['kr_name']}={kr:.4g} ({tag})")
            else:
                k = self.k_fit[r['k_name']] if self.k_fit else r['k_guess']
                tag = "fitted" if self.k_fit else "guess"
                print(f"  {i}. {r['equation']}   {r['k_name']}={k:.4g} ({tag})")
