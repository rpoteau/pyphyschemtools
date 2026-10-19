############################################################
#                       Periodic Table
############################################################

class TableauPeriodique:
    """
    Classe permettant de manipuler et d'afficher les données du tableau périodique.

    Cette classe utilise la bibliothèque 'mendeleev' pour récupérer les données chimiques
    et 'bokeh' pour la visualisation interactive. Elle francise les noms et corrige
    certaines classifications de familles chimiques.

    Initialise l'objet TableauPeriodique en chargeant les données de la bibliothèque mendeleev.

    .. note::
       See also :class:`PeriodicTable`, the English-language counterpart of this
       class (same data and methods, but without the French localization patch).
    """

    nomsFr=['Hydrogène','Hélium','Lithium','Béryllium','Bore','Carbone','Azote','Oxygène',
            'Fluor','Néon','Sodium','Magnésium','Aluminium','Silicium','Phosphore','Soufre',
            'Chlore','Argon','Potassium','Calcium','Scandium','Titane','Vanadium','Chrome',
            'Manganèse','Fer','Cobalt','Nickel','Cuivre','Zinc','Gallium','Germanium',
            'Arsenic','Sélénium','Brome','Krypton','Rubidium','Strontium','Yttrium',
            'Zirconium','Niobium','Molybdène','Technétium','Ruthénium','Rhodium',
            'Palladium','Argent','Cadmium','Indium',
            'Étain','Antimoine','Tellure','Iode','Xénon','Césium','Baryum','Lanthane','Cérium',
            'Praséodyme','Néodyme','Prométhium','Samarium','Europium','Gadolinium','Terbium',
            'Dysprosium','Holmium','Erbium','Thulium','Ytterbium','Lutetium','Hafnium','Tantale',
            'Tungstène','Rhénium','Osmium','Iridium','Platine','Or','Mercure','Thallium','Plomb',
            'Bismuth','Polonium','Astate','Radon','Francium','Radium','Actinium','Thorium','Protactinium',
            'Uranium','Neptunium','Plutonium','Americium','Curium','Berkelium','Californium','Einsteinium',
            'Fermium','Mendelevium','Nobelium','Lawrencium','Rutherfordium','Dubnium','Seaborgium','Bohrium',
            'Hassium','Meitnerium','Darmstadtium','Roentgenium','Copernicium','Nihonium','Flerovium',
            'Moscovium','Livermorium','Tennesse','Oganesson',
            ]
    trad = {'Nonmetals':'Non métal',
            'Noble gases':'Gaz noble',
            'Alkali metals':'Métal alcalin',
            'Alkaline earth metals':'Métal alcalino-terreux',
            'Metalloids':'Métalloïde',
            'Halogens':'Halogène',
            'Poor metals':'Métal pauvre',
            'Transition metals':'Métal de transition',
            'Lanthanides':'Lanthanide',
            'Actinides':'Actinide',
            'Metals':'Métal',
           }

    # Éléments reclassés par l'IUPAC comme binucléidiques (ou plus), malgré un seul
    # isotope présent dans la nature au sens de mendeleev (abundance non None), à
    # cause d'un isotope radiogénique issu de la désintégration d'un "parent" plus
    # lourd. Source : CIAAW (ciaaw.org), fiche par élément.
    #
    # À ce jour, le thorium est le seul cas confirmé par l'IUPAC/CIAAW : le 230-Th,
    # issu de la chaîne de désintégration de l'238U, est présent en traces (jusqu'à
    # 0.02% du Th total dans les eaux océaniques profondes), ce qui a conduit la
    # Commission à le reclasser en 2013 (auparavant considéré monoisotopique).
    #
    # Cas vérifié mais écarté : le protactinium a lui aussi un isotope radiogénique
    # de ce type (234-Pa, issu de la désintégration de l'238U), mais la CIAAW ne l'a
    # PAS reclassé car le 234-Pa est à l'équilibre dans les minerais d'uranium et
    # disparaît rapidement après séparation chimique — la masse atomique standard du
    # Pa reste basée uniquement sur le 231-Pa.
    #
    # D'autres actinides/lanthanides pourraient être concernés par des traces
    # radiogéniques analogues (ex: Ac, U lui-même a plusieurs isotopes primordiaux
    # donc n'est pas mononucléidique de toute façon) : à vérifier au cas par cas sur
    # ciaaw.org avant d'ajouter une entrée ici.
    exceptionsRadiogeniques = {
        90: "230-Th, issu de la désintégration de l'238U (IUPAC/CIAAW, 2013)",
    }

    # English-language version of exceptionsRadiogeniques, used by the PeriodicTable
    # subclass. Keys must stay in sync with exceptionsRadiogeniques.
    exceptionsRadiogenic = {
        90: "230-Th, a decay product of 238U (IUPAC/CIAAW, 2013)",
    }

    # Éléments pour lesquels l'IUPAC/CIAAW ne publie pas une masse atomique standard
    # unique mais un INTERVALLE [min, max], parce que la composition isotopique
    # naturelle varie sensiblement selon l'origine (géologique et/ou, pour Li
    # notamment, anthropique : séparation isotopique industrielle). La valeur unique
    # "atomic_weight" fournie par mendeleev (et donc par self.elements / prop() /
    # isotopes()) n'est qu'une valeur CONVENTIONNELLE choisie dans cet intervalle,
    # pas nécessairement égale à Σ(abondance × masse isotopique) calculée à partir
    # de la composition isotopique "représentative" également fournie par mendeleev.
    # Source : fiche par élément sur ciaaw.org (ex: ciaaw.org/lithium.htm). This table
    # is language-agnostic (just numbers) and is shared by TableauPeriodique and
    # PeriodicTable.
    intervallesCIAAW = {
        1:  (1.00784,  1.00811),    # H
        3:  (6.938,    6.997),      # Li
        5:  (10.806,   10.821),     # B
        6:  (12.0096,  12.0116),    # C
        7:  (14.00643, 14.00728),   # N
        8:  (15.99903, 15.99977),   # O
        12: (24.304,   24.307),     # Mg
        14: (28.084,   28.086),     # Si
        16: (32.059,   32.076),     # S
        17: (35.446,   35.457),     # Cl
        35: (79.901,   79.907),     # Br
        81: (204.382,  204.385),    # Tl
    }

    def __init__(self):
        from mendeleev.vis import create_vis_dataframe
        self.elements = create_vis_dataframe()
        self.patch_elements()

    def patch_elements(self):
        '''
        Ce patch, appliqué à self.elements, créé par l'appel à create_vis_dataframe(), va servir à :
        - corriger certaines classifications de familles (retire les éléments du groupe 12 de la
          famille des métaux de transition, qui est le choix CONTESTABLE par défaut de mendeleev) ;
          cette partie est commune à TableauPeriodique et à sa sous-classe anglaise PeriodicTable
        - ajouter des informations localisées (noms des éléments et des familles) via
          self._localize_names(), surchargeable par une sous-classe
        - ajouter des données supplémentaires (énergie de première ionisation) via
          self._ajouter_donnees()

        input :
        elements est un dataframe pandas préalablement créé par la fonction create_vis_dataframe() de mendeleev.vis

        output :
        elements avec les colonnes ajoutées par _localize_names() et _ajouter_donnees()
        + correction des données name_series et series_id pour les éléments Zn, Cd, Hg, Cn
        '''
        # les éléments du groupe 12 ne sont pas des métaux de transition
        # (correction indépendante de la langue, donc commune aux deux classes)
        self.elements.loc[29,"name_series"] = 'Metals'
        self.elements.loc[47,"name_series"] = 'Metals'
        self.elements.loc[79,"name_series"] = 'Metals'
        self.elements.loc[111,"name_series"] = 'Metals'
        self.elements.loc[29,"series_id"] = 11
        self.elements.loc[47,"series_id"] = 11
        self.elements.loc[79,"series_id"] = 11
        self.elements.loc[111,"series_id"] = 11
        self.elements.loc[29,"color"] = "#bbd3a5"
        self.elements.loc[47,"color"] =  "#bbd3a5"
        self.elements.loc[79,"color"] =  "#bbd3a5"
        self.elements.loc[111,"color"] =  "#bbd3a5"

        self._localize_names()
        self._ajouter_donnees()
        return

    def _localize_names(self):
        """
        Ajoute les colonnes francisées 'name_seriesFr' (famille) et 'nom' (nom de
        l'élément) à self.elements.

        Surchargée (en no-op) par la sous-classe anglaise :class:`PeriodicTable`,
        puisque les colonnes 'name' et 'name_series' fournies par mendeleev sont
        déjà en anglais et n'ont donc pas besoin de traduction dans ce cas.
        """
        def series_eng2fr(s):
            '''Correspondance entre nom des séries (familles) en anglais et en français'''
            return TableauPeriodique.trad[s]

        # english > français. Ajout d'une nouvelle colonne
        self.elements["name_seriesFr"] = self.elements["name_series"].apply(series_eng2fr)
        # english > français. Nouvelle colonne avec les noms des éléments en français
        self.elements["nom"] = TableauPeriodique.nomsFr

    def _ajouter_donnees(self):
        """Récupère et fusionne des données supplémentaires comme l'énergie de première ionisation."""
        import numpy as np
        from mendeleev.fetch import fetch_table
        import pandas as pd
        # dfElts = fetch_table("elements")
        # display(dfElts)
        # NB : fetch_ionization_energies() est cassée avec SQLAlchemy >= 2.0 (elle passe
        # un objet déjà compilé à pd.read_sql_query, ce qui lève ObjectNotExecutableError).
        # fetch_table(), elle, a été corrigée pour ce cas (PR #159 sur le dépôt mendeleev) ;
        # on reconstruit donc l'équivalent de fetch_ionization_energies(degree=1) à la main :
        # degree = ion_charge + 1, donc IE1 correspond à ion_charge == 0.
        dfIE = fetch_table("ionizationenergies")
        dfEi1 = (dfIE[dfIE["ion_charge"] == 0]
                 [["atomic_number", "ionization_energy"]]
                 .rename(columns={"ionization_energy": "IE1"}))
        # display(dfEi1)
        b = pd.DataFrame({'atomic_number':[x for x in range(1, 119)]})
        dfEi1tot = pd.merge(left=dfEi1, right=b, on='atomic_number', how='outer').sort_values(by='atomic_number')
        self.elements["Ei1"] = dfEi1tot["IE1"]

    def prop(self,elt_id):
        """
        Affiche les propriétés détaillées d'un élément chimique.

        Args:
            elt_id (str ou int): Symbole de l'élément (ex: 'O') ou numéro atomique (ex: 8).
        """
        from mendeleev import element

        elt = element(elt_id)
        print(f"Nom de l'élement = {TableauPeriodique.nomsFr[elt.atomic_number-1]} ({elt.symbol}, Z = {elt.atomic_number})")
        print(f"Nom en anglais = {elt.name}")
        print(f"Origine du nom = {elt.name_origin}")
        print()
        print(f"CEF = {elt.ec} = {elt.econf}")
        print(f"Nombre d'électrons célibataires = {elt.ec.unpaired_electrons()}")
        print(f"Groupe {elt.group_id}, Période {elt.period}, bloc {elt.block}")
        print(f"Famille = {self.elements.loc[elt.atomic_number-1,'name_seriesFr']}")
        print()
        print(f"Masse molaire = {elt.atomic_weight} g/mol")
        isotopes = ""
        X = elt.symbol
        for i in elt.isotopes:
            if i.abundance is not None:
                isotopes = isotopes + str(i.mass_number)+ "^" + X + f"({i.abundance}%) / "
        print("Isotopes naturels = ",isotopes[:-2])
        print()
        if elt.electronegativity(scale='pauling') is None:
            print(f"Électronégativité de Pauling = Non définie")
        else:
            print(f"Électronégativité de Pauling = {elt.electronegativity(scale='pauling')}")
        print(f"Énergie de 1ère ionisation = {elt.ionenergies[1]:.2f} eV")
        if elt.electron_affinity is None:
            print(f"Affinité électronique = Non définie")
        else:
            print(f"Afinité électronique = {elt.electron_affinity:.2f} eV")
        print(f"Rayon atomique = {elt.atomic_radius:.1f} pm")
        print()
        print("▶ Description : ",elt.description)
        print("▶ Sources : ",elt.sources)
        print("▶ Utilisation : ",elt.uses)
        print("---------------------------------------------------------------------------------------")
        print()

    def isotopes(self, elt_id):
        """
        Affiche la table des isotopes d'un élément (numéro de masse, abondance naturelle,
        demi-vie), triée par numéro de masse croissant.

        Généralise l'exemple du thorium : pour un radioélément, l'absence d'abondance
        signifie simplement qu'aucun isotope n'est stable/naturel au sens de mendeleev.

        Args:
            elt_id (str ou int): Symbole de l'élément (ex: 'Th') ou numéro atomique (ex: 90).
        """
        from mendeleev import element
        import pandas as pd

        elt = element(elt_id)
        print(f"{elt.symbol} — {TableauPeriodique.nomsFr[elt.atomic_number-1]} (Z = {elt.atomic_number})")
        print(f"Masse molaire = {elt.atomic_weight} g/mol\n")

        lignes = []
        for iso in sorted(elt.isotopes, key=lambda i: i.mass_number):
            ab = "" if iso.abundance is None else f"{iso.abundance} %"
            hl_val = getattr(iso, "half_life", None)
            if hl_val is None:
                hl = "stable"
            else:
                hl = f"{hl_val} {getattr(iso, 'half_life_unit', '') or ''}".strip()
            masse = getattr(iso, "mass", None)
            lignes.append((iso.mass_number, masse, ab, hl))

        dfIso = pd.DataFrame(lignes, columns=["A", "masse (u)", "abondance", "demi-vie"])
        dfIso = dfIso.set_index("A")
        try:
            from IPython.display import display
            display(dfIso)
        except ImportError:
            print(dfIso.to_string())

        if elt.atomic_number in TableauPeriodique.exceptionsRadiogeniques:
            print()
            print("⚠️  Élément reclassé binucléidique par l'IUPAC (2013) :",
                  TableauPeriodique.exceptionsRadiogeniques[elt.atomic_number])

        if elt.atomic_number in TableauPeriodique.intervallesCIAAW:
            mini, maxi = TableauPeriodique.intervallesCIAAW[elt.atomic_number]
            print()
            print(f"⚠️  Composition isotopique naturelle variable : l'IUPAC/CIAAW ne donne pas "
                  f"une masse atomique unique mais un intervalle [{mini}, {maxi}] g/mol. "
                  f"La valeur {elt.atomic_weight} g/mol ci-dessus est une valeur conventionnelle, "
                  f"pas forcément égale à Σ(abondance × masse) calculée avec le tableau ci-dessus.")

    def mononucleides(self, exclure_radiogeniques=True):
        """
        Affiche les éléments mononucléidiques, c'est-à-dire possédant un seul isotope
        naturel (au sens de mendeleev : abundance non None).

        Args:
            exclure_radiogeniques (bool): si True (par défaut), exclut les éléments
                comme le thorium, reclassés binucléidiques (ou plus) par l'IUPAC en 2013
                à cause d'isotopes radiogéniques présents en traces
                (voir TableauPeriodique.exceptionsRadiogeniques).
        """
        from mendeleev.fetch import fetch_table
        import pandas as pd

        # Une seule requête en masse (table isotopes + table elements pour les symboles),
        # plutôt qu'un appel element(Z) par élément (118 requêtes ORM : beaucoup plus lent).
        dfIso = fetch_table("isotopes")
        dfElt = fetch_table("elements")[["atomic_number", "symbol", "atomic_weight"]]
        symboles = dict(zip(dfElt["atomic_number"], dfElt["symbol"]))
        massesMolaires = dict(zip(dfElt["atomic_number"], dfElt["atomic_weight"]))

        naturels = dfIso[dfIso["abundance"].notna()]
        compte = naturels.groupby("atomic_number").size()
        # Éléments strictement mononucléidiques au sens de mendeleev (count == 1). NB :
        # la base intègre déjà la reclassification IUPAC/CIAAW 2013 du thorium (elle donne
        # une abondance à la fois pour 230-Th et 232-Th), donc Th a count == 2 et n'est
        # JAMAIS inclus ici, que exclure_radiogeniques soit True ou False.
        Z_mono = set(compte[compte == 1].index)

        if not exclure_radiogeniques:
            # Réintègre les éléments comme Th : un isotope très majoritaire (>99%) et un
            # ou plusieurs isotopes radiogéniques en trace, qu'on choisit d'ignorer ici.
            Z_mono |= set(TableauPeriodique.exceptionsRadiogeniques.keys())

        mono = []
        for Z in sorted(Z_mono):
            sous = naturels[naturels["atomic_number"] == Z]
            # isotope dominant (le seul, ou le plus abondant pour les cas réintégrés ci-dessus)
            iso = sous.loc[sous["abundance"].idxmax()]
            hl_val = iso["half_life"] if pd.notna(iso["half_life"]) else None
            hl_unit = iso["half_life_unit"] if pd.notna(iso["half_life_unit"]) else None
            mono.append((Z, symboles[Z], int(iso["mass_number"]), iso["mass"],
                         massesMolaires[Z], hl_val, hl_unit))

        dfMono = pd.DataFrame(
            mono, columns=["Z", "symbole", "A", "masse (u)", "masse molaire (g/mol)", "demi-vie", "unité"]
        )
        dfMono["demi-vie"] = dfMono.apply(
            lambda r: "stable" if pd.isna(r["demi-vie"])
            else f"{r['demi-vie']:g} {r['unité']}", axis=1
        )
        # marque d'un astérisque les éléments réintégrés malgré un isotope
        # radiogénique en trace (voir TableauPeriodique.exceptionsRadiogeniques)
        dfMono["symbole"] = dfMono.apply(
            lambda r: r["symbole"] + "*"
            if r["Z"] in TableauPeriodique.exceptionsRadiogeniques else r["symbole"],
            axis=1
        )
        dfMono = dfMono.drop(columns="unité").set_index("Z")
        try:
            from IPython.display import display
            display(dfMono)
        except ImportError:
            print(dfMono.to_string())
        if not exclure_radiogeniques and any(
            Z in TableauPeriodique.exceptionsRadiogeniques for Z, *_ in mono
        ):
            print("* isotope(s) radiogénique(s) en trace ignoré(s) "
                  "(voir TableauPeriodique.exceptionsRadiogeniques)")
        note = " (hors exceptions radiogéniques)" if exclure_radiogeniques else ""
        print(f"\n{len(mono)} éléments possèdent un seul isotope naturel{note}")

        return mono

    def afficher(self):
        """
        Génère et affiche le tableau périodique interactif dans un notebook Jupyter.

        Le tableau permet de visualiser les propriétés au survol de la souris.
        """
        from bokeh.plotting import show, output_notebook
        from mendeleev.vis import periodic_table_bokeh

        # Toute cette partie du code est une copie du module bokeh de mendeleev.vis
        # La fonction periodic_table_bokeh étant faiblement configurable avec des args/kwargs,
        # elle est adaptée ici pour un affichage personnalisé

        from collections import OrderedDict

        import pandas as pd
        from pandas.api.types import is_float_dtype

        from bokeh.plotting import figure
        from bokeh.models import HoverTool, ColumnDataSource, FixedTicker

        from mendeleev.vis.utils import colormap_column


        def periodic_table_bokeh(
            elements: pd.DataFrame,
            attribute: str = "atomic_weight",
            cmap: str = "RdBu_r",
            colorby: str = "color",
            decimals: int = 3,
            height: int = 800,
            missing: str = "#ffffff",
            title: str = "Periodic Table",
            wide_layout: bool = False,
            width: int = 1200,
        ):
            """
            Use Bokeh backend to plot the periodic table. Adaptation by Romuald Poteau (romuald.poteau@univ-tlse3.fr) of the orignal periodic_table_bokeh() function of the mendeleev library

            Args:
                elements : Pandas DataFrame with the elements data. Needs to have `x` and `y`
                    columns with coordianates for each tile.
                attribute : Name of the attribute to be displayed
                cmap : Colormap to use, see matplotlib colormaps
                colorby : Name of the column containig the colors
                decimals : Number of decimals to be displayed in the bottom row of each cell
                height : Height of the figure in pixels
                missing : Hex code of the color to be used for the missing values
                title : Title to appear above the periodic table
                wide_layout: wide layout variant of the periodic table
                width : Width of the figure in pixels
            """

            if any(col not in elements.columns for col in ["x", "y"]):
                raise ValueError(
                    "Coordinate columns named 'x' and 'y' are required "
                    "in 'elements' DataFrame. Consider using "
                    "'mendeleev.vis.utils.create_vis_dataframe' and try again."
                )

            # additional columns for positioning of the text

            elements.loc[:, "y_anumber"] = elements["y"] - 0.3
            elements.loc[:, "y_name"] = elements["y"] + 0.2

            if attribute:
                elements.loc[elements[attribute].notnull(), "y_prop"] = (
                    elements.loc[elements[attribute].notnull(), "y"] + 0.35
                )
            else:
                elements.loc[:, "y_prop"] = elements["y"] + 0.35

            ac = "display_attribute"
            if is_float_dtype(elements[attribute]):
                elements[ac] = elements[attribute].round(decimals=decimals)
            else:
                elements[ac] = elements[attribute]

            if colorby == "attribute":
                colored = colormap_column(elements, attribute, cmap=cmap, missing=missing)
                elements.loc[:, "attribute_color"] = colored
                colorby = "attribute_color"

            # bokeh configuration

            source = ColumnDataSource(data=elements)

            TOOLS = "hover,save,reset"

            fig = figure(
                title=title,
                tools=TOOLS,
                x_axis_location="above",
                x_range=(elements.x.min() - 0.5, elements.x.max() + 0.5),
                y_range=(elements.y.max() + 0.5, elements.y.min() - 0.5),
                width=width,
                height=height,
                toolbar_location="above",
                toolbar_sticky=False,
            )

            fig.rect("x", "y", 0.9, 0.9, source=source, color=colorby, fill_alpha=0.6)

            # adjust the ticks and axis bounds
            fig.yaxis.bounds = (1, 7)
            fig.axis[1].ticker.num_minor_ticks = 0
            if wide_layout:
                # Turn off tick labels
                fig.axis[0].major_label_text_font_size = "0pt"
                # Turn off tick marks
                fig.axis[0].major_tick_line_color = None  # turn off major ticks
                fig.axis[0].ticker.num_minor_ticks = 0  # turn off minor ticks
            else:
                fig.axis[0].ticker = FixedTicker(ticks=list(range(1, 19)))

            text_props = {
                "source": source,
                "angle": 0,
                "color": "black",
                "text_align": "center",
                "text_baseline": "middle",
            }

            fig.text(
                x="x",
                y="y",
                text="symbol",
                text_font_style="bold",
                text_font_size="15pt",
                **text_props,
            )
            fig.text(
                x="x", y="y_anumber", text="atomic_number", text_font_size="9pt", **text_props
            )
            fig.text(x="x", y="y_name", text="name", text_font_size="6pt", **text_props)
            fig.text(x="x", y="y_prop", text=ac, text_font_size="7pt", **text_props)

            fig.grid.grid_line_color = None

            hover = fig.select(dict(type=HoverTool))
            hover.tooltips = OrderedDict(
                [
                    ("nom", "@nom"),
                    ("name", "@name"),
                    ("famille", "@name_seriesFr"),
                    ("numéro atomique", "@atomic_number"),
                    ("masse molaire", "@atomic_weight"),
                    ("rayon atomique", "@atomic_radius"),
                    ("énergie de première ionisation", "@Ei1"),
                    ("affinité électronique", "@electron_affinity"),
                    ("EN Pauling", "@en_pauling"),
                    ("CEF", "@electronic_configuration"),
                ]
            )

            return fig

        output_notebook()

        fig = periodic_table_bokeh(self.elements, colorby="color")
        show(fig)


class PeriodicTable(TableauPeriodique):
    """
    English-language counterpart of :class:`TableauPeriodique`.

    Same underlying data (fetched through the 'mendeleev' library) and the same
    set of methods, but without the French localization patch: element and
    family/series names are left as mendeleev provides them (English), and all
    printed output (``prop``, ``isotopes``, ``mononucleides``) is in English.

    Everything else — the group-12-is-not-a-transition-metal correction, the
    extra data merged in (first ionization energy), the radiogenic-isotope and
    CIAAW-interval warnings — is shared with :class:`TableauPeriodique` via
    inheritance.
    """

    def _localize_names(self):
        """
        No-op: mendeleev's own 'name' (element name) and 'name_series' (family
        name) columns are already in English, so there is nothing to translate
        here (contrast with :meth:`TableauPeriodique._localize_names`, which adds
        the French 'nom'/'name_seriesFr' columns).
        """
        pass

    def prop(self, elt_id):
        """
        Prints detailed properties of a chemical element.

        Args:
            elt_id (str or int): Element symbol (e.g. 'O') or atomic number (e.g. 8).
        """
        from mendeleev import element

        elt = element(elt_id)
        print(f"Element = {elt.name} ({elt.symbol}, Z = {elt.atomic_number})")
        print(f"Name origin = {elt.name_origin}")
        print()
        print(f"Electron configuration = {elt.ec} = {elt.econf}")
        print(f"Number of unpaired electrons = {elt.ec.unpaired_electrons()}")
        print(f"Group {elt.group_id}, Period {elt.period}, block {elt.block}")
        print(f"Family = {self.elements.loc[elt.atomic_number-1,'name_series']}")
        print()
        print(f"Molar mass = {elt.atomic_weight} g/mol")
        isotopes = ""
        X = elt.symbol
        for i in elt.isotopes:
            if i.abundance is not None:
                isotopes = isotopes + str(i.mass_number)+ "^" + X + f"({i.abundance}%) / "
        print("Natural isotopes = ",isotopes[:-2])
        print()
        if elt.electronegativity(scale='pauling') is None:
            print("Pauling electronegativity = Not defined")
        else:
            print(f"Pauling electronegativity = {elt.electronegativity(scale='pauling')}")
        print(f"1st ionization energy = {elt.ionenergies[1]:.2f} eV")
        if elt.electron_affinity is None:
            print("Electron affinity = Not defined")
        else:
            print(f"Electron affinity = {elt.electron_affinity:.2f} eV")
        print(f"Atomic radius = {elt.atomic_radius:.1f} pm")
        print()
        print("▶ Description: ",elt.description)
        print("▶ Sources: ",elt.sources)
        print("▶ Uses: ",elt.uses)
        print("---------------------------------------------------------------------------------------")
        print()

    def isotopes(self, elt_id):
        """
        Prints an element's isotope table (mass number, natural abundance,
        half-life), sorted by increasing mass number.

        Generalizes the thorium example: for a radioelement, the absence of an
        abundance value simply means that no isotope is stable/natural in
        mendeleev's sense.

        Args:
            elt_id (str or int): Element symbol (e.g. 'Th') or atomic number (e.g. 90).
        """
        from mendeleev import element
        import pandas as pd

        elt = element(elt_id)
        print(f"{elt.symbol} — {elt.name} (Z = {elt.atomic_number})")
        print(f"Molar mass = {elt.atomic_weight} g/mol\n")

        lignes = []
        for iso in sorted(elt.isotopes, key=lambda i: i.mass_number):
            ab = "" if iso.abundance is None else f"{iso.abundance} %"
            hl_val = getattr(iso, "half_life", None)
            if hl_val is None:
                hl = "stable"
            else:
                hl = f"{hl_val} {getattr(iso, 'half_life_unit', '') or ''}".strip()
            masse = getattr(iso, "mass", None)
            lignes.append((iso.mass_number, masse, ab, hl))

        dfIso = pd.DataFrame(lignes, columns=["A", "mass (u)", "abundance", "half-life"])
        dfIso = dfIso.set_index("A")
        try:
            from IPython.display import display
            display(dfIso)
        except ImportError:
            print(dfIso.to_string())

        if elt.atomic_number in TableauPeriodique.exceptionsRadiogenic:
            print()
            print("⚠️  Reclassified as binuclidic by the IUPAC (2013):",
                  TableauPeriodique.exceptionsRadiogenic[elt.atomic_number])

        if elt.atomic_number in TableauPeriodique.intervallesCIAAW:
            mini, maxi = TableauPeriodique.intervallesCIAAW[elt.atomic_number]
            print()
            print(f"⚠️  Naturally variable isotopic composition: IUPAC/CIAAW does not give "
                  f"a single atomic weight but a range [{mini}, {maxi}] g/mol. "
                  f"The value {elt.atomic_weight} g/mol above is a conventional value, "
                  f"not necessarily equal to Σ(abundance × mass) computed from the table above.")

    def mononucleides(self, exclude_radiogenic=True):
        """
        Prints the mononuclidic elements, i.e. elements with a single isotope of
        measurable natural abundance (in mendeleev's sense: abundance is not None).

        Args:
            exclude_radiogenic (bool): if True (default), excludes elements such
                as thorium, reclassified as binuclidic (or more) by the IUPAC in
                2013 because of a trace radiogenic isotope (see
                TableauPeriodique.exceptionsRadiogeniques).
        """
        from mendeleev.fetch import fetch_table
        import pandas as pd

        # Single bulk query (isotopes table + elements table for the symbols),
        # rather than one element(Z) ORM call per element (118 calls: much slower).
        dfIso = fetch_table("isotopes")
        dfElt = fetch_table("elements")[["atomic_number", "symbol", "atomic_weight"]]
        symboles = dict(zip(dfElt["atomic_number"], dfElt["symbol"]))
        massesMolaires = dict(zip(dfElt["atomic_number"], dfElt["atomic_weight"]))

        naturels = dfIso[dfIso["abundance"].notna()]
        compte = naturels.groupby("atomic_number").size()
        # Strictly mononuclidic elements in mendeleev's sense (count == 1). NB: the
        # database already includes the IUPAC/CIAAW 2013 reclassification of
        # thorium (it gives an abundance to both 230-Th and 232-Th), so Th has
        # count == 2 and is NEVER included here, whether exclude_radiogenic is
        # True or False.
        Z_mono = set(compte[compte == 1].index)

        if not exclude_radiogenic:
            # Reinstate elements like Th: one strongly dominant isotope (>99%) plus
            # one or more trace radiogenic isotopes, ignored here.
            Z_mono |= set(TableauPeriodique.exceptionsRadiogeniques.keys())

        mono = []
        for Z in sorted(Z_mono):
            sous = naturels[naturels["atomic_number"] == Z]
            # Dominant isotope (the only one, or the most abundant for the
            # reinstated cases above)
            iso = sous.loc[sous["abundance"].idxmax()]
            hl_val = iso["half_life"] if pd.notna(iso["half_life"]) else None
            hl_unit = iso["half_life_unit"] if pd.notna(iso["half_life_unit"]) else None
            mono.append((Z, symboles[Z], int(iso["mass_number"]), iso["mass"],
                         massesMolaires[Z], hl_val, hl_unit))

        dfMono = pd.DataFrame(
            mono, columns=["Z", "symbol", "A", "mass (u)", "molar mass (g/mol)", "half-life", "unit"]
        )
        dfMono["half-life"] = dfMono.apply(
            lambda r: "stable" if pd.isna(r["half-life"])
            else f"{r['half-life']:g} {r['unit']}", axis=1
        )
        # flags with an asterisk the elements reinstated despite a trace
        # radiogenic isotope (see TableauPeriodique.exceptionsRadiogeniques)
        dfMono["symbol"] = dfMono.apply(
            lambda r: r["symbol"] + "*"
            if r["Z"] in TableauPeriodique.exceptionsRadiogeniques else r["symbol"],
            axis=1
        )
        dfMono = dfMono.drop(columns="unit").set_index("Z")
        try:
            from IPython.display import display
            display(dfMono)
        except ImportError:
            print(dfMono.to_string())
        if not exclude_radiogenic and any(
            Z in TableauPeriodique.exceptionsRadiogeniques for Z, *_ in mono
        ):
            print("* trace radiogenic isotope(s) ignored "
                  "(see TableauPeriodique.exceptionsRadiogeniques)")
        note = " (excluding radiogenic exceptions)" if exclude_radiogenic else ""
        print(f"\n{len(mono)} elements have a single natural isotope{note}")

        return mono

    def display(self):
        """
        Generates and displays the interactive periodic table in a Jupyter notebook.

        Hover over a tile to see the element's properties.

        English-named counterpart of :meth:`TableauPeriodique.afficher`. Also
        reachable as ``afficher()`` (kept as an alias below) so that code
        written against either class can call the same method name.
        """
        from bokeh.plotting import show, output_notebook
        from mendeleev.vis import periodic_table_bokeh

        # This whole block is a copy of mendeleev.vis' own bokeh module. Since
        # periodic_table_bokeh() is not very configurable through args/kwargs, it
        # is adapted here for a custom display (English tooltip labels; 'name' and
        # 'name_series' instead of the French 'nom'/'name_seriesFr' columns, since
        # PeriodicTable does not localize them).

        from collections import OrderedDict

        import pandas as pd
        from pandas.api.types import is_float_dtype

        from bokeh.plotting import figure
        from bokeh.models import HoverTool, ColumnDataSource, FixedTicker

        from mendeleev.vis.utils import colormap_column


        def periodic_table_bokeh(
            elements: pd.DataFrame,
            attribute: str = "atomic_weight",
            cmap: str = "RdBu_r",
            colorby: str = "color",
            decimals: int = 3,
            height: int = 800,
            missing: str = "#ffffff",
            title: str = "Periodic Table",
            wide_layout: bool = False,
            width: int = 1200,
        ):
            """
            Use Bokeh backend to plot the periodic table. Adaptation by Romuald Poteau (romuald.poteau@univ-tlse3.fr) of the orignal periodic_table_bokeh() function of the mendeleev library

            Args:
                elements : Pandas DataFrame with the elements data. Needs to have `x` and `y`
                    columns with coordianates for each tile.
                attribute : Name of the attribute to be displayed
                cmap : Colormap to use, see matplotlib colormaps
                colorby : Name of the column containig the colors
                decimals : Number of decimals to be displayed in the bottom row of each cell
                height : Height of the figure in pixels
                missing : Hex code of the color to be used for the missing values
                title : Title to appear above the periodic table
                wide_layout: wide layout variant of the periodic table
                width : Width of the figure in pixels
            """

            if any(col not in elements.columns for col in ["x", "y"]):
                raise ValueError(
                    "Coordinate columns named 'x' and 'y' are required "
                    "in 'elements' DataFrame. Consider using "
                    "'mendeleev.vis.utils.create_vis_dataframe' and try again."
                )

            # additional columns for positioning of the text

            elements.loc[:, "y_anumber"] = elements["y"] - 0.3
            elements.loc[:, "y_name"] = elements["y"] + 0.2

            if attribute:
                elements.loc[elements[attribute].notnull(), "y_prop"] = (
                    elements.loc[elements[attribute].notnull(), "y"] + 0.35
                )
            else:
                elements.loc[:, "y_prop"] = elements["y"] + 0.35

            ac = "display_attribute"
            if is_float_dtype(elements[attribute]):
                elements[ac] = elements[attribute].round(decimals=decimals)
            else:
                elements[ac] = elements[attribute]

            if colorby == "attribute":
                colored = colormap_column(elements, attribute, cmap=cmap, missing=missing)
                elements.loc[:, "attribute_color"] = colored
                colorby = "attribute_color"

            # bokeh configuration

            source = ColumnDataSource(data=elements)

            TOOLS = "hover,save,reset"

            fig = figure(
                title=title,
                tools=TOOLS,
                x_axis_location="above",
                x_range=(elements.x.min() - 0.5, elements.x.max() + 0.5),
                y_range=(elements.y.max() + 0.5, elements.y.min() - 0.5),
                width=width,
                height=height,
                toolbar_location="above",
                toolbar_sticky=False,
            )

            fig.rect("x", "y", 0.9, 0.9, source=source, color=colorby, fill_alpha=0.6)

            # adjust the ticks and axis bounds
            fig.yaxis.bounds = (1, 7)
            fig.axis[1].ticker.num_minor_ticks = 0
            if wide_layout:
                # Turn off tick labels
                fig.axis[0].major_label_text_font_size = "0pt"
                # Turn off tick marks
                fig.axis[0].major_tick_line_color = None  # turn off major ticks
                fig.axis[0].ticker.num_minor_ticks = 0  # turn off minor ticks
            else:
                fig.axis[0].ticker = FixedTicker(ticks=list(range(1, 19)))

            text_props = {
                "source": source,
                "angle": 0,
                "color": "black",
                "text_align": "center",
                "text_baseline": "middle",
            }

            fig.text(
                x="x",
                y="y",
                text="symbol",
                text_font_style="bold",
                text_font_size="15pt",
                **text_props,
            )
            fig.text(
                x="x", y="y_anumber", text="atomic_number", text_font_size="9pt", **text_props
            )
            fig.text(x="x", y="y_name", text="name", text_font_size="6pt", **text_props)
            fig.text(x="x", y="y_prop", text=ac, text_font_size="7pt", **text_props)

            fig.grid.grid_line_color = None

            hover = fig.select(dict(type=HoverTool))
            hover.tooltips = OrderedDict(
                [
                    ("name", "@name"),
                    ("family", "@name_series"),
                    ("atomic number", "@atomic_number"),
                    ("molar mass", "@atomic_weight"),
                    ("atomic radius", "@atomic_radius"),
                    ("1st ionization energy", "@Ei1"),
                    ("electron affinity", "@electron_affinity"),
                    ("Pauling EN", "@en_pauling"),
                    ("electron config.", "@electronic_configuration"),
                ]
            )

            return fig

        output_notebook()

        fig = periodic_table_bokeh(self.elements, colorby="color")
        show(fig)

    # alias so that generic/polymorphic code can call .afficher() regardless
    # of which of the two classes (TableauPeriodique or PeriodicTable) it holds
    afficher = display
