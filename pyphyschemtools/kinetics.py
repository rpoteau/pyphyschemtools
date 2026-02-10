import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.optimize import curve_fit
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import r2_score as r2
from .visualID_Eng import fg, hl, bg, color
import os
import re

class KORD:
    """
    Initialize the kinetic study with experimental data.
    Reaction: alpha A = beta B
    
    Args:
        - t_exp (array-like): Time values.
        - G_exp (array-like): Measured physical quantity (Absorbance, Conductivity, etc.).

    Fixed Parameters:
        - alpha (float): Stoichiometric coefficient for A reactant (smallest positive integer). Default is 1.0.
        - beta (float): Stoichiometric coefficient for B product (smallest positive integer). Default is 1.0.
        - a0 (float): Initial concentration. Note: G_theo is independent of a0 for Order 1. Default is 1.0.

    Adjustable Variables (Initial Guesses):
        - k_guess (float, optional): Initial estimate for the rate constant. Default is 0.01.
        - G_0_guess (float, optional): Initial estimate for the initial measured value (G at t=0). if None, it will be initialized as G_exp[0]
        - G_inf_guess (float, optional): Initial estimate for the final measured value (G at infinity). if None, it will be initialized as G_exp[-1]
        - b_inf_guess (float, optional): Initial estimate for the final concentration of B ([B] at infinity). if None, it will be initialized as [A](t=0), i.e. a0
        
    Other Args:
        - t_simul_max (float): the maximum time duration for theoretical simulations.
          It defines the range of the time vector [0, t_simul_max}] used to simulate and plot kinetic curves with simulate_plot
        - verbose (bool): If True defaulk), enables detailed optimization logs (recommended).
        - headers (tuple of strings): headers of the t_exp and G_exp arrays read in the excel file

    """
    
    def __init__(self, t_exp=None, G_exp=None, headers=("Time / s", "G"), a0=1.0, alpha=1.0, beta=1.0, k_guess=None,
                 G_0_guess = None, G_inf_guess = None, b_inf_guess=None, t_simul_max = 15, verbose=True):

        # Force conversion to float arrays to prevent 'object' dtype issues.
        # This ensures [0, 0.5, ...] (objects) become [0.0, 0.5, ...] (floats),
        # allowing NumPy mathematical functions (like np.exp) to operate correctly.
        # It is optional if a user wants to simulate a spectrum
        self.t_exp = np.array(t_exp, dtype=float) if t_exp is not None else None
        self.G_exp = np.array(G_exp, dtype=float) if G_exp is not None else None
        self.headers = headers
        # Ensure fixed parameters are treated as native floats.
        # This prevents errors during optimization if values are passed as strings or tuples.
        self.a0 = float(a0)
        self.alpha = float(alpha)
        self.beta = float(beta)

        self.t_simul_max = t_simul_max

        # Logic for Guesses: Priority to provided values, then data, then defaults
        if G_0_guess is not None:
            self.G_0_guess = G_0_guess
        elif self.G_exp is not None:
            self.G_0_guess = self.G_exp[0]
        else:
            self.G_0_guess = 0.0 # Default fallback if no data and no guess (this is for a simulation plot)

        if G_inf_guess is not None:
            self.G_inf_guess = G_inf_guess
        elif self.G_exp is not None:
            self.G_inf_guess = self.G_exp[-1]
        else:
            self.G_inf_guess = 1.0 # Default fallback (this is for a simulation plot)
            
        self.b_inf_guess = b_inf_guess if b_inf_guess is not None else self.a0*self.beta/self.alpha

        self.k_guess = k_guess
        
        self.results = {}
        self.verbose = verbose
        # Color mapping for orders
        self.order_colors = {0: 'blue', 1: 'red', 2: 'green'}
        self.ansi_colors = {0: "\033[94m", 1: "\033[91m", 2: "\033[92m"}
        self.reset = "\033[0m"
        # t_fin = self.a0 / (self.alpha * self.k_guess)
        # print(f"{t_fin=}")

    @staticmethod
    def load_from_excel(file_path, exp_number, sheet_name=0, show_data=True):
        """
        Static method to extract data from an Excel file.
        Selects the pair of columns (t, G) corresponding to the experiment number.
        Also loads parameters (a0, alpha, beta)
        
        Format:
            - Row 1: Headers for t and G
            - Row 2: [A]0 value (in the G column)
            - Row 3: alpha value (in the G column)
            - Row 4: beta value (in the G column)
            - Row 5+: [t, G] data points
        
        """
        # 1. Check if file exists
        if not os.path.exists(file_path):
            print(f"❌ Error: The file '{file_path}' was not found.")
            return None

        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        except Exception as e:
            print(f"❌ Error while reading the Excel file: {e}")
            return None

        idx_t, idx_G = 2*(exp_number-1), 2*(exp_number-1)+1

        # --- Parameter Extraction (Now looking in the G column: idx_G) ---
        def parse_param(val):
            if isinstance(val, str):
                match = re.search(r"(\d+\.?\d*)", val)
                return float(match.group(1)) if match else 1.0
            return float(val) if pd.notnull(val) else 1.0
            
        total_cols = len(df.columns)
        num_experiments = total_cols // 2
        print(f"Experiments detected: {num_experiments}")
        
        # Parameters are expected in rows 2, 3, and 4 of Excel (indices 0, 1, 2)
        a0    = parse_param(df.iloc[0, idx_G])
        alpha = parse_param(df.iloc[1, idx_G])
        beta  = parse_param(df.iloc[2, idx_G])

        # --- Data Extraction (From index 3 onwards) ---
        data = df.iloc[3:, [idx_t, idx_G]].dropna()
        
        label_t = KORD._clean_pandas_suffix(df.columns[idx_t])
        label_G = KORD._clean_pandas_suffix(df.columns[idx_G])
        data.columns = [label_t, label_G]

        print(f"✅ Loaded: {label_G} (Exp {exp_number})")
        print(f"   [Parameters from {label_G}] a0: {a0:.4e} mol.L-1 | alpha: {alpha} | beta: {beta}\n")
        
        if show_data:
            from IPython.display import display
            display(data)
        
        return data.iloc[:, 0].values, data.iloc[:, 1].values, (label_t, label_G), (a0, alpha, beta)
        
    @staticmethod
    def _clean_pandas_suffix(name):
        """
        Safely removes the '.1', '.2' suffixes added by Pandas for duplicate names.
        Only matches a dot followed by digits at the VERY END of the string.
        Example: 'mol.L-1.1' -> 'mol.L-1'
        """
        return re.sub(r'\.\d+$', '', str(name))
        
    def simulate_plot(self, save_img=None):
        """
        Plots the initial guesses for all three kinetic orders (0, 1, 2),
        for example to visualize the starting point of a fit.

        If save_img is provided, saves the plot (png, svg, jpg, pdf according to the extension).
        Vectorial svg is recommended
        """
        plt.figure(figsize=(10, 6))
        
        # 2. Generate smooth time vector for simulation
        t_sim = np.linspace(0, self.t_simul_max, 500)
        
        # 3. Define the models to loop through
        models = {0: self.G0_theo, 1: self.G1_theo, 2: self.G2_theo}
        
        for order, func in models.items():
            # Use the color defined in your __init__
            color = self.order_colors[order]
            
            # Simulate using the guess values from __init__
            if order != 1:
                G_sim = func(t_sim, 
                             k=self.k_guess, 
                             G0=self.G_0_guess, 
                             Ginf=self.G_inf_guess, 
                             binf=self.b_inf_guess)
            else:
                G_sim = func(t_sim, 
                             k=self.k_guess, 
                             G0=self.G_0_guess, 
                             Ginf=self.G_inf_guess)

            
            plt.plot(t_sim, G_sim, 
                     label=f"Guess Order {order}", 
                     color=color, lw=2.5, linestyle='-')

        # Formatting
        plt.xlabel(self.headers[0])
        plt.ylabel(self.headers[1])
        plt.title(f"Initial Kinetic Guesses for {self.headers[1]}")
        
        # Add a text box with the guess values for clarity
        guess_text = (f"Guesses:\n"
                      f"k: {self.k_guess:.2e}\n"
                      f"G0: {self.G_0_guess:.4f}\n"
                      f"Ginf: {self.G_inf_guess:.4f}\n"
                      f"binf: {self.b_inf_guess:.2e}")
        
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
        
    def G0_theo(self, t, k, G0, Ginf, binf):
        """
        Model for Order 0 kinetics
        """
        return np.where(t <= binf/(self.beta*k), G0 + self.beta * k*t * (Ginf-G0) / binf, Ginf)
        
    def G1_theo(self, t, k, G0, Ginf):
        """Model for Order 1 kinetics"""
        return Ginf + np.exp(-self.alpha * k * t) * (G0 - Ginf)

    def G2_theo(self, t, k, G0, Ginf, binf):
        """Model for Order 2 kinetics"""
        # return np.where(t !=0, G0 + (Ginf - G0)/(1 + self.beta/(binf*self.alpha**2*k*t)), G0)
        tterm = binf*self.alpha**2*k*t
        return G0 + (Ginf-G0)*tterm/(tterm+self.beta)

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
    
    def fit(self, k_guess, G_0_guess, G_inf_guess, A_0_guess, order=1):
        """
        Fits the selected kinetic model (Order 0, 1, or 2) to the experimental data.

        This method uses non-linear least squares (scipy.optimize.curve_fit) to 
        simultaneously optimize the rate constant (k), the initial and final 
        physical property values (G0, Ginf), and the final product concentration (binf).
        Physical constraints are applied via parameter bounds to ensure k and binf 
        remain positive.

        It includes automated 'Smart Guessing' for k, based on initial rates and computes 
        information criteria (AIC/BIC) for model selection.

        Args:
            k_guess (float): Initial estimate for the rate constant (k).
            G_0_guess (float): Initial estimate for the physical property at t=0.
            G_inf_guess (float): Initial estimate for the physical property at t=infinity.
            b_inf_guess (float): Initial estimate for the final product concentration [B].
            order (int, optional): Kinetic order of the reaction (0, 1, or 2). Defaults to 1.

        Returns:
            dict: A dictionary containing the optimized parameters ('k', 'G0', 'Ginf', 'binf') 
                  and statistical metrics ('RMSE', 'R2', 't_half'). Returns None if 
                  optimization fails to converge.

        Notes:
            - For Order 1, the timing is independent of binf, and alpha is coupled with k; hence binf is kept fixed during optimization.
            - For Orders 0 and 2, binf and stoichiometry (alpha, beta) explicitly define the reaction's capacity and end-point timing,
              ensuring physical consistency between the plateau and the slope.
        """
        models = {0: self.G0_theo, 1: self.G1_theo, 2: self.G2_theo}
        func = models[order]

        # smart guess for k
        n = max(2, len(self.t_exp) // 10) 
        delta_G = abs(self.G_exp[n] - self.G_exp[0])
        delta_t = self.t_exp[n] - self.t_exp[0]
        initial_slope = delta_G / delta_t if delta_t > 0 else 1e-5

        # 2. Apply the smart guess logic
        delta_G_total = abs(self.G_inf_guess - self.G_0_guess)
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
        
        # Initial guess vector [k, G0, Ginf, a0]
        if order != 1:
            p0 = [final_k_guess, self.G_0_guess, self.G_inf_guess, self.b_inf_guess]
        else:
            p0 = [final_k_guess, self.G_0_guess, self.G_inf_guess]
        
        if self.verbose:
            c = self.ansi_colors[order]
            print(f"{c}{hl.BOLD}--- Order {order} ---{self.reset}")
            if order != 1:
                print(f"  GUESS: k: {p0[0]:.2e} | G0: {p0[1]:.4f} | Ginf: {p0[2]:.4f} | binf: {p0[3]:.3e}")
            else:
                print(f"  GUESS: k: {p0[0]:.2e} | G0: {p0[1]:.4f} | Ginf: {p0[2]:.4f}")
            
        try:
            # k, G0, Ginf, binf
            if order != 1:
                lower_bounds = [0, -np.inf, -np.inf, 0]
                upper_bounds = [np.inf, np.inf, np.inf, np.inf]
            else:
                lower_bounds = [0, -np.inf, -np.inf]
                upper_bounds = [np.inf, np.inf, np.inf]
            popt, _ = curve_fit(func, self.t_exp, self.G_exp, p0=p0, bounds=(lower_bounds, upper_bounds))
            if order != 1:
                k_opt, G0_opt, Ginf_opt, binf_opt = popt
            else:
                k_opt, G0_opt, Ginf_opt = popt
                binf_opt = self.a0*self.beta/self.alpha
                
            G_theo = func(self.t_exp, *popt)

            RMSE = rmse(self.G_exp, G_theo)
            R2 = r2(self.G_exp, G_theo)
            
            # t1/2 calculation
            t_half = self.halflife(order, binf_opt, k_opt)

            # AIC & BIC calculations
            # n: number of data points, p: number of parameters
            n = len(self.G_exp)
            p = len(popt)
            
            # Using the simplified formula for AIC/BIC (assuming Gaussian errors)
            # RSS (Residual Sum of Squares) = RMSE^2 * n
            aic = n * np.log(RMSE**2) + 2 * p
            bic = n * np.log(RMSE**2) + p * np.log(n)
            
            if self.verbose:
                # Aligned exactly with the GUESS print for easy comparison
                if order != 1:
                    print(f"  OPTIM: k: {k_opt:.2e} | G0: {G0_opt:.4f} | Ginf: {Ginf_opt:.4f} | binf: {binf_opt:.3e}")
                else:
                    print(f"  OPTIM: k: {k_opt:.2e} | G0: {G0_opt:.4f} | Ginf: {Ginf_opt:.4f}")
                print(f"  ✅ RMSE: {RMSE:.2e}")
                print(f"       R2: {R2:.2f}")
                print(f"      AIC: {aic:.2f} (lower is better)")
                print(f"      BIC: {bic:.2f} (more strict on complexity)")
            
            self.results[order] = {
                'k': k_opt, 'G0': G0_opt, 'Ginf': Ginf_opt, 'binf': binf_opt,
                'RMSE': RMSE, 'R2': R2, 'AIC': aic, 'BIC': bic,
                't_half': t_half, 'G_theo': G_theo
            }
            return self.results[order]
        except Exception as e:
            print(f"Could not fit order {order}: {e}")
            return None

    def plot_all_fits(self, save_img=None):
        """
        Plots experimental data and all three kinetic models for visual comparison.
        If save_img is provided, saves the plot (png, svg, jpg, pdf according to the extension).
        Vectorial svg is recommended
        """
        mosaic = [['fit','o0'],
                  ['fit','o1'],
                  ['fit','o2']]
        cgraph = ['o0', 'o1', 'o2']
        background_graph = ['#dbeeff', '#ffe4e4', '#d2ffdb']
        Fig, Graph = plt.subplot_mosaic(mosaic, figsize=(14,10), gridspec_kw=dict(width_ratios = [1.4, 1]))
        
        ## Fit ###############
        Graph['fit'].scatter(self.t_exp, self.G_exp, label=f"Experimental ($a_0$ = {self.a0:.2e})", color='black', marker="x", s=35, alpha=0.7)

        t_smooth = np.linspace(self.t_exp.min(), self.t_exp.max(), 500)
        models = {0: self.G0_theo, 1: self.G1_theo, 2: self.G2_theo}

        for order in [0, 1, 2]:
            if order not in self.results:
                self.fit(self.k_guess, self.G_0_guess, self.G_inf_guess, self.b_inf_guess, order)
            
            res = self.results[order]

            if order != 1:
                G_smooth = models[order](t_smooth, res['k'], res['G0'], res['Ginf'], res['binf'])
            else:
                G_smooth = models[order](t_smooth, res['k'], res['G0'], res['Ginf'])
            
            Graph['fit'].plot(t_smooth, G_smooth, 
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
        
        Graph['fit'].axhline(best_res['G0'], color=best_color, linestyle='--', alpha=0.6)
        Graph['fit'].axhline(best_res['Ginf'], color=best_color, linestyle='--', alpha=0.6)
        Graph['fit'].yaxis.grid(
            True, 
            which='major', 
            linestyle='--',   # Ligne hachurée
            linewidth=0.8,    # Épaisseur
            color='gray',     # Couleur discrète
            alpha=0.5         # Transparence pour ne pas surcharger
        )
        # Désactiver la grille verticale (optionnel, pour garder le focus sur G)
        Graph['fit'].xaxis.grid(False)
        
        # Add the second axis for the fitted values
        ax2 = Graph['fit'].twinx()
        ax2.set_ylim(Graph['fit'].get_ylim()) # Keep scales aligned
        ax2.set_yticks([best_res['G0'], best_res['Ginf']])
        ax2.set_yticklabels([f"G0_fit={best_res['G0']:.3f}", f"Ginf_fit={best_res['Ginf']:.3f}"])
        ax2.tick_params(axis='y', labelcolor=best_color)
        
        Graph['fit'].set_xlabel("Time",size=12,fontweight='bold',color='b')
        Graph['fit'].set_ylabel("G property",size=12,fontweight='bold',color='b')
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
            if i not in self.results: self.fit(self.k_guess, self.G_0_guess,
                                               self.G_inf_guess, self.a0, i)
            
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
            print(f"G_exp(t=0): {self.G_exp[0]:.3e}")
            print(f"G_fit(t=0): {res['G0']:.3e}")
            print(f"G_fit(t=inf): {res['Ginf']:.3e}")
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
