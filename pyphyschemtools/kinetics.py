import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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
        - A0 (float): Initial concentration. Note: G_theo is independent of A0 for Order 1. Default is 1.0.

    Adjustable Variables (Initial Guesses):
        - k_guess (float, optional): Initial estimate for the rate constant. Default is 0.01.
        - G_0_guess (float, optional): Initial estimate for the initial measured value (G at t=0). if None, it will be initialized as G_exp[0]
        - G_inf_guess (float, optional): Initial estimate for the final measured value (G at infinity). if None, it will be initialized as G_exp[-1]
        
    Other Args:
        - verbose (bool): If True, enables debug messages and detailed optimization logs.
        - headers (tuple of strings): headers of the t_exp and G_exp arrays read in the excel file
    """
    
    def __init__(self, t_exp, G_exp, headers, A0=1.0, alpha=1.0, beta=1.0, k_guess=0.01,
                 G_0_guess = None, G_inf_guess = None, verbose=False):

        # Force conversion to float arrays to prevent 'object' dtype issues.
        # This ensures [0, 0.5, ...] (objects) become [0.0, 0.5, ...] (floats),
        # allowing NumPy mathematical functions (like np.exp) to operate correctly.
        self.t_exp = np.array(t_exp, dtype=float)
        self.G_exp = np.array(G_exp, dtype=float)
        self.headers = headers
        # Ensure fixed parameters are treated as native floats.
        # This prevents errors during optimization if values are passed as strings or tuples.
        self.A0 = float(A0)
        self.alpha = float(alpha)
        self.beta = float(beta)

        self.k_guess = k_guess
        self.G_0_guess = G_0_guess if G_0_guess is not None else G_exp[0]
        self.G_inf_guess = G_inf_guess if G_inf_guess is not None else G_exp[-1]
        self.results = {}
        self.verbose = verbose
        # Color mapping for orders
        self.order_colors = {0: 'red', 1: 'green', 2: 'blue'}
        self.ansi_colors = {0: "\033[91m", 1: "\033[92m", 2: "\033[94m"}
        self.reset = "\033[0m"
        # t_fin = self.A0 / (self.alpha * self.k_guess)
        # print(f"{t_fin=}")

    @staticmethod
    def load_from_excel(file_path, exp_number, sheet_name=0, show_data=True):
        """
        Static method to extract data from an Excel file.
        Selects the pair of columns (t, G) corresponding to the experiment number.
        Also loads parameters (A0, alpha, beta)
        Format:
            Row 1: Headers for t and G
            Row 2: [A]0 value (in the G column)
            Row 3: alpha value (in the G column)
            Row 4: beta value (in the G column)
            Row 5+: [t, G] data points
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
        print(f"   [Parameters from {label_G}] A0: {a0:.4e} mol.L-1 | alpha: {alpha} | beta: {beta}\n")
        
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
        
    def G0_theo(self, t, k, G0, Ginf):
        """
        Continuous linear model for optimization.
        Allows A(t) to be negative so curve_fit can find the gradient.
        """
        return G0 + (float(self.alpha) * k * t / float(self.A0)) * (Ginf - G0)

    def G1_theo(self, t, k, G0, Ginf):
        """Model for Order 1 kinetics"""
        return Ginf + np.exp(-self.alpha * k * t) * (G0 - Ginf)

    def G2_theo(self, t, k, G0, Ginf):
        """Model for Order 2 kinetics"""
        return Ginf - (Ginf - G0) / (1 + self.A0 * self.alpha * k * t)

    def fit(self, k_guess, G_0_guess, G_inf_guess, order=1):
        """
        Fits the chosen kinetic model to the experimental data, with order=order (Default: 1)
        verbose=True: prints the initial guess vector p0
        """
        models = {0: self.G0_theo, 1: self.G1_theo, 2: self.G2_theo}
        func = models[order]
        
        # Initial guess vector [k, G0, Ginf]
        p0 = [self.k_guess, self.G_0_guess, self.G_inf_guess]
        
        # 1. Inspection des types des paramètres de classe
        # print(f"--- TYPE CHECK (Order {order}) ---")
        # print(f"self.A0:    {type(self.A0)} | Value: {self.A0}")
        # print(f"self.alpha: {type(self.alpha)} | Value: {self.alpha}")
        # print("----------------------------------")
        # print(f"p0 types:  {[type(x) for x in p0]}")
        # print(f"t_exp type: {type(self.t_exp)} | dtype: {self.t_exp.dtype}")
        # print(f"G_exp type: {type(self.G_exp)} | dtype: {self.G_exp.dtype}")
        # print("----------------------------------")
        # print(self.t_exp)
        # print("----------------------------------")
        
        if self.verbose:
            c = self.ansi_colors[order]
            print(f"{c}--- DEBUG INITIAL GUESS (Order {order}) ---{self.reset}")
            print(f"  GUESS: k: {p0[0]:.2e} | G0: {p0[1]:.4f} | Ginf: {p0[2]:.4f}")
            
        try:
            popt, _ = curve_fit(func, self.t_exp, self.G_exp, p0=p0)
            k_opt, G0_opt, Ginf_opt = popt
            G_theo = func(self.t_exp, *popt)

            rmsd = np.sqrt(np.mean((self.G_exp - G_theo)**2))
            
            # t1/2 calculation
            if order == 0: t_half = self.A0 / (2 * self.alpha * k_opt)
            elif order == 1: t_half = np.log(2) / (self.alpha * k_opt)
            else: t_half = 1 / (self.A0 * self.alpha * k_opt)

            if self.verbose:
                # Aligned exactly with the GUESS print for easy comparison
                print(f"  OPTIM: k: {k_opt:.2e} | G0: {G0_opt:.4f} | Ginf: {Ginf_opt:.4f}")
                print(f"  ✅ RMSD: {rmsd:.2e}")
            
            self.results[order] = {
                'k': k_opt, 'G0': G0_opt, 'Ginf': Ginf_opt, 
                'rmsd': rmsd, 't_half': t_half, 'G_theo': G_theo
            }
            return self.results[order]
        except Exception as e:
            print(f"Could not fit order {order}: {e}")
            return None

    def plot_all_fits(self):
        """Plots experimental data and all three kinetic models for visual comparison."""
        plt.figure(figsize=(10, 6))
        ax1 = plt.gca()
        
        ax1.scatter(self.t_exp, self.G_exp, label="Experimental", color='black', s=35, alpha=0.5)

        t_smooth = np.linspace(self.t_exp.min(), self.t_exp.max(), 500)
        models = {0: self.G0_theo, 1: self.G1_theo, 2: self.G2_theo}

        for order in [0, 1, 2]:
            if order not in self.results:
                self.fit(self.k_guess, self.G_0_guess, self.G_inf_guess, order)
            
            res = self.results[order]

            G_smooth = models[order](t_smooth, res['k'], res['G0'], res['Ginf'])
            # if order == 0:
            #     t_fin = self.A0 / (self.alpha * res['k'])
            #     print(f"{t_fin=}")
            
            ax1.plot(t_smooth, G_smooth, 
                     label=f"Order {order} (RMSD: {res['rmsd']:.2e})", 
                     color=self.order_colors[order], lw=2)

        # 3. Add horizontal lines for the BEST model
        best_order = self.get_best_order(verbose=False)
        best_res = self.results[best_order]
        best_color = self.order_colors[best_order]
        
        ax1.axhline(best_res['G0'], color=best_color, linestyle='--', alpha=0.6)
        ax1.axhline(best_res['Ginf'], color=best_color, linestyle='--', alpha=0.6)

        # 4. Add the second axis for the fitted values
        ax2 = ax1.twinx()
        ax2.set_ylim(ax1.get_ylim()) # Keep scales aligned
        ax2.set_yticks([best_res['G0'], best_res['Ginf']])
        ax2.set_yticklabels([f"G0_fit={best_res['G0']:.3f}", f"Ginf_fit={best_res['Ginf']:.3f}"])
        ax2.tick_params(axis='y', labelcolor=best_color)
        
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Quantity G")
        ax1.set_title(f"KORD Kinetic Models Comparison (0, 1, 2). Label exp = {self.headers[1]}")
        ax1.legend()
        # ax1.grid(True, linestyle=':', alpha=0.6)
        plt.show()
        
    def get_best_order(self, verbose=True):
        """Determines and prints the best model based on the lowest RMSD."""
        for i in [0, 1, 2]:
            if i not in self.results: self.fit(self.k_guess, self.G_0_guess,
                                               self.G_inf_guess, i)
            
        best_order = min(self.results, key=lambda x: self.results[x]['rmsd'])
        res = self.results[best_order]
        
        if verbose:
            # ANSI Escape sequences for color in terminal/notebook
            reset = self.reset
            color = self.ansi_colors[best_order]
    
            print(f"--- {color}KORD CONCLUSION ---")
            print(f"Best model: ORDER {best_order}")
            print(f"Initial concentration: {self.A0:.3e} mol.L-1")
            print(f"alpha: {self.alpha}")
            print(f"beta: {self.beta}")
            print()
            print(f"RMSD: {res['rmsd']:.2e}")
            print(f"k: {res['k']:.3e}")
            print(f"t1/2: {res['t_half']:.3f}")
            print()
            print(f"G0_exp: {self.G_exp[0]:.3e}")
            print(f"G0_fit: {res['G0']:.3e}")
            print(f"Ginf_fit: {res['Ginf']:.3e}")
            print(f"------------------------{reset}")
            return 
        else:
            return best_order
