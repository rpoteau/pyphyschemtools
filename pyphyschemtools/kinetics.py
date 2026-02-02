import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import re

class KORD:
    def __init__(self, t_exp, G_exp, headers, A0=1.0, alpha=1.0, k_guess=0.01, verbose=False):
        """
        Initialize the kinetic study with experimental data.
        
        Parameters:
        t_exp (array-like): Time values.
        G_exp (array-like): Measured physical quantity (Absorbance, Conductivity, etc.).
        A0 (float): Initial concentration. Default is 1.0.
        alpha (float): Stoichiometric coefficient. Default is 1.0.
        k_guess (float): rate constant. Default is 0.01
        verbose (bool): kind of debug variable
        """
        self.t_exp = np.array(t_exp)
        self.G_exp = np.array(G_exp)
        self.headers = headers
        self.A0 = A0
        self.alpha = alpha
        self.k_guess = k_guess
        self.results = {}
        self.verbose = verbose
        # Color mapping for orders
        self.order_colors = {0: 'red', 1: 'green', 2: 'blue'}
        self.ansi_colors = {0: "\033[91m", 1: "\033[92m", 2: "\033[94m"}
        self.reset = "\033[0m"

    @staticmethod
    def load_from_excel(file_path, exp_number, sheet_name=0, show_data=True):
        """
        Static method to extract data from an Excel file.
        Selects the pair of columns (t, G) corresponding to the experiment number.
        """
        # 1. Check if file exists
        if not os.path.exists(file_path):
            print(f"❌ Error: The file '{file_path}' was not found.")
            return None, None

        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        except Exception as e:
            print(f"❌ Error while reading the Excel file: {e}")
            return None, None

        total_cols = len(df.columns)
        num_experiments = total_cols // 2
        
        print(f"Experiments detected: {num_experiments}")
        
        for i in range(1, num_experiments + 1):
            it, ig = 2*(i-1), 2*(i-1)+1
            # Clean names for display using regex
            n_t = KORD._clean_pandas_suffix(df.columns[it])
            n_g = KORD._clean_pandas_suffix(df.columns[ig])
            pts = len(df.iloc[:, [it, ig]].dropna())
            print(f"  [{i}] {n_t} | {n_g} ({pts} points)")
        print(f"-----------------------\n")
        
        if exp_number < 1 or exp_number > num_experiments:
            print(f"⚠️ Error: Experiment {exp_number} does not exist (Choice: 1 to {num_experiments}).")
            return None, None, None, None
        
        # Position-based extraction
        idx_t, idx_G = 2*(exp_number-1), 2*(exp_number-1)+1
        label_t = KORD._clean_pandas_suffix(df.columns[idx_t])
        label_G = KORD._clean_pandas_suffix(df.columns[idx_G])

        data = df.iloc[:, [idx_t, idx_G]].dropna()
        data.columns = [label_t, label_G]
        print(f"✅ Loaded: {label_G} (Exp {exp_number})\n")
        
        if show_data: display(data)
        
        return data.iloc[:, 0].values, data.iloc[:, 1].values, (label_t, label_G)
        
    @staticmethod
    def _clean_pandas_suffix(name):
        """
        Safely removes the '.1', '.2' suffixes added by Pandas for duplicate names.
        Only matches a dot followed by digits at the VERY END of the string.
        Example: 'mol.L-1.1' -> 'mol.L-1'
        """
        return re.sub(r'\.\d+$', '', str(name))
        
    def G0_theo(self, t, k, G0, Ginf):
        """Model for Order 0 kinetics."""
        t_fin = self.A0 / (self.alpha * k)
        return np.where(t <= t_fin, 
                        G0 + (self.alpha * k * t / self.A0) * (Ginf - G0), 
                        Ginf)

    def G1_theo(self, t, k, G0, Ginf):
        """Model for Order 1 kinetics."""
        return Ginf + np.exp(-self.alpha * k * t) * (G0 - Ginf)

    def G2_theo(self, t, k, G0, Ginf):
        """Model for Order 2 kinetics."""
        return Ginf - (Ginf - G0) / (1 + self.A0 * self.alpha * k * t)

    def fit(self, order=1, k_guess=None):
        """
        Fits the chosen kinetic model to the experimental data, with order=order (Default: 1)
        verbose=True: prints the initial guess vector p0
        """
        models = {0: self.G0_theo, 1: self.G1_theo, 2: self.G2_theo}
        func = models[order]
        
        # Initial guess [k, G0, Ginf]
        current_k = k_guess if k_guess is not None else self.k_guess
        # Initial guess vector [k, G0, Ginf]
        p0 = [current_k, self.G_exp[0], self.G_exp[-1]]
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
        plt.scatter(self.t_exp, self.G_exp, label="Experimental", color='black', s=35, alpha=0.5)

        t_smooth = np.linspace(self.t_exp.min(), self.t_exp.max(), 500)
        models = {0: self.G0_theo, 1: self.G1_theo, 2: self.G2_theo}
        
        for order in [0, 1, 2]:
            if order not in self.results:
                self.fit(order)
            
            res = self.results[order]

            G_smooth = models[order](t_smooth, res['k'], res['G0'], res['Ginf'])
            
            plt.plot(t_smooth, G_smooth, 
                     label=f"Order {order} (RMSD: {res['rmsd']:.2e})", 
                     color=self.order_colors[order], lw=2)
        
        plt.xlabel("Time")
        plt.ylabel("Quantity G")
        plt.title(f"KORD Kinetic Models Comparison (0, 1, 2). Label exp = {self.headers[1]}")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.show()
        
    def get_best_order(self):
        """Determines and prints the best model based on the lowest RMSD."""
        for i in [0, 1, 2]:
            if i not in self.results: self.fit(i)
            
        best_order = min(self.results, key=lambda x: self.results[x]['rmsd'])
        res = self.results[best_order]
        
        # ANSI Escape sequences for color in terminal/notebook
        reset = self.reset
        color = self.ansi_colors[best_order]

        print(f"--- {color}KORD CONCLUSION ---")
        print(f"Best model: ORDER {best_order}")
        print(f"RMSD: {res['rmsd']:.2e}")
        print(f"k: {res['k']:.3e}")
        print(f"t1/2: {res['t_half']:.3f}")
        print(f"------------------------{reset}")
        return
