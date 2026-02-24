############################################################
#                    Nano tools
############################################################
from .visualID_Eng import fg, bg, hl
from .core import centerTitle, centertxt

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve
from scipy.stats import norm, lognorm

import os, io
from pathlib import Path

class NanoparticleDistribution:
    """
    A class to analyze and fit nanoparticle size distributions using Gaussian models.
    
    This tool provides methods for curve fitting, statistical breakdown by size bins, 
    and publication-ready visualization.
    """

    def __init__(self, sizes=None, counts=None):
        """
        Initialize the distribution with experimental data.
        
        Args:
            sizes (array-like): Measured particle sizes (e.g., in nm).
            counts (array-like): Number of nanoparticles for each size.
        """
        self.sizes = np.array(sizes)
        self.counts = np.array(counts)
        self.params = None  # Will store [A, mu, sigma] after fitting
        self.cov = None     # Covariance matrix for error analysis
        self.model_type = 'gaussian'  # Default model
        self._results_dict = {}       # Private storage for results

    @classmethod
    def from_gaussian_params(cls, mu, sigma, total_n=1000):
        """
        Instantiate with a specific total population (total_n).
        The amplitude is calculated so the integral equals total_n.
        """
        instance = cls()
        
        # Calculate amplitude to ensure the sum (area) equals total_n
        # A = N / (sigma * sqrt(2 * pi))
        amplitude = total_n / (sigma * np.sqrt(2 * np.pi))
        
        instance.params = np.array([amplitude, mu, sigma])
        
        # Generate enough points so that np.sum(counts * dx) is accurate
        # But for your binned stats, we just need to store the intention
        instance.sizes = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
        instance.counts = instance._gaussian_model(instance.sizes, *instance.params)
        
        # We manually set a 'total_n' attribute or just rely on the math
        instance.total_n_expected = total_n 
        
        instance.cov = np.zeros((3, 3))
        instance.total_n_expected = total_n
        return instance

    @classmethod
    def from_polydispersity(cls, mu, pd_pct, amplitude=1000):
        """
        Instantiate the class using mean diameter and Polydispersity Index (CV%).
        
        Formula: sigma = (PD% / 100) * mu
        """
        sigma = (pd_pct / 100) * mu
        return cls.from_gaussian_params(mu, sigma, amplitude)

    @classmethod
    def from_saxs_data(cls, mu_vol, pd_vol_pct, amplitude=1000):
        """
        Instantiate the class by converting SAXS (volume-weighted) parameters 
        into Number-weighted parameters using numerical cube-weighting.
        
        This method uses a robust iterative solver to find the number-weighted 
        mean (mu_n) that corresponds to the observed SAXS volume-weighted mean.
        """
        
        cv = pd_vol_pct / 100

        # --- 1. Hatch-Choate Approximation (for comparison) ---
        mu_n_hc = mu_vol / (1 + 3 * (cv**2))
        sigma_n_hc = mu_n_hc * cv

        # --- 2. Numerical Integration Approach ---
        def objective(mu_n_guess):
            # fsolve can pass an array, we need the scalar value
            m = float(mu_n_guess[0]) if isinstance(mu_n_guess, np.ndarray) else float(mu_n_guess)
            s = m * cv
            
            # Use a fixed number of points and a range based on the target mu_vol 
            # to keep the array size and limits stable for fsolve
            x = np.linspace(mu_vol * 0.1, mu_vol * 2.0, 1000)
            
            # Calculate Gaussian
            y_num = np.exp(-0.5 * ((x - m) / s)**2)
            y_vol = y_num * (x**3)
            
            # Avoid division by zero
            denom = np.trapz(y_vol, x)
            if denom == 0:
                return 1e6 # Penalty for invalid mu_n
                
            calc_mu_vol = np.trapz(y_vol * x, x) / denom
            return calc_mu_vol - mu_vol

        # Solve for the true mu_n
        # We use mu_n_hc as a much better starting guess than mu_vol
        solution = fsolve(objective, x0=mu_n_hc)
        mu_n_num = float(solution[0])
        sigma_n_num = mu_n_num * cv
        
        # --- Output Comparison ---
        print(f"\n{' SAXS to Number Conversion ':-^60}")
        print(f"Input (SAXS)       : μ={mu_vol:.3f} nm, PD={pd_vol_pct:.1f}%")
        print("-" * 60)
        print(f"{'Method':<20} | {'Mean (nm)':<15} | {'Sigma (nm)':<15}")
        print(f"{'Hatch-Choate':<20} | {mu_n_hc:>14.3f} | {sigma_n_hc:>14.3f}")
        print(f"{'Numerical (Full)':<20} | {mu_n_num:>14.3f} | {sigma_n_num:>14.3f}")
        print("-" * 60)
        
        return cls.from_gaussian_params(mu_n_num, sigma_n_num, amplitude)

    @classmethod
    def from_lognormal_params(cls, median, sigma_g, amplitude=1000):
        """
        Instantiates the class using Log-Normal parameters.
        
        Args:
            median (float): The median diameter (nm).
            sigma_g (float): The geometric standard deviation.
            amplitude (float): Total number of particles.
        """
        instance = cls()
        instance.model_type = 'lognormal'
        
        # Calculate arithmetic equivalents for summary reporting
        ln_sig_g = np.log(sigma_g)
        arith_mean = median * np.exp((ln_sig_g**2) / 2)
        arith_sigma = arith_mean * np.sqrt(np.exp(ln_sig_g**2) - 1)
        
        instance._results_dict = {
            'mean': arith_mean,
            'median': median,
            'sigma': arith_sigma,
            'sigma_g': sigma_g,
            'amplitude': amplitude,
            'cv_percentage': (arith_sigma / arith_mean) * 100,
            'fwhm': 0 # FWHM is less standard for lognormal, can be calculated if needed
        }
        
        # Generate representative data for plotting
        x = np.linspace(median / (sigma_g**2), median * (sigma_g**2), 500)
        instance.sizes = x
        instance.counts = instance._lognormal_model(x, amplitude, median, sigma_g)
        instance.params = [amplitude, median, sigma_g]
        instance.total_n_expected = amplitude
        return instance

    @classmethod
    def from_saxs_lognormal(cls, mu_vol, pd_vol_pct, total_n=1000):
        """
        Converts SAXS Volume-weighted Mean to Number-weighted Log-normal parameters
        using exact Hatch-Choate identities.
        """
        import math
        cv = pd_vol_pct / 100
        ln_sig_g_sq = math.log(1 + cv**2)
        sigma_g = math.exp(math.sqrt(ln_sig_g_sq))
        
        # Exact Hatch-Choate: median_n = mu_vol / exp(3.5 * ln(sigma_g)^2)
        median_n = mu_vol / math.exp(3.5 * ln_sig_g_sq)
        
        print(f"\n{' SAXS to Log-Normal Conversion (Exact) ':-^60}")
        print(f"Input (SAXS Vol): μ={mu_vol:.3f} nm, PD={pd_vol_pct:.1f}%")
        print(f"Result (Num)    : Median={median_n:.3f} nm, σg={sigma_g:.3f}")
        print("-" * 60)
        
        return cls.from_lognormal_params(median=median_n, sigma_g=sigma_g, amplitude=total_n)

    @staticmethod
    def _lognormal_model(x, amplitude, median, sigma_g):
        """Probability Density Function for Log-normal."""
        x = np.where(x <= 0, 1e-9, x)
        ln_sig_g = np.log(sigma_g)
        term1 = amplitude / (x * ln_sig_g * np.sqrt(2 * np.pi))
        term2 = np.exp(- (np.log(x / median))**2 / (2 * ln_sig_g**2))
        return term1 * term2
        
    @staticmethod
    def _gaussian_model(x, A, mu, sigma):
        """
        Internal Gaussian model function.
        
        Args:
            x (float/array): Size values.
            A (float): Amplitude (Peak height).
            mu (float): Mean (Center of distribution).
            sigma (float): Standard deviation (Width).
            
        Returns:
            float/array: Probability density values.
        """
        return A * np.exp(-0.5 * ((x - mu) / sigma)**2)

    def fit(self, p0=None):
        """
        Perform a non-linear least squares Gaussian fit on the data.
        
        Args:
            p0 (list, optional): Initial guesses for [A, mu, sigma]. 
                                 Defaults to automatic estimates from data.
                                 
        Returns:
            numpy.ndarray: Optimized parameters [A, mu, sigma].
        """
        if p0 is None:
            # Automatic guess: [Peak height, average size, rough spread]
            p0 = [np.max(self.counts), np.mean(self.sizes), 0.2]
        
        self.params, self.cov = curve_fit(self._gaussian_model, self.sizes, self.counts, p0=p0)
        self.print_accuracy_report()
        return self.params

    def get_fit_accuracy(self):
        """
        Calculates the statistical uncertainty of the fit parameters 
        and the R-squared value.
        """
        if self.cov is None:
            raise ValueError("Fit the model first.")

        # 1. Parameter Errors (1-sigma uncertainty)
        perr = np.sqrt(np.diag(self.cov))
        
        # 2. R-squared calculation
        residuals = self.counts - self._gaussian_model(self.sizes, *self.params)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((self.counts - np.mean(self.counts))**2)
        r_squared = 1 - (ss_res / ss_tot)

        return {
            "mu_error": perr[1],
            "sigma_error": perr[2],
            "r_squared": r_squared
        }

    def print_accuracy_report(self):
        """Prints a detailed report on the fit reliability."""
        acc = self.get_fit_accuracy()
        res = self.results
        
        centerTitle('Fit Accuracy Report')
        print(f"R-squared (Fit quality) : {acc['r_squared']:.4f}")
        print(f"Mean Size Accuracy      : {res['mean']:.3f} ± {acc['mu_error']:.3f} nm")
        print(f"Sigma Accuracy         : {res['sigma']:.3f} ± {acc['sigma_error']:.3f} nm")
        
        if acc['r_squared'] > 0.95:
            print(f"{fg.GREEN}Highly reliable fit.{fg.OFF}")
        elif acc['r_squared'] > 0.85:
            print(f"{fg.ORANGE}Acceptable fit.{fg.OFF}")
        else:
            print(f"{fg.RED}Poor fit. Consider checking for outliers or asymmetry.{fg.OFF}")

    @property
    def results(self):
        """
        Calculate physical parameters from the fitted model.
        
        Returns:
            dict: Dictionary containing amplitude, mean, sigma, FWHM, and CV (%).
            
        Raises:
            ValueError: If called before running the .fit() method.
        """
        if self.model_type == 'lognormal' and self._results_dict:
            return self._results_dict
        
        if self.params is None:
            raise ValueError("Fit has not been performed yet.")
        
        # Handle Gaussian logic (as per your original code)
        A, mu, sigma = self.params
        fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
        cv = (sigma / mu) * 100
        return {
            "amplitude": A,
            "mean": mu,
            "sigma": sigma,
            "fwhm": fwhm,
            "cv_percentage": cv
        }

    def get_relative_height(self, x_value):
        """Calculates height relative to the peak (mu)."""
        mu = self.results['mean']
        sigma = self.results['sigma']
        z = (x_value - mu) / sigma
        return np.exp(-0.5 * z**2)
        
    def print_results(self):
        """Print a formatted summary with dynamically calculated coverage."""
        from scipy.stats import norm, lognorm
        
        centerTitle("Summary of the distribution statistics")
        stats = self.results
        model = getattr(self, 'model_type', 'gaussian')

        # Basic Stats
        print(f"Model Type          : {model.upper()}")
        print(f"Amplitude           : {stats['amplitude']:.0f} particles")
        print(f"Average Size (Mean) : {stats['mean']:.3f} nm")
        print(f"Polydispersity (CV) : {stats['cv_percentage']:.2f}%")

        if model == 'lognormal':
            med, sg = stats['median'], stats['sigma_g']
            s_param = np.log(sg)
            
            # Dynamic calculation function for Lognormal
            def get_prob(low, high):
                return (lognorm.cdf(high, s=s_param, scale=med) - 
                        lognorm.cdf(low, s=s_param, scale=med)) * 100

            # Ranges
            r1 = (med/sg, med*sg)
            r2 = (med/(sg**2), med*(sg**2))
            r3 = (med/(sg**3), med*(sg**3))
            
            print(f"Geometric SD (σg)   : {sg:.3f}")
            print(f"Median Size         : {med:.3f} nm")
            print("-" * 40)
            print(f"Theoretical Population Coverage (Dynamic):")
            print(f"  Med */ 1σg ({r1[0]:>5.2f}-{r1[1]:<5.2f} nm) : {get_prob(*r1):.1f}%")
            print(f"  Med */ 2σg ({r2[0]:>5.2f}-{r2[1]:<5.2f} nm) : {get_prob(*r2):.1f}%")
            print(f"  Med */ 3σg ({r3[0]:>5.2f}-{r3[1]:<5.2f} nm) : {get_prob(*r3):.1f}%")
        
        else:
            mu, sigma = stats['mean'], stats['sigma']
            
            # Dynamic calculation function for Gaussian
            def get_prob(low, high):
                return (norm.cdf(high, mu, sigma) - norm.cdf(low, mu, sigma)) * 100

            print(f"Std Deviation (σ)   : {sigma:.3f} nm")
            print("-" * 40)
            print(f"Theoretical Population Coverage (Dynamic):")
            print(f"  μ ± 1σ     ({mu-sigma:>5.2f}-{mu+sigma:<5.2f} nm) : {get_prob(mu-sigma, mu+sigma):.1f}%")
            print(f"  μ ± 2σ     ({mu-2*sigma:>5.2f}-{mu+2*sigma:<5.2f} nm) : {get_prob(mu-2*sigma, mu+2*sigma):.1f}%")
            print(f"  μ ± 3σ     ({mu-3*sigma:>5.2f}-{mu+3*sigma:<5.2f} nm) : {get_prob(mu-3*sigma, mu+3*sigma):.1f}%")
        
    def get_binned_statistics(self, bin_width_nm=None, total_n=None):
        """
        Calculate theoretical populations using a fixed bin width in nanometers.
        Supports both Gaussian and Log-normal models.
        """
        from scipy.stats import norm, lognorm

        if self.params is None and not hasattr(self, '_results_dict'):
            raise ValueError(f"No parameters available. Fit the model or instantiate from params first.")

        # Extract stats from the unified results property
        stats = self.results
        mu = stats['mean']
        sigma = stats['sigma']
        
        # Set defaults
        if bin_width_nm is None: 
            bin_width_nm = sigma
        if total_n is None:
            # 1. On regarde d'abord si une valeur a été stockée explicitement
            if hasattr(self, 'total_n_expected') and self.total_n_expected is not None:
                total_n = self.total_n_expected
            # 2. Sinon, si c'est du Log-Normal, on prend l'amplitude stockée
            elif getattr(self, 'model_type', 'gaussian') == 'lognormal':
                total_n = self.results.get('amplitude', 1000)
            # 3. En dernier recours, on somme (cas des vraies données expérimentales)
            elif len(self.counts) > 0 and self.cov is not None and np.any(self.cov > 0):
                total_n = np.sum(self.counts)
            else:
                total_n = 1000
            
        # 1. Define coverage limits (±3.5 sigma)
        # For lognormal, we must ensure we don't go below or equal to zero
        limit_min = mu - 3.5 * sigma
        if getattr(self, 'model_type', 'gaussian') == 'lognormal':
            limit_min = max(0.01, limit_min)
        limit_max = mu + 3.5 * sigma
        
        # 2. Generate bin edges starting from mu to ensure symmetry (for Gaussian)
        # or just a consistent range for Log-normal
        right_edges = np.arange(mu + bin_width_nm/2, limit_max + bin_width_nm, bin_width_nm)
        left_edges = np.arange(mu - bin_width_nm/2, limit_min - bin_width_nm, -bin_width_nm)
        edges = np.sort(np.concatenate([left_edges, right_edges]))
        
        # --- 1. First pass: calculate data and sum of ratios for normalization ---
        bins_results = []
        total_ratio_sum = 0
        
        for i in range(len(edges) - 1):
            s1, s2 = edges[i], edges[i+1]
            bin_center = (s1 + s2) / 2
            
            if getattr(self, 'model_type', 'lognormal') == 'lognormal' and hasattr(self, '_lognormal_model'):
                # Lognormal probability and relative height
                s_param = np.log(stats['sigma_g'])
                prob = lognorm.cdf(s2, s=s_param, scale=stats['median']) - \
                       lognorm.cdf(s1, s=s_param, scale=stats['median'])
                peak_val = self._lognormal_model(stats['median'], 1, stats['median'], stats['sigma_g'])
                current_val = self._lognormal_model(bin_center, 1, stats['median'], stats['sigma_g'])
                ratio_to_peak = current_val / peak_val
            else:
                # Gaussian probability and relative height
                prob = norm.cdf(s2, mu, sigma) - norm.cdf(s1, mu, sigma)
                ratio_to_peak = self.get_relative_height(bin_center)
            
            total_ratio_sum += ratio_to_peak
            bins_results.append({
                'range': f"[{s1:>5.2f}, {s2:<5.2f}[",
                'count': prob * total_n,
                'prob': prob,
                'ratio': ratio_to_peak
            })

        # --- 2. Formatting and Output ---
        w_range, w_count, w_prob, w_ratio, w_norm = 18, 10, 10, 12, 12
        
        centerTitle(f'Binned Population (Step={bin_width_nm:.3f} nm, N={total_n})')
        
        header = (f"{'Size Range (nm)':<{w_range}} | {'Count':<{w_count}}| "
                  f"{'Area (%)':<{w_prob}}| {'Ratio/Peak':<{w_ratio}}| {'Norm. (1)':<{w_norm}}")
        print(header)
        print("-" * len(header))
        
        running_count, running_prob, running_norm = 0, 0, 0
        
        for b in bins_results:
            # Normalize so that the sum of the column equals 1.000
            norm_val = b['ratio'] / total_ratio_sum if total_ratio_sum > 0 else 0
            
            running_count += b['count']
            running_prob += b['prob']
            running_norm += norm_val
            
            print(f"{b['range']:<{w_range}} | "
                  f"{int(round(b['count'])):>{w_count-1}} | "
                  f"{b['prob']*100:>{w_prob-2}.1f}% | "
                  f"{b['ratio']:>{w_ratio-2}.3f} | "
                  f"{norm_val:>{w_norm-2}.3f}")
        
        print("-" * len(header))
        print(f"{'Total Covered':<{w_range}} | "
              f"{int(round(running_count)):>{w_count-1}} | "
              f"{running_prob*100:>{w_prob-2}.1f}% | "
              f"{'-':>{w_ratio-2}} | "
              f"{running_norm:>{w_norm-2}.3f}")
        
    def plot(self, title='Nanoparticle Size Distribution', color_histo="skyblue", color_gaussian="red", save_img=None, dpi=300):
        """
        Visualize the experimental histogram overlaid with the Gaussian fit.
        
        Args:
            title (str): Graph title.
            color_histo (str): Hex code or name for the bars.
            color_gaussian (str): Hex code or name for the fit line.
            save_img (str, optional): Filename or path for saving. Supports .png and .svg.
            dpi (int): Resolution for raster exports (default 300).
        """
        res = self.results
        mu = res['mean']
        sigma = res['sigma']
        plt.figure(figsize=(10, 6))
        
        # Calculate bar width based on data spacing
        bar_width = (self.sizes[1] - self.sizes[0]) * 0.9
        
        # Plotting Data and Fit
        plt.bar(self.sizes, self.counts, width=bar_width, color=color_histo, label='Exp. data')
        
        x_smooth = np.linspace(self.sizes.min() * 0.8, self.sizes.max() * 1.2, 500)
        y_smooth = self._gaussian_model(x_smooth, *self.params)
        
        label_text = (f"Gaussian Fit:\n"
                      f"$\mu$ = {res['mean']:.2f} nm\n"
                      f"$\sigma$ = {res['sigma']:.2f} nm\n"
                      f"Polydispersity = {res['cv_percentage']:.1f}%")
        
        plt.plot(x_smooth, y_smooth, color=color_gaussian, lw=2, label=label_text)
        plt.hlines(y=res['amplitude']/2, 
                   xmin=res['mean'] - res['fwhm']/2, 
                   xmax=res['mean'] + res['fwhm']/2, 
                   colors='green', linestyles='--',
                   label=f"FWHM span ({res['fwhm']:.2f} nm)")

        # Polydispersity Visualization (Vertical lines at ±1 sigma)
        # We use axvline for vertical lines
        plt.axvline(x=mu - sigma, color='#3f8188', linestyle=':', lw=1.5, label=f"$\pm 1\sigma$ (Spread)")
        plt.axvline(x=mu + sigma, color='#3f8188', linestyle=':', lw=1.5)
        plt.axvline(x=mu - 2*sigma, color='#3f8188', linestyle=':', lw=1.5, label=f"$\pm 2\sigma$ (Spread)")
        plt.axvline(x=mu + 2*sigma, color='#3f8188', linestyle=':', lw=1.5)

        plt.title(title)
        plt.xlabel('Size (nm)')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        # --- Enhanced Save Logic ---
        if save_img:
            save_path = Path(save_img)
            
            # Ensure the directory exists
            if save_path.parent:
                save_path.parent.mkdir(parents=True, exist_ok=True)
            
            ext = save_path.suffix.lower()
            
            if ext == '.svg':
                plt.savefig(save_path, format='svg')
                print(f"Plot saved as vector graphics: {save_path}")
            elif ext == '.png':
                plt.savefig(save_path, format='png', dpi=dpi)
                print(f"Plot saved as PNG: {save_path}")
            else:
                # Default fallback for other formats supported by matplotlib (pdf, jpg, etc.)
                plt.savefig(save_path)
                print(f"Plot saved with {ext} format: {save_path}")

        plt.show()

# =====================================================================================================
#                           general tools
# =====================================================================================================

import numpy as np
import py3Dmol
from ase import Atoms
from ase.io import write
from ase.neighborlist import NeighborList
from matplotlib.patches import Patch

def get_coordination_numbers(mol: Atoms, cutoff: float = None):
    """
    Calculates the coordination number (CN) for each atom in an ASE Atoms object.
    
    This function determines how many neighbors each atom has based on a distance 
    threshold. If no cutoff is provided, it automatically estimates one based on 
    the 1st percentile of the interatomic distance distribution.

    Args:
        mol (ase.Atoms): The structural model (nanoparticle, molecule, or crystal).
        cutoff (float, optional): The distance threshold (in Angstroms) to define 
            a chemical bond. Defaults to None (automatic detection).

    Returns:
        tuple: A tuple containing:
            - cn (numpy.ndarray): An array of integers representing the CN of each atom.
            - used_cutoff (float): The actual cutoff value used for the calculation.
    """
    nat = len(mol)
    
    if cutoff is None:
        # Automatic cutoff detection: 1.2x the 1st percentile of bond distances
        dist = mol.get_all_distances()
        non_zero_dist = dist[dist > 0]
        if len(non_zero_dist) == 0:
            used_cutoff = 3.0 # Fallback for single-atom systems
        else:
            used_cutoff = np.percentile(non_zero_dist, 1) * 1.2
    else:
        used_cutoff = cutoff

    # Initialize NeighborList with a flat cutoff radius for all atoms
    cutoffs = [used_cutoff / 2.0] * nat
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(mol)

    cn = np.array([len(nl.get_neighbors(i)[0]) for i in range(nat)], dtype=int)
    return cn, used_cutoff

def view_coordination(mol: Atoms, cutoff: float = None, stick_radius: float = 0.1, sphere_scale: float = 0.6, color_map = "YlOrRd"):
    """
    Visualizes a structure using py3Dmol with atoms color-coded by coordination number.
    
    This function computes the coordination environment and generates a 3D ball-and-stick 
    model where colors represent the connectivity (e.g., surface vs. bulk atoms). 
    A Matplotlib legend is displayed alongside the 3D view.

    The color logic is optimized for nanoparticles:
    - CN < 5 : Pastel (low coordination/isolated)
    - CN 5-13: Sequential Gradient (surface to bulk transition)
    - CN > 13: Deep Dark (high density/interstitials)

    Args:
        mol (ase.Atoms): The structural model to visualize.
        cutoff (float, optional): Distance threshold for bond detection. 
            Defaults to None (auto-detect).
        stick_radius (float, optional): The thickness of the bonds in the 3D view. 
            Defaults to 0.1.
        sphere_scale (float, optional): The size multiplier for the atomic spheres. 
            Defaults to 0.6.
        color_map (str, optional): The Matplotlib colormap to use for discrete CN values.
            Defaults to "YlOrRd" (recommended!).

    Returns:
        py3Dmol.view: The interactive viewer object.
    """
    # --- color map for CN 1→20 (just in case...) ---
    def cn_palette():
        from matplotlib import colormaps, colors
        low_map = colormaps.get_cmap("Pastel1")    # Distinct pastels
        mid_map = colormaps.get_cmap(color_map)    # Sequential gradient. YlOrRd is recommended
        high_map = colormaps.get_cmap("Dark2")     # Distinct dark colors
        palette = {}
        for cn in range(1, 21):
            if cn <= 4:
                # Zone 1: Unique Pastel for each (1, 2, 3, 4)
                palette[cn] = colors.to_hex(low_map(cn - 1))
                
            elif 5 <= cn <= 12:
                # Zone 2: Sequential Gradient for surface-to-bulk
                # --- THE HIGH-CONTRAST HACK ---
                # We map specific CNs to hardcoded positions on the YlGnBu scale:
                # 5-6: Bright Yellow/Green (Start of scale)
                # 7-8: Teal/Turquoise (Middle)
                # 9-11: Strong Blue (Upper middle)
                # 12-13: Dark Navy (End of scale)
                
                anchors = {
                    5:  0.00, # Pale Yellow
                    6:  0.15, # Yellow-Green (Vertices)
                    7:  0.30, # Green
                    8:  0.45, # Teal/Cyan (Edges)
                    9:  0.60, # Bright Blue (Facets)
                    10: 0.75, # Royal Blue
                    11: 0.90, # Deep Blue
                    12: 1.00  # Midnight Blue (Bulk)
                }
                # Use .get() to find the anchor, or interpolate if missing
                val = anchors.get(cn, 0.5) 
                palette[cn] = colors.to_hex(mid_map(val))
                
            else:
                # Zone 3: Unique Dark colors for high coordination (> 13)
                # We use cn-14 to restart the index for the high_map
                palette[cn] = colors.to_hex(high_map((cn - 14) % 8))
        
        return palette
    
    def colors_for_cn(cn, palette):
        return [palette[val] for val in cn]

    # 1. Compute CN using the helper function
    cn, used_cutoff = get_coordination_numbers(mol, cutoff=cutoff)
    unique_cns = sorted(np.unique(cn))
    
    # 2. Setup Palette (using tab20 for discrete, distinct categories)
    palette = cn_palette()
    colors = colors_for_cn(cn, palette)

    # 3. py3Dmol Visualization Logic
    buf = io.StringIO()
    write(buf, mol, format="xyz")
    xyz_str = buf.getvalue()
    buf.close()

    v = py3Dmol.view(width=600, height=400)
    v.addModel(xyz_str, "xyz")

    for i, color in enumerate(colors):
        v.setStyle({"serial": i},
                   {"sphere": {"color": color, "scale": sphere_scale},
                    "stick": {"radius": stick_radius, "color": "gray"}})
    
    v.zoomTo()
    v.zoom(0.9)
    
    # 4. Legend rendering using Matplotlib
    legend_elements = [Patch(facecolor=palette[val], edgecolor="k", label=f"CN = {val}")
                       for val in unique_cns]

    fig, ax = plt.subplots(figsize=(3, len(unique_cns) * 0.4))
    ax.axis("off")
    ax.legend(handles=legend_elements, loc="center left", frameon=False, 
              title=f"Coordination (Cutoff: {used_cutoff:.2f}Å)")
    plt.show()
    v.show()