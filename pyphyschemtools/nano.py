############################################################
#                    Nano tools
############################################################
from .visualID_Eng import fg, bg, hl
from .core import centerTitle, centertxt

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import os
from pathlib import Path

class NanoparticleDistribution:
    """
    A class to analyze and fit nanoparticle size distributions using Gaussian models.
    
    This tool provides methods for curve fitting, statistical breakdown by size bins, 
    and publication-ready visualization.
    """

    def __init__(self, sizes, counts):
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
        if self.params is None:
            raise ValueError("Fit has not been performed yet. Call .fit() first.")
        
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
        """Print a formatted summary of the distribution statistics."""
        # Access parameters
        centerTitle("Summary of the distribution statistics")

        stats = self.results
        mu, sigma = stats['mean'], stats['sigma']
        
        print(f"Amplitude           : {stats['amplitude']:.0f} particles")
        print(f"Average Size (μ ± σ): {stats['mean']:.2f} ± {stats['sigma']:.2f} nm")
        print(f"Polydispersity      : {stats['cv_percentage']:.2f}%")
        print(f"Total Span (FWHM)   :    {stats['fwhm']:.2f} nm")
        print("-" * 40)
        print(f"Theoretical Population Coverage:")
        print(f"  μ ± 1σ   ({mu-sigma:>5.2f}-{mu+sigma:<5.2f} nm) : 68.3%")
        print(f"  μ ± FWHM ({mu-stats['fwhm']/2:>5.2f}-{mu+stats['fwhm']/2:<5.2f} nm) : 76.1%")
        print(f"  μ ± 2σ   ({mu-2*sigma:>5.2f}-{mu+2*sigma:<5.2f} nm) : 95.4%")
        print(f"  μ ± 3σ   ({mu-3*sigma:>5.2f}-{mu+3*sigma:<5.2f} nm) : 99.7%")
        
    def get_binned_statistics(self, bin_width_nm=None, total_n=None):
        """
        Calculate theoretical populations using a fixed bin width in nanometers.
        The range automatically adjusts to cover approximately ±3.5 sigma.

        Args:
            bin_width_nm (float, optional): The width of each bin in nm. 
                                            Defaults to the fitted sigma.
            total_n (int, optional): Total number of particles for scaling. 
                                     Defaults to the sum of input counts.
        """
        from scipy.stats import norm

        if self.params is None:
            raise ValueError("Fit the model first using .fit()")

        mu, sigma = self.results['mean'], self.results['sigma']
        
        # Set defaults
        if bin_width_nm is None: 
            bin_width_nm = sigma
        if total_n is None: 
            total_n = np.sum(self.counts)
            
        # 1. Define coverage limits (±3.5 sigma)
        limit_min, limit_max = mu - 3.5 * sigma, mu + 3.5 * sigma
        
        # 2. Generate bin edges starting from mu to ensure symmetry
        right_edges = np.arange(mu + bin_width_nm/2, limit_max + bin_width_nm, bin_width_nm)
        left_edges = np.arange(mu - bin_width_nm/2, limit_min - bin_width_nm, -bin_width_nm)
        edges = np.sort(np.concatenate([left_edges, right_edges]))
        
        # 3. Alignment Setup (The fix is here: using nested braces)
        w_range, w_count, w_ratio, w_peak = 20, 10, 12, 12
        
        centerTitle(f'Binned Population (Step={bin_width_nm:.3f} nm, N={total_n})')
        
        header = f"{'Size Range (nm)':<{w_range}} | {'Count':<{w_count}}| {'Area (%)':<{w_ratio}}| {'Ratio/Peak':<{w_peak}}"
        print(header)
        print("-" * len(header))
        
        running_count, running_prob = 0, 0
        
        for i in range(len(edges) - 1):
            s1, s2 = edges[i], edges[i+1]
            z1, z2 = (s1 - mu) / sigma, (s2 - mu) / sigma
            
            prob = norm.cdf(z2) - norm.cdf(z1)
            count = prob * total_n
            
            # Ratio at the center of the bin relative to mu
            bin_center = (s1 + s2) / 2
            ratio_to_peak = self.get_relative_height(bin_center)
            
            running_count += count
            running_prob += prob
            
            s_range = f"{s1:>6.2f} - {s2:<6.2f}"
            
            # CORRECTED ALIGNMENT SYNTAX: {var:<{width}}
            print(f"{s_range:<{w_range}} | {int(round(count)):>{w_count-1}} | {prob*100:>{w_ratio-2}.1f}% | {ratio_to_peak:>{w_peak-2}.3f}")
        
        print("-" * len(header))
        print(f"{'Total Covered':<{w_range}} | {int(round(running_count)):>{w_count-1}} | {running_prob*100:>{w_ratio-2}.1f}% | {'-':>{w_peak-2}}")
        
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