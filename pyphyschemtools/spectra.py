############################################################
#                       Absorption spectra
############################################################
from .visualID_Eng import fg, bg, hl
from .core import centerTitle, centertxt

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc

class SpectrumSimulator:

    """
    Initializes the spectrum simulator

    Args:
        - sigma_ev (float): Gaussian half-width at half-maximum in electron-volts (eV).
                            Default is 0.3 eV (GaussView default is 0.4 eV).
        - plotWH (tuple(int,int)): Width and Height of the matplotlib figures in inches. Default is (12,8).
        - colorS: color of the simulated spectrum (default ='#3e89be')
        - colorVT: color of the vertical transition line (default = '#469cd6')

    Returns:
        None: This method initializes the instance attributes.
    Calculates:
        sigmanm = half-width of the Gaussian band, in nm
        
    """
    
    def __init__(self, sigma_ev=0.3, plotWH=(12,8), \
                 fontSize_axisText=14, fontSize_axisLabels=14, fontSize_legends=12,
                 fontsize_peaks=12,
                 colorS='#3e89be',colorVT='#469cd6'
                ):

        self.sigma_ev = sigma_ev
        # Conversion constante eV -> nm sigma
        self.ev2nm_const = (sc.h * sc.c) * 1e9 / sc.e
        self.sigmanm = self.ev2nm_const / self.sigma_ev
        self.plotW = plotWH[0]
        self.plotH = plotWH[1]
        self.colorS = colorS
        self.colorVT = colorVT
        self.fig = None
        self.graph = None
        self.fontSize_axisText = fontSize_axisText
        self.fontSize_axisLabels = fontSize_axisLabels
        self.fontSize_legends = fontSize_legends
        self.fontsize_peaks = fontsize_peaks

        print(f"sigma = {sigma_ev} eV -> sigmanm = {self.sigmanm:.1f} nm")
    
    def _initializePlot(self):
        fig, graph = plt.subplots(figsize=(self.plotW,self.plotH))
        plt.subplots_adjust(wspace=0)
        plt.xticks(fontsize=self.fontSize_axisText,fontweight='bold')
        plt.yticks(fontsize=self.fontSize_axisText,fontweight='bold')
        return fig, graph
    
    def _calc_epsiG(self,lambdaX,lambdai,fi):
        """
        calculates a Gaussian band shape around a vertical transition
        
        input:
            - lambdaX = wavelength variable, in nm
            - lambdai = vertical excitation wavelength for i_th state, in nm
            - fi = oscillator strength for state i (dimensionless)
            
        output :
            molar absorption coefficient, in L mol-1 cm-1
            
        """
        import scipy.constants as sc
        import numpy as np
        c = sc.c*1e2 #cm-1
        NA = sc.N_A #mol-1
        me = sc.m_e*1000 #g
        e = sc.e*sc.c*10 #elementary charge in esu 
        pf = np.sqrt(np.pi)*e**2*NA/(1000*np.log(10)*c**2*me)
        nubarX = 1e7 / lambdaX # nm to cm-1
        nubari = 1e7 / lambdai
        sigmabar = 1e7 / self.sigmanm
        epsi = pf * (fi / sigmabar) * np.exp(-((nubarX - nubari)/sigmabar)**2)
        return epsi
    
    def _Absorbance(self,eps,opl,cc):
        """
        Calculates the Absorbance with the Beer-Lambert law
        
        input:
            - eps = molar absorption coefficient, in L mol-1 cm-1
            - opl = optical path length, in cm
            - cc = concentration of the attenuating species, in mol.L-1
        
        output :
            Absorbance, A (dimensionless)
            
        """
        return eps*opl*cc
    
    def _sumStatesWithGf(self,wavel,wavelTAB,feTAB):
        import numpy as np
        sumInt = np.zeros(len(wavel))
        for l in wavel:
            for i in range(len(wavelTAB)):
                sumInt[np.argwhere(l==wavel)[0][0]] += self._calc_epsiG(l,wavelTAB[i],feTAB[i])
        return sumInt
    
    def _FindPeaks(self,sumInt,height,prom=1):
        """
        Finds local maxima within the spectrum based on height and prominence.
        
        Prominence is crucial when switching between linear and logarithmic scales:
        - In Linear mode: A large prominence (e.g., 1 to 1000) filters out noise.
        - In Log mode: Data is compressed into a range of ~0 to 5. A large 
          prominence will 'hide' real peaks. A smaller value (0.01 to 0.1) 
          is required to detect shoulders and overlapping bands.

        Input:
            - sumInt: Array of intensities (Epsilon or Absorbance).
            - height: Minimum height a peak must reach to be considered.
            - prom: Required vertical distance between the peak and its lowest contour line.
        
        Returns:
            - PeakIndex: Indices of the detected peaks in the wavelength array.
            - PeakHeight: The intensity values at these peak positions.
            
        """
        from scipy.signal import find_peaks
        peaks = find_peaks(sumInt, height = height, threshold = None, distance = 1, prominence=prom)
        PeakIndex = peaks[0]
        # Check if 'peak_heights' exists in the properties dictionary
        if 'peak_heights' in peaks[1]:
            PeakHeight = peaks[1]['peak_heights']
        else:
            # If height=None, we extract values manually from the input data
            PeakHeight = sumInt[PeakIndex]
        return PeakIndex,PeakHeight

    def _FindShoulders(self, data, tP):
        """
        ###not working###
        Detects shoulders using the second derivative.
        A shoulder appears as a peak in the negative second derivative.
        
        Note on scales:
        - If ylog is True: data should be log10(sumInt) and tP should be log10(tP).
          The second derivative on log data is much more sensitive to subtle 
          inflection points in weak transitions (like n -> pi*).
        - If ylog is False: data is linear sumInt and tP is linear.
        
        Returns:
            - shoulder_idx (ndarray): Array of indices where shoulders were found.
            - shoulder_heights (ndarray): The intensity values at these positions 
              extracted from the input data.
              
        """
        import numpy as np
        # Calculate the second derivative (rate of change of the slope)
        d2 = np.gradient(np.gradient(data))
        
        # We search for peaks in the opposite of the second derivative (-d2).
        # A local maximum in -d2 corresponds to a point of maximum curvature 
        # (inflection), which identifies a shoulder.
        # We use a very low prominence threshold to capture subtle inflections.
        shoulder_idx, _ = self._FindPeaks(-d2, height=None, prom=0.0001)
        shoulder_heights = data[shoulder_idx]
        print(shoulder_idx, shoulder_heights )
        
        return shoulder_idx, shoulder_heights 
    
    def _pickPeak(self,wavel,peaksIndex,peaksH,color,\
                  shift=500,height=500,posAnnotation=200, ylog=False):
        """
        Annotates peaks with a small vertical tick and the wavelength value.
        Adjusts offsets based on whether the plot is in log10 scale or linear.
        In log mode, peaksH must already be log10 values.
        
        """
        
        s=shift
        h=height
        a=posAnnotation
        

        for i in range(len(peaksIndex)):
            x = wavel[peaksIndex[i]]
            y = peaksH[i]
            if ylog:
                # In log scale, we use multipliers to keep the same visual distance
                # 1.1 means "10% above the peak"
                # Adjust these factors based on your preference
                y_s = y * 1.1
                y_h = y * 1.3
                y_a = y * 1.5
                self.graph.vlines(x, y_s, y_h, colors=color, linestyles='solid')
                self.graph.annotate(f"{x:.1f}",xy=(x,y),xytext=(x,y_a),rotation=90,size=self.fontsize_peaks,ha='center',va='bottom', color=color)
            else:
                # Classic linear offsets
                self.graph.vlines(x, y+s, y+s+h, colors=color, linestyles='solid')
                self.graph.annotate(f"{x:.1f}",xy=(x,y),xytext=(x,y+s+h+a),rotation=90,size=self.fontsize_peaks,ha='center',va='bottom',color=color)
        return

    def _setup_axes(self, lambdamin, lambdamax, ymax, ylabel="Absorbance"):
            self.graph.set_xlabel('wavelength / nm', size=self.fontSize_axisLabels, fontweight='bold', color='#2f6b91')
            self.graph.set_ylabel(ylabel, size=self.fontSize_axisLabels, fontweight='bold', color='#2f6b91')
            self.graph.set_xlim(lambdamin, lambdamax)
            self.graph.set_ylim(0, ymax)
            self.graph.tick_params(axis='both', labelsize=self.fontSize_axisText,labelcolor='black')
            for tick in self.graph.xaxis.get_majorticklabels(): tick.set_fontweight('bold') #it is both powerful
                                                # (you can specify the type of a specific tick) and annoying
            for tick in self.graph.yaxis.get_majorticklabels(): tick.set_fontweight('bold')
    
    def plotTDDFTSpectrum(self,wavel,sumInt,wavelTAB,feTAB,tP,ylog,labelSpectrum,colorS='#0000ff',colorT='#0000cf'):
        
        """
        Called by plotEps_lambda_TDDFT. Plots a single simulated UV-Vis spectrum, i.e. after
        gaussian broadening, together with the TDDFT vertical transitions (i.e. plotted as lines)
        
        Args:
            wavel: array of gaussian-broadened wavelengths, in nm
            sumInt: corresponding molar absorptiopn coefficients, in L. mol-1 cm-1
            wavelTAB: wavelength of TDDFT, e.g. discretized, transitions
            ylog: log plot of epsilon
            tP: threshold for finding the peaks
            feTAB: TDDFT oscillator strength for each transition of wavelTAB
            labelSpectrum: title for the spectrum
            
        """

        # # --- DEBUG START ---
        # if ylog:
        #     print(f"\n--- DEBUG LOG MODE ---")
        #     print(f"Max sumInt (linear): {np.max(sumInt):.2f}")
        #     print(f"Max sumInt (log10):  {np.log10(max(np.max(sumInt), 1e-5)):.2f}")
        # # --- DEBUG END ---
        if ylog:
            # Apply safety floor to the entire array
            self.graph.set_yscale('log')
            ymin_val = 1.0 # Epsilon = 1
        else:
            self.graph.set_yscale('linear')
            ymin_val = 0
            
        # vertical lines
        for i in range(len(wavelTAB)):
            val_eps = self._calc_epsiG(wavelTAB[i],wavelTAB[i],feTAB[i])
            self.graph.vlines(x=wavelTAB[i], ymin=ymin_val, ymax=max(val_eps, ymin_val), colors=colorT)

        self.graph.plot(wavel,sumInt,linewidth=3,linestyle='-',color=colorS,label=labelSpectrum)

        self.graph.legend(fontsize=self.fontSize_legends)
        if ylog:
            # Use log-transformed data and log-transformed threshold
            # Clipping tP to 1e-5 ensures we don't take log of 0 or negative
            tPlog = np.log10(max(tP, 1e-5))
            # prom=0.05 allows detection of peaks that are close in log-magnitude
            peaks, peaksH_log = self._FindPeaks(np.log10(np.clip(sumInt, 1e-5, None)), tPlog, prom=0.05)
            peaksH = 10**peaksH_log
            # shoulders, shouldersH_log = self._FindShoulders(np.log10(np.clip(sumInt, 1e-5, None)), tPlog)
            # all_idx = np.concatenate((peaks, shoulders))
            # allH_log = np.concatenate((peaksH_log, shouldersH_log))
            # allH = 10**allH_log
        else:
            peaks, peaksH = self._FindPeaks(sumInt,tP)
            # shoulders, shouldersH = self._FindShoulders(wavel, sumInt, tP)
            # all_idx = np.concatenate((peaks, shoulders))
            # allH = np.concatenate((peaksH, shouldersH))
        self._pickPeak(wavel,peaks,peaksH,colorS,500,500,200,ylog)
        
    
    def plotEps_lambda_TDDFT(self,datFile,lambdamin=200,lambdamax=800,\
                             epsMax=None, titles=None, tP = 10, \
                             ylog=False,\
                             filename=None):
        """
        Plots a TDDFT VUV simulated spectrum (vertical transitions and transitions summed with gaussian functions)
        between lambdamin and lambdamax

        The sum of states is done in the range
        [lambdamin-50, lambdamax+50] nm.
        
        Args:
            datFile: list of pathway/names to "XXX_ExcStab.dat" files generated by 'GParser Gaussian.log -S'
            lambdamin, lambdamax: plot range
            epsMax: y axis graph limit
            titles: list of titles (1 per spectrum plot)
            tP: threshold for finding the peaks (default = 10 L. mol-1 cm-1)
            ylog: y logarithmic axis (default: False).
            save: saves in a png file (300 dpi) if True (default = False)
            filename: saves figure in a 300 dpi png file if not None (default), with filename=full pathway
            
        """
        import matplotlib.ticker as ticker

        if self.fig is not None:
            graph = self.graph
            fig = self.fig
            lambdamin = self.lambdamin
            lambdamax = self.lambdamax
            epsMax = self.epsMax
        else:
            fig, graph = self._initializePlot()

        graph.set_prop_cycle(None)

        if self.fig is None:
            self.fig = fig
            self.graph = graph
            self.lambdamin = lambdamin
            self.lambdamax = lambdamax
            self.epsMax = epsMax
            
            graph.set_xlabel('wavelength / nm',size=self.fontSize_axisLabels,fontweight='bold',color='#2f6b91')

            graph.set_xlim(lambdamin,lambdamax)

            graph.xaxis.set_major_locator(ticker.MultipleLocator(50)) # sets a tick for every integer multiple of the base (here 250) within the view interval
        
        istate,state,wavel,fe,SSq = np.genfromtxt(datFile,skip_header=1,dtype="<U20,<U20,float,float,<U20",unpack=True)
        wavel = np.array(wavel)
        fe = np.array(fe)
        if wavel.size == 1:
            wavel = np.array([wavel])
            fe = np.array([fe])
        wvl = np.arange(lambdamin-50,lambdamax+50,1)
        sumInt = self._sumStatesWithGf(wvl,wavel,fe)
        self.plotTDDFTSpectrum(wvl,sumInt,wavel,fe,tP,ylog,titles,self.colorS,self.colorVT)
        if ylog:
            graph.set_ylabel('log(molar absorption coefficient / L mol$^{-1}$ cm$^{-1})$',size=self.fontSize_axisLabels,fontweight='bold',color='#2f6b91')
            graph.set_ylim(1, epsMax * 5 if epsMax else None)
        else:
            graph.set_yscale('linear')
            graph.set_ylabel('molar absorption coefficient / L mol$^{-1}$ cm$^{-1}$',size=self.fontSize_axisLabels,fontweight='bold',color='#2f6b91')
            graph.set_ylim(0, epsMax if epsMax else np.max(sumInt)*1.18)
        if filename is not None: self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

        peaksI, peaksH = self._FindPeaks(sumInt,tP)
        print(f"{bg.LIGHTREDB}{titles}{bg.OFF}")
        for i in range(len(peaksI)):
            print(f"peak {i:3}. {wvl[peaksI[i]]:4} nm. epsilon_max = {peaksH[i]:.1f} L mol-1 cm-1")
        if ylog:
            print()
            # prom=0.05 allows detection of peaks that are close in log-magnitude
            peaksI, peaksH = self._FindPeaks(np.log10(np.clip(sumInt, 1e-5, None)), np.log10(max(tP, 1e-5)), prom=0.05)
            for i in range(len(peaksI)):
                print(f"peak {i:3}. {wvl[peaksI[i]]:4} nm. log10(epsilon_max) = {peaksH[i]:.1f}")

    def plotAbs_lambda_TDDFT(self, datFiles=None, C0=1e-5, lambdamin=200, lambdamax=800, Amax=2.0,\
                             titles=None, linestyles=[], annotateP=[], tP = 0.1,\
                             resetColors=False,\
                             filename=None):
        """
        Plots a simulated TDDFT VUV absorbance spectrum (transitions summed with gaussian functions)
        between lambdamin and lambdamax (sum of states done in the range [lambdamin-50, lambdamlax+50] nm)
        
        Args:
            datFiles: list of pathway/name to files generated by 'GParser Gaussian.log -S'
            C0: list of concentrations needed to calculate A = epsilon x l x c (in mol.L-1)
            lambdamin, lambdamax: plot range (x axis)
            Amax: y axis graph limit
            titles: list of titles (1 per spectrum plot)
            linestyles: list of line styles(default = "-", i.e. a continuous line)
            annotateP: list of Boolean (annotate lambda max True or False. Default = True)
            tP: threshold for finding the peaks (default = 0.1)
            resetColors (bool): If True, resets the matplotlib color cycle 
                to the first color. This allows different series 
                (e.g., gas phase vs. solvent) to share the same 
                color coding for each molecule across multiple calls. Default: False
            save: saves in a png file (300 dpi) if True (default = False)
            filename: saves figure in a 300 dpi png file if not None (default), with filename=full pathway
            
        """

        if self.fig is None:
            fig, graph = self._initializePlot()
            self.fig = fig
            self.graph = graph
            self.lambdamin = lambdamin
            self.lambdamax = lambdamax
            self.Amax = Amax
        else:
            graph = self.graph
            fig = self.fig
            lambdamin = self.lambdamin
            lambdamax = self.lambdamax
            Amax = self.Amax
            if resetColors: graph.set_prop_cycle(None)
            
        if linestyles == []: linestyles = len(datFiles)*['-']
        if annotateP == []: annotateP = len(datFiles)*[True]

        self._setup_axes(lambdamin, lambdamax, self.Amax, ylabel="Absorbance")
        
        wvl = np.arange(lambdamin-50,lambdamax+50,1)
        for f in range(len(datFiles)):
            istate,state,wavel,fe,SSq = np.genfromtxt(datFiles[f],skip_header=1,dtype="<U20,<U20,float,float,<U20",unpack=True)
            sumInt = self._sumStatesWithGf(wvl,wavel,fe)
            Abs = self._Absorbance(sumInt,1,C0[f])
            plot=self.graph.plot(wvl,Abs,linewidth=3,linestyle=linestyles[f],label=f"{titles[f]}. TDDFT ($C_0$={C0[f]} mol/L)")
            peaksI, peaksH = self._FindPeaks(Abs,tP,0.01)
            if (annotateP[f]): self._pickPeak(wvl,peaksI,peaksH,plot[0].get_color(),0.01,0.04,0.02)
            print(f"{bg.LIGHTREDB}TDDFT. {titles[f]}{bg.OFF}")
            for i in range(len(peaksI)):
                print(f"peak {i:3}. {wvl[peaksI[i]]:4} nm. A = {peaksH[i]:.2f}")
                
        self.graph.legend(fontsize=self.fontSize_legends)

        if filename is not None: self.fig.savefig(filename, dpi=300, bbox_inches='tight')

        return
    
    def plotAbs_lambda_exp(self, csvFiles, C0, lambdamin=200, lambdamax=800,\
                             Amax=2.0, titles=None, linestyles=[], annotateP=[], tP = 0.1,\
                             filename=None):
        """
        Plots an experimental VUV absorbance spectrum read from a csv file between lambdamin and lambdamax
        
        Args:
            - superpose: False = plots a new graph, otherwise the plot is superposed to a previously created one
              (probably with plotAbs_lambda_TDDFT())
            - csvfiles: list of pathway/name to experimental csvFiles (see examples for the format)
            - C0: list of experimental concentrations, i.e. for each sample
            - lambdamin, lambdamax: plot range (x axis)
            - Amax: graph limit (y axis)
            - titles: list of titles (1 per spectrum plot)
            - linestyles: list of line styles(default = "--", i.e. a dashed line)
            - annotateP: list of Boolean (annotate lambda max True or False. Default = True)
            - tP: threshold for finding the peaks (default = 0.1)
            - save: saves in a png file (300 dpi) if True (default = False)
            - filename: saves figure in a 300 dpi png file if not None (default), with filename=full pathway
            
        """
        if linestyles == []: linestyles = len(csvFiles)*['--']
        if annotateP == []: annotateP = len(csvFiles)*[True]

        if self.fig is not None:
            graph = self.graph
            fig = self.fig
            lambdamin = self.lambdamin
            lambdamax = self.lambdamax
            Amax = self.Amax
        else:
            fig, graph = self._initializePlot()
            
        graph.set_prop_cycle(None)
        
        if self.fig is None:
            self.graph = graph
            self.fig = fig
            self.lambdamin = lambdamin
            self.lambdamax = lambdamax
            self.Amax = Amax
            
        self._setup_axes(lambdamin, lambdamax, self.Amax, ylabel="Absorbance")
                
        for f in range(len(csvFiles)):
            wavel,Abs = np.genfromtxt(csvFiles[f],skip_header=1,unpack=True,delimiter=";")
            wavel *= 1e9
            plot=graph.plot(wavel,Abs,linewidth=3,linestyle=linestyles[f],label=f"{titles[f]}. exp ($C_0$={C0[f]} mol/L)")
            peaksI, peaksH = self._FindPeaks(Abs,tP,0.01)
            if (annotateP[f]): self._pickPeak(wavel,peaksI,peaksH,plot[0].get_color(),0.01,0.04,0.02)
            print(f"{bg.LIGHTREDB}exp. {titles[f]}{bg.OFF}")
            for i in range(len(peaksI)):
                print(f"peak {i:3}. {wavel[peaksI[i]]:4} nm. A = {peaksH[i]:.2f}")

        graph.legend(fontsize=self.fontSize_legends)

        if filename is not None: self.fig.savefig(filename, dpi=300, bbox_inches='tight')
    
        return
        
