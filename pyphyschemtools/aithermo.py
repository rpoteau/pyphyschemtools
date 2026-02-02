import os
import sys
import glob
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from matplotlib import pyplot as plt

class aiThermo:
    """
    A class to handle thermodynamic surface stability analysis and visualization
    within the tools4pyPhysChem framework.
    """

    def __init__(self, folder_path=None, color_scales=None):
        """
        Initialize the aiThermo object.

        Args:
            folder_path (str or Path): Path to the working directory.
            color_scales (list, optional): List of plotly-compatible color scales.
        """
        self.folder_path = Path(folder_path) if folder_path else None
        self.color_scales = color_scales or [
            [[0, "#dadada"], [1, "#dadada"]], [[0, "#99daaf"], [1, "#99daaf"]],
            [[0, "#f1aeaf"], [1, "#f1aeaf"]], [[0, "#81bbda"], [1, "#81bbda"]],
            [[0, "#da9ac9"], [1, "#da9ac9"]], [[0, "#79dad7"], [1, "#79dad7"]],
            [[0, "#da9f6e"], [1, "#da9f6e"]], [[0, "#b5a8da"], [1, "#b5a8da"]],
            [[0, "#edf1c6"], [1, "#edf1c6"]], [[0, "#c4ffe3"], [1, "#c4ffe3"]],
            [[0, "#61b3ff"], [1, "#61b3ff"]]
        ]
        self.palette = [c[0][1] for c in self.color_scales]
    def _check_folder(self):
        """Internal check to ensure folder_path is set before file operations."""
        if self.folder_path is None:
            raise ValueError(
                "❌ Error: folder_path is not defined for this instance. "
                "Please provide a path when initializing: aiThermo(folder_path='...')"
            )
        if not self.folder_path.exists():
             raise FileNotFoundError(f"❌ Error: The directory {self.folder_path} does not exist.")
            
    def ListOfStableSurfaceCompositions(self, vib):
        """
        Identify and list the relevant thermodynamic data files for the current analysis.

        This method scans the working directory for data files matching specific 
        naming conventions (TPcoverage or TPcoveragevib). It cross-references 
        these with a local configuration file 'ListOfStableSurfaces.dat' to 
        extract surface names and legend labels.

        Args:
            vib (bool): If True, filters for files including vibrational corrections (prefixed with ``vib_``). If False, looks for standard thermodynamic data.

        Returns:
            tuple: A triplet containing:
            - file_paths (list of str): Absolute or relative paths to the .dat files.
            - names (list of str): Internal identifiers for each surface phase.
            - legends (list of str): LaTeX-formatted or plain text labels for graphical legends.

        Notes:
            - The lists are returned in reverse order to ensure correct layering 
              during 3D plotting.
            - Relies on 'ListOfStableSurfaces.dat' existing in the folder_path.
            
        """
        self._check_folder()
        from .core import centertxt
        import glob
        pattern = "TPcoveragevib_*.dat" if vib else "TPcoverage_*.dat"
        file_paths = glob.glob(str(self.folder_path / pattern))
        listOfMinCov = self.folder_path / "ListOfStableSurfaces.dat"
        print(f"List of Stable surfaces is in: {listOfMinCov}")
        # if vib:
        #     file_paths = glob.glob(os.path.join(self.folder_path, "TPcoveragevib_*.dat"))
        # else:
        #     file_paths = glob.glob(os.path.join(self.folder_path, "TPcoverage_*.dat"))
        # print(vib,file_paths)
        # listOfMinCov = os.path.join(self.folder_path, "ListOfStableSurfaces.dat")
        # print("list of Stable surfaces is in: ",listOfMinCov)
        try:
            with open(listOfMinCov, "r") as f:
                lines = [line.rstrip('\n').split() for line in f]
        
            file_paths = []
            names = []
            legends = []
            for l in lines:
                # file_paths = file_paths + glob.glob(os.path.join(self.folder_path, l[0]))
                file_paths = file_paths + glob.glob(str(self.folder_path / l[0]))
                names = names + [l[1]]
                # legends = legends + [l[2]]
                legends.append(fr"{l[2]}") # The 'fr' ensures it is a Raw Formatted string
            names = names[::-1]
            legends = legends[::-1]
            file_paths = file_paths[::-1]
            centertxt(f"List of stable surface compositions. Vibrations = {vib}",size=14,weight="bold") 
            if not vib:
                file_paths = [f.replace('vib_', '_') for f in file_paths]
            for i,f in enumerate(file_paths):
                print(f"{f}  {names[i]}    {legends[i]}")
    
        except FileNotFoundError:
            print(f"ListOfStableSurfaces.dat file has not been found in the {self.folder_path} folder. Exiting...")
            sys.exit()
        return file_paths,names,legends
    
    def plot_surface(self, saveFig=None, vib=True, texLegend=False, xLegend=0.5, yLegend=0.4):
        """
        Generate an interactive 3D thermodynamic stability map using Plotly.

        This method visualizes multiple Gibbs free energy surfaces as a function of 
        Temperature (X) and Pressure (Y). It automatically handles log-scale 
        transformations for the pressure axis and projects reference experimental 
        conditions and phase boundaries onto the plot.

        Args:
            saveFig (str, optional): The filename (without extension) to export 
                the resulting plot as a PNG image. Defaults to None (no save).
            vib (bool): Whether to use vibration-corrected data. Defaults to True.
            texLegend (bool): If True, uses LaTeX legends extracted from the 
                configuration file. Defaults to False.
            xLegend (float): Horizontal position of the legend box (0 to 1). 
                Defaults to 0.5.
            yLegend (float): Vertical position of the legend box (0 to 1). 
                Defaults to 0.4.

        Returns:
            plotly.graph_objects.FigureWidget: An interactive widget containing 
                the 3D surfaces, experimental markers, and reference lines.

        Workflow:
            1. Scans data files and parses Temperature/Pressure/Energy grids.
            2. Traces individual 3D surfaces with mapped color scales.
            3. Calculates and plots intersection boundaries between surface phases.
            4. Overlays experimental markers (e.g., specific T/P conditions).
            5. Optionally exports and crops the resulting image using Pillow.
        """
        self._check_folder()
        import os
        # Define tick values explicitly for log scale (powers of 10)
        color_scales = self.color_scales
        logmin = -20
        logmax = 5
        log_tick_vals = np.logspace(logmin, logmax, num=1+(logmax-logmin)//5)  # Example range from 10^20 to 10^5
        log_tick_labels = [f"10<sup>{int(np.log10(tick))}</sup>" for tick in log_tick_vals]  # Format labels as 10^n
        import plotly.graph_objects as go
        
        stableSurfaces, nameOfStableSurfaces, legendOfStableSurfaces = self.ListOfStableSurfaceCompositions(vib)

        # FIX: Track all Z values to find a true global minimum for the floor
        all_z_mins = []
        
        fig = go.Figure()
        
        for i, file_path in enumerate(stableSurfaces):
            with open(file_path, "r") as f:
                lines = f.readlines()
        
            series = []
            temp = []
            for line in lines:
                if line.strip():  
                    temp.append(list(map(float, line.split())))  
                else:
                    if temp:
                        series.append(np.array(temp)) 
                        temp = []
            if temp: 
                series.append(np.array(temp))
        
            data = np.array(series)
        
            X = data[:, :, 0]
            Y = data[:, :, 1]
            Z = data[:, :, 2]
            all_z_mins.append(np.min(Z))
        
            fig.add_trace(go.Surface(
                x=X, 
                y=Y,
                z=Z, 
                colorscale=color_scales[i % len(color_scales)],  
                showscale=False,
                name = nameOfStableSurfaces[i]))
        
            if legendOfStableSurfaces[i] != "None" and texLegend:
                name=f"{legendOfStableSurfaces[i]}"
            else:
                name=f"{nameOfStableSurfaces[i]}"
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],  # Invisible point
                mode="markers",
                name=f"{name}",
                marker=dict(color=color_scales[i % len(color_scales)][-1][1], size=10),
                showlegend=True))
    
        # FIX: Calculate zmin globally
        zmin = np.min(all_z_mins) - 50

        fig.add_trace(go.Scatter3d(
            x=[55+273.15,90+273.15], y=[np.log10(2),np.log10(4)], z=[zmin-10,zmin-10],  # Invisible point
            mode="markers",
            marker=dict(color='red', size=10, symbol='cross'),
            name='exp. Conditions (55°C, 2 bar & 90°C, 4 bar)',
            showlegend=True))
        
        fig.add_trace(go.Scatter3d(
            x=[0, 1000], y=[np.log10(1),np.log10(1)], z=[zmin+600]*2,
            mode="lines",
            line=dict(color="blue", width=3),
            name="1 bar",
            showlegend=False
        ))        
        
        fig.add_trace(go.Scatter3d(
            x=[298, 298], y=[-20,5], z=[zmin+600]*2,
            mode="lines",
            line=dict(color="black", width=3),
            name="298 K",
            showlegend=False
        ))        
            
                
        fig.update_layout(
            width=1200,  # Increase figure width (default is ~700)
            height=1200,
            paper_bgcolor='rgba(0,0,0,0)',  # White background outside the plot
            plot_bgcolor='rgba(0,0,0,0)',   # White background inside the 3D plot
    
            margin = dict(l=0,r=0,t=0,b=0),

            scene=dict(
                aspectmode="manual",  # Allows custom aspect ratio
                aspectratio=dict(x=1.15, y=1.15, z=1),  # Adjust scaling
                xaxis=dict(
                    title=dict(
                        text="Temperature / K",
                        font=dict(size=16, family="Arial", color="blue", weight='bold'),
                    ),
                    autorange="reversed", # This inverts the x-axis direction
                    showgrid=True,
                    zeroline=True,
                    tickfont=dict(color="black", size=15,weight="bold"),
                    tickangle=0,
                    ticklen=10,
                    tickwidth=2,
                    ticks="outside",
                    showbackground=False,  # Enable background to create a frame
                    backgroundcolor="grey"  # Black frame
                ),
                yaxis=dict(
                    title=dict(
                        text="Pressure / bar",
                        font=dict(size=16, family="Arial", color="blue", weight='bold'),
                    ),
                    tickangle=0,  # Rotate Y-axis ticks
                    showgrid=True,
                    zeroline=True,
                    type='log',
                    tickvals=log_tick_vals.tolist(),  # Set tick positions
                    ticktext=log_tick_labels,  # Display ticks as 10^(-n)
                    tickfont=dict(color="black", size=15,weight="bold"),
                    ticklen=10,
                    tickwidth=2,
                    ticks="outside",
                    showbackground=False,  # Enable background to create a frame
                    backgroundcolor="grey"  # Black frame
                ),
                zaxis=dict(
                    title="",  
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    showbackground=False,  # Enable background to create a frame
                    backgroundcolor="grey"  # Black frame
                ),
                camera=dict(
                    eye=dict(x=1e-5, y=-1e-2, z=-1000),
                    # eye=dict(x=1e-5, y=-1e-2, z=-1000),
                    up=dict(x=0, y=1, z=0),
                    projection=dict(type="orthographic")
                ),
            ),
            legend=dict(
                # y=0,
                # x=0.2,
                x = xLegend, y = yLegend,
                font=dict(size=13, color="black"),
                bgcolor="rgba(255, 255, 255, 1)",  # Light transparent background
                bordercolor="grey",
                borderwidth=1,
                itemsizing='constant'
            ),
            showlegend=True
        )

        if saveFig is not None:
            from .core import crop_images
            # pngFile = os.path.join(folder_path, savedFig+".png")
            # import plotly.io as pio
            # fig.write_image(pngFile, format="png", width=1200, height=1200, scale=3)
            pngFile = self.folder_path / f"{saveFig}.png"
            fig.write_image(pngFile, format="png", width=1200, height=1200, scale=3)
            # Automatic crop after saving
            crop_images(pngFile)
            
        fig_widget = go.FigureWidget(fig)
        fig_widget.show()
        return fig_widget

    def plot_palette(self, angle=0, save_png=None):
        """
        Visualize the 1D color palette used for surface identification.

        This method generates a horizontal bar of colors corresponding to the 
        different surface phases defined in the instance. Each color is labeled 
        with its numerical index, allowing for quick cross-referencing between 
        the palette and the 3D surface plot.

        Args:
            angle (int, optional): Rotation angle of the x-axis tick labels (indices). 
                Defaults to 0.
            save_png (str, optional): Filename (including .png extension) to save 
                the palette image to the working directory. Defaults to None 
                (display only).

        Returns:
            None: Displays the plot using matplotlib.pyplot.show().

        Notes:
            - Requires 'seaborn' for the palplot generation.
            - If 'save_png' is provided, the image is saved with a resolution of 
              300 DPI and a transparent background.
        """
        names = [str(i) for i in range(len(self.palette))]
        sns.palplot(sns.color_palette(self.palette))
        ax = plt.gca()
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, weight='bold', size=10, rotation=angle)
        
        if save_png:
            plt.tight_layout()
            plt.savefig(self.folder_path / save_png, dpi=300, transparent=True)
        plt.show()

