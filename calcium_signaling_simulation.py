#!/usr/bin/env python3
# Calcium Signaling Simulation - Optimized for Windows local machine

import numpy as np
import random
import math
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import pandas as pd
import sys
import subprocess
import urllib.request
import zipfile
import shutil
from pathlib import Path
import time

# Check and install required packages
required_packages = ['peakutils']
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Installing required package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Now import after ensuring it's installed
import peakutils

from numpy import dot, multiply, diag, power
from scipy.signal import find_peaks
from scipy.signal import peak_widths
from scipy.integrate import simps
from matplotlib.animation import FuncAnimation
from matplotlib import colors
from matplotlib import animation
from base64 import b64encode

# Pouch class for calcium signaling simulation
class Pouch(object):
    def __init__(self, params=None, size = 'xsmall', sim_number=0, save=False, saveName='default'):
        """Class implementing pouch structure and simulating Calcium signaling.
        Inputs:

        params (dict)
            A Python dictionary of parameters to simulate with the keys:
            ['K_PLC', 'K_5', 'k_1' , 'k_a', 'k_p', 'k_2', 'V_SERCA', 'K_SERCA', 'c_tot', 'beta', 'k_i', 'D_p', 'tau_max', 'k_tau', 'lower', 'upper','frac', 'D_c_ratio']

        size (string)
            Size of the pouch to simulate:
            [xsmall, small, medium, or large]

        sim_number (integer)
            Represents ID of a simulation to save the figures with unique names and set the random number generator seed

        save (boolean)
            If True, the simulation outputs will be saved

        saveName (string)
            Additional distinct name to save the output files as
        """
        # Create characteristics of the pouch object
        self.size=size
        self.saveName=saveName
        self.sim_number=sim_number
        self.save=save
        self.param_dict=params

        # If parameters are not set, then use baseline values
        if self.param_dict==None:
            self.param_dict={'K_PLC': 0.2, 'K_5':0.66, 'k_1':1.11 , 'k_a': 0.08, 'k_p':0.13, 'k_2':0.0203, 'V_SERCA':0.9, 'K_SERCA':0.1,
            'c_tot':2, 'beta':.185, 'k_i':0.4, 'D_p':0.005, 'tau_max':800, 'k_tau':1.5, 'lower':0.5, 'upper':0.7, 'frac':0.007680491551459293, 'D_c_ratio':0.1}

        # If a dictionary is given, assure all parameters are provided
        if sorted([r for r in self.param_dict])!=['D_c_ratio','D_p','K_5','K_PLC','K_SERCA','V_SERCA','beta','c_tot','frac','k_1','k_2','k_a', 'k_i','k_p','k_tau','lower','tau_max','upper']:
            print("Improper parameter input, please assure all parameters are specified")
            return

        # Load statics for wing disc geometries
        geometry_dir = Path("./geometry")
        try:
            disc_vertices=np.load(geometry_dir / "disc_vertices.npy", allow_pickle=True).item() # Vertices
            disc_laplacians=np.load(geometry_dir / "disc_sizes_laplacian.npy", allow_pickle=True).item() # Laplacian Matrix
            disc_adjs=np.load(geometry_dir / "disc_sizes_adj.npy", allow_pickle=True).item() # Adjacency matrix
        except FileNotFoundError as e:
            print(f"Error loading geometry files: {e}")
            print("Make sure geometry files are in the correct location.")
            raise

        self.adj_matrix=disc_adjs[self.size] # Adjacency Matrix
        self.laplacian_matrix=disc_laplacians[size] # Laplacian Matrix
        self.new_vertices=disc_vertices[size] # Vertices

        # Establish characteristics of the pouch for simulations
        self.n_cells=self.adj_matrix.shape[0] # Number of cells in the pouch
        self.dt=.2 # Time step for ODE approximations
        self.T=int(3600/self.dt) # Simulation to run for 3600 seconds (1 hour)

        # Establish baseline parameter values for the simulation
        self.K_PLC=self.param_dict['K_PLC']  # .2
        self.K_5=self.param_dict['K_5'] # 0.66
        self.k_1=self.param_dict['k_1'] # 1.11
        self.k_a=self.param_dict['k_a'] # 0.08
        self.k_p=self.param_dict['k_p'] # 0.13
        self.k_2=self.param_dict['k_2'] # 0.0203
        self.V_SERCA=self.param_dict['V_SERCA'] # .9
        self.K_SERCA=self.param_dict['K_SERCA'] # .1
        self.c_tot=self.param_dict['c_tot'] # 2
        self.beta=self.param_dict['beta'] # .185
        self.k_i=self.param_dict['k_i'] # 0.4
        self.D_p =self.param_dict['D_p'] # 0.005
        self.D_c =self.param_dict['D_c_ratio']*self.D_p
        self.tau_max=self.param_dict['tau_max'] # 800
        self.k_tau=self.param_dict['k_tau'] # 1.5
        self.lower=self.param_dict['lower'] # Lower bound of standby cell VPLCs
        self.upper=self.param_dict['upper'] # Upper bound of standy cell VPLCs
        self.frac=self.param_dict['frac']   # Fraction of initiator cells

        self.disc_dynamics=np.zeros((self.n_cells,4,self.T)) # Initialize disc_dynamics to save simulation calcium, IP3, calcium_ER, ratio
        self.VPLC_state=np.zeros((self.n_cells,1)) # Initialize VPLC array for cells

    def simulate(self): # Simulate dynamics of system
            np.random.seed(self.sim_number) # Set the seed for reproducibility (keep initiator cells consistent each run)

            self.disc_dynamics[:,2,0] = (self.c_tot-self.disc_dynamics[:,0,0])/self.beta # Initialize simulation ER Calcium
            self.disc_dynamics[:,3,0]=np.random.uniform(.5,.7,size=(self.n_cells,1)).T # Initialize simulation fraction of inactivated IP3R receptors
            self.VPLC_state=np.random.uniform(self.lower,self.upper,(self.n_cells,1)) # Initialize the values for VPLCs of standby cells to be random uniformly distributed from lower to upper
            stimulated_cell_idxs=np.random.choice(self.n_cells, int(self.frac*self.n_cells)) # Choose which cells are initiator cells
            self.VPLC_state[stimulated_cell_idxs,0]=np.random.uniform(1.3,1.5,len(stimulated_cell_idxs)) # Set the VPLC of initiator cells to be random uniformly distributed between 1.3 and 1.5

            V_PLC=self.VPLC_state.reshape((self.n_cells,1)) # Establish the VPLCs to be passed into the ODE approximations

            # Progress tracking
            print("Starting simulation...")
            total_steps = self.T - 1
            update_interval = max(1, total_steps // 20)  # Update progress every 5%
            last_update = 0
            start_time = time.time()

            # ODE approximation solving
            for step in range(1, self.T):
                # Progress update
                if step % update_interval == 0:
                    progress = step / total_steps * 100
                    elapsed = time.time() - start_time
                    eta = (elapsed / step) * (total_steps - step)
                    print(f"Progress: {progress:.1f}% | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", end="\r")
                    last_update = step

                # ARRAY REFORMATTING
                ca=self.disc_dynamics[:,0,step-1].reshape(-1,1)
                ipt=self.disc_dynamics[:,1,step-1].reshape(-1,1)
                s=self.disc_dynamics[:,2,step-1].reshape(-1,1)
                r=self.disc_dynamics[:,3,step-1].reshape(-1,1)
                ca_laplacian=self.D_c*np.dot(self.laplacian_matrix,ca)
                ipt_laplacian=self.D_p*np.dot(self.laplacian_matrix,ipt)

                # ODE EQUATIONS
                self.disc_dynamics[:,0,step]=(ca+self.dt*(ca_laplacian+(self.k_1*(np.divide(np.divide(r*np.multiply(ca,ipt),(self.k_a+ca)),(self.k_p+ipt)))**3 +self.k_2)*(s-ca)-self.V_SERCA*(ca**2)/(ca**2+self.K_SERCA**2))).T
                self.disc_dynamics[:,1,step]=(ipt+self.dt*(ipt_laplacian+np.multiply(V_PLC,np.divide(ca**2,(ca**2+self.K_PLC**2)))-self.K_5*ipt)).T
                self.disc_dynamics[:,2,step]=((self.c_tot-ca)/self.beta).T
                self.disc_dynamics[:,3,step]=(r+self.dt*((self.k_tau**4+ca**4)/(self.tau_max*self.k_tau**4))*((1-r*(self.k_i+ca)/self.k_i))).T
            
            print(f"\nSimulation completed in {time.time() - start_time:.2f} seconds.")

    def make_animation(self, path=None): # Creation of calcium video
        print("Generating animation (this may take several minutes)...")
        start_time = time.time()
        
        colormap = plt.cm.Greens
        normalize = matplotlib.colors.Normalize(vmin=np.min(self.disc_dynamics[:,0,:]), vmax=max(np.max(self.disc_dynamics[:,0,:]),1))
        with sns.axes_style("white"):
                fig=plt.figure(figsize=(25,15))
                fig.patch.set_alpha(0.)
                ax = fig.add_subplot(1,1,1)
                ax.axis('off')
                sm = plt.cm.ScalarMappable(cmap=colormap, norm=normalize)
                sm._A = []
                cbar=fig.colorbar(sm, ax=ax)
                cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=15,fontweight="bold")
                for cell in self.new_vertices:
                    ax.plot(cell[:,0],cell[:,1], linewidth=0.0, color='w', alpha = 0.0)
                patches = [matplotlib.patches.Polygon(verts) for verts in self.new_vertices ]
                def time_stamp_gen(n):
                    j=0
                    while j < n: # 0.2 sec interval to 1 hour time lapse
                        yield "Elapsed time: "+'{0:02.0f}:{1:02.0f}'.format(*divmod(j*self.dt , 60))
                        j+= 50
                time_stamps=time_stamp_gen(self.T)
                def init():
                    return [ax.add_patch(p) for p in patches]

                def animate(frame,time_stamps):
                    for j in range(len(patches)):
                        c=colors.to_hex(colormap(normalize(frame[j])), keep_alpha=False)
                        patches[j].set_facecolor(c)
                    ax.set_title( next(time_stamps) ,fontsize=50, fontweight="bold")
                    return patches

                # Use original frame count as in the notebook
                frames = self.disc_dynamics[:,0,::50].T  # Keep original step of 50
                print(f"Generating animation with {frames.shape[0]} frames...")
                
                anim = animation.FuncAnimation(fig, animate,
                                               init_func=init,
                                               frames=frames,
                                               fargs=(time_stamps,),
                                               interval=70,
                                               blit=True)
        if self.save:
            if path!=None:
                Path(path).mkdir(parents=True, exist_ok=True)
                output_file = Path(path) / 'demo_video.mp4'
                
                # Use a better codec and bitrate for Windows compatibility
                writer = animation.FFMpegWriter(fps=15, bitrate=1800)
                print(f"Saving animation to {output_file}...")
                anim.save(str(output_file), writer=writer)
                print(f"Animation saved successfully in {time.time() - start_time:.2f} seconds.")
            else:
                print("Please provide a path for saving videos")
                
        plt.close(fig)  # Close the figure to free memory

    def draw_profile(self, path=None): # Draw the VPLC Profile for the simulation
        print("Drawing VPLC profile...")
        colormap = plt.cm.Blues
        normalize = matplotlib.colors.Normalize(vmin=.0, vmax=1.5)
        with sns.axes_style("white"):
                fig=plt.figure(figsize=(18,10))
                ax = fig.add_subplot(1,1,1)
                ax.axis('off')
                fig.patch.set_alpha(0.)
                sm = plt.cm.ScalarMappable(cmap=colormap, norm=normalize)
                sm._A = []
                cbar=fig.colorbar(sm, ax=ax)
                cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=80,fontweight="bold" )
                for cell in self.new_vertices:
                    ax.plot(cell[:,0],cell[:,1], linewidth=1.0, color='k')
                for k in range(len(self.new_vertices)):
                        cell=self.new_vertices[k]
                        c=colors.to_hex(colormap(normalize(self.VPLC_state[k]))[0], keep_alpha=False)
                        ax.fill(cell[:,0],cell[:,1], c)

        if self.save:
            if path!=None:
                Path(path).mkdir(parents=True, exist_ok=True)
                output_file = Path(path) / f"{self.size}Disc_VPLCProfile_{self.sim_number}_{self.saveName}.png"
                fig.savefig(output_file, transparent=True, bbox_inches="tight")
                print(f"VPLC profile saved to {output_file}")
            else:
                print("Please provide a path for saving images")
        plt.close(fig)  # Close the figure to free memory

    def draw_kymograph(self, path=None): # Draw the calcium Kymograph for the simulation
        print("Drawing kymograph...")
        with sns.axes_style("white"):
            centeriods= np.zeros((self.adj_matrix.shape[0],2))
            for j in range(self.adj_matrix.shape[0]):
                x_center, y_center=self.new_vertices[j].mean(axis=0)
                centeriods[j,0],centeriods[j,1]=x_center, y_center
            y_axis=centeriods[:,1]
            kymograp_index=np.where((y_axis<(-490)) & (y_axis>(-510))) # Location of where to draw the kymograph line

            colormap = plt.cm.Greens
            normalize = matplotlib.colors.Normalize(vmin=np.min(self.disc_dynamics[:,0,:]), vmax=max(1,np.max(self.disc_dynamics[:,0,:])))
            fig=plt.figure(figsize=(30,10))
            kymograph=self.disc_dynamics[kymograp_index,0,::][0][:,::2]
            kymograph=np.repeat(kymograph,60,axis=0)

            plt.imshow(kymograph.T,cmap=colormap,norm=normalize)
            ax = plt.gca()
            plt.yticks(np.arange(0,self.T/2,1498) , [0,10,20,30,40,50,60],fontsize=30, fontweight="bold")
            plt.xticks([])
            plt.ylabel('Time (min)',fontsize=30,fontweight='bold')
            if self.size=='xsamll':
                plt.xlabel('Position',fontsize=20,fontweight='bold')
            else:
                plt.xlabel('Position',fontsize=30,fontweight='bold')

            if self.save:
                if path!=None:
                    Path(path).mkdir(parents=True, exist_ok=True)
                    output_file = Path(path) / f"{self.size}Disc_Kymograph_{self.sim_number}_{self.saveName}.png"
                    fig.savefig(output_file, transparent=True, bbox_inches="tight")
                    print(f"Kymograph saved to {output_file}")
                else:
                    print("Please provide a path for saving images")

        plt.close(fig)  # Close the figure to free memory
        del kymograph

    # Pouch method to obtain the signal dynamics of cells above/below a certain threshold
    # Method also stores the cell VPLC value
    # threshold: Used to adjust a baseline threshold that you want to filter cells by
    # whichSignal: Used to determine if we want 'Calcium', 'IP3', 'ER Calcium', or 'IP3R' signaling dynamics
    # wantSignal: Boolean to determine if we want cells above threshold (True), or cells below threshold (False)
    def getSignalCells(self, threshold = 0.06, whichSignal = 'Calcium', wantSignal = True):
        if whichSignal == 'Calcium':
            selector = 0
        elif whichSignal == 'IP3':
            selector = 1
        elif whichSignal == 'ER Calcium':
            selector = 2
        elif whichSignal == 'IP3R':
            selector = 3
        else:
            selector = 0

        VPLCdf = pd.DataFrame(self.VPLC_state) # Obtain the pouch VPLCs
        VPLCdf.rename(columns={VPLCdf.columns[-1]: "VPLC" }, inplace = True) # Rename the dataframe column to "VPLC"
        temp = pd.concat([pd.DataFrame(self.disc_dynamics[:,selector,:]), VPLCdf], axis=1)
        signalsIndex = []
        nonSignalsIndex = []

        checkSignal = temp.iloc[:,0:-1].mean(axis=1)

        for i in range(0, self.n_cells):
            if checkSignal[i] >= threshold:
                signalsIndex.append(i)
            else:
                nonSignalsIndex.append(i)

        signalCells = temp.iloc[signalsIndex]
        nonSignalCells = temp.iloc[nonSignalsIndex]
        if wantSignal:
            return signalCells
        else:
            return nonSignalCells

    # Pouch method to get baseline signal of desired dynamic plus 5 percent
    # whichSignal: 'Calcium', 'IP3', 'ER Calcium', 'IP3R'
    def getBaselineSignal(self, whichSignal = 'Calcium'):
         return 1.05*peakutils.baseline(self.getSignalCells(threshold = 0.00, whichSignal = whichSignal, wantSignal = True).iloc[:,:-1].mean()).mean()

    # Pouch method to plot the frequency width at half-maximum (FWHM) for a desired signal type
    # peakWidthDist: Adjustable parameter to increase/decrease the desired range to search for peaks in the signal
    # typeActivity: String to be passed into the plotting parameters
    # whichSignal: Plot the FWHM for which of the four: 'Calcium', 'IP3', 'ER Calcium', 'IP3R'
    # baselineThreshold: Argument passed to plot the signal of cells that have above the baseline threshold
    # getStats: Boolean to determine if we want to display signal statistics (True), or suppress signal statistics output (False)
    # saveFig: Boolean to determine if we want to save the output figure
    # path: Where do we want to save the output figure?
    def plotFWHM(self, plotLim, peakWidthDist = 100, typeActivity = 'Spike', whichSignal = 'Calcium', baselineThreshold = 'auto', getStats = True, saveFig = False, path = './testResults/cellSizeControl', outputPlot = True):
        if baselineThreshold == 'auto':
            baselineThreshold = self.getBaselineSignal(whichSignal = whichSignal)
        else:
            baselineThreshold = baselineThreshold
        signaling = self.getSignalCells(threshold = baselineThreshold, whichSignal = whichSignal).iloc[:,1:-1] # Signaling Cells. 1:-1 selects time steps but not the VPLC value
        nonSignaling = self.getSignalCells(threshold = baselineThreshold, whichSignal = whichSignal, wantSignal = False).iloc[:,1:-1] # Non-Spiking Cells. 1:-1 selects time steps but not the VPLC value
        a = pd.DataFrame(signaling.iloc[:,:-1].mean().transpose())
        a = a.rename(columns={0: 'signal'})
        b = pd.DataFrame(nonSignaling.iloc[:,:-1].mean().transpose())
        b = b.rename(columns={0: 'nosignal'})
        b['Time (s)'] = b.index.to_frame()*0.2
        concat = pd.concat([a,b], axis=1)

        peakIndexes = list(find_peaks(concat['signal'][0:], distance=peakWidthDist))[0]
        peaksPlot = concat.sort_index().loc[peakIndexes]
        nPeaks = len(peakIndexes)
        minimumIntensity = concat['signal'].min()
        maximumIntensity = concat['signal'].max()
        medPeakIntensity = peaksPlot['signal'].median()

        getPeakWidths = peak_widths(concat['signal'][0:], peakIndexes, rel_height=0.5)
        WHM = [[x*0.2 for x in y] for y in getPeakWidths]
        WHM[1] = [x/0.2 for x in WHM[1]]
        FWHM = WHM[0]
        medianFWHM = pd.DataFrame(FWHM).median()[0]

        period = []
        for i in range(0,nPeaks-1):
            period.append(peaksPlot['Time (s)'].iloc[i+1] - peaksPlot['Time (s)'].iloc[i])
        freq = 1/pd.DataFrame(period).median()[0]
        fullFreq = 1/pd.DataFrame(period)

        integratedInt = simps(concat['signal'], dx=self.dt)

        stdErrMedPeak = 1.96 * 1.2533 * peaksPlot['signal'].std() / np.sqrt(len(peaksPlot['signal']))
        stdErrMedFWHM = 1.96 * 1.2533 * pd.DataFrame(FWHM).std() / np.sqrt(len(pd.DataFrame(FWHM)))
        stdErrFreq = 1.96 * 1.2533 * fullFreq.std() / np.sqrt(len(fullFreq))

        statsDict = {'Min Intensity': minimumIntensity,
                     'Max Intensity': maximumIntensity,
                     'Median Peak Intensity': medPeakIntensity,
                     'Number of Peaks': nPeaks,
                     'Median FWHM': medianFWHM,
                     'Median Frequency': freq,
                     'Integrated Intensity': integratedInt,
                     'Std. Error Median Peak Intensity': stdErrMedPeak,
                     'Std. Error Median FWHM': stdErrMedFWHM[0],
                     'Std. Error Median Frequency': stdErrFreq[0]}

        if outputPlot:
            plt.figure()
            ax = concat.plot(x='Time (s)', legend=None)
            if isinstance(plotLim,list):
                ax.set_ylim(plotLim)
            else:
                ax.set_ylim(auto=True)
            ax.set_ylabel(whichSignal + ' Intensity (a.u.)')
            ax.set_xlabel('Time (s)')
            ax.set_title('Raw Signal ' + typeActivity)
            ax.legend(['Signal Cells', 'Non-signal Cells'], loc=2, bbox_to_anchor=(1.1,0.8))
            ax.plot(peaksPlot['Time (s)'], peaksPlot['signal'], 'o')
            ax.hlines(*WHM[1:], color="C2")
            plt.show()

        if saveFig:
            if path!=None:
                Path(path).mkdir(parents=True, exist_ok=True)
                output_file = Path(path) / f"{typeActivity}_{whichSignal}_{self.n_cells}Cells_FWHM.pdf"
                ax.get_figure().savefig(output_file, transparent=True, bbox_inches="tight")
            else:
                print("Please provide a path for saving images")

        if getStats:
            for x, y in statsDict.items():
                print(x, y)
            return statsDict

    # Pouch method to plot all four of the signaling dynamics on a single plot
    # typeAcitivty: String to be passed into the plot title
    # baselineThreshold: Array of size 4 to filter cell signals above a certain threshold
    # saveFig: Boolean to determine if we would like to save the output figure
    # path: Where do we want to save the output figure?
    # nameAppend: String to append to the save file name
    def plotAllDynamics(self, typeActivity = 'Spike', baselineThreshold = 'auto', saveFig = False, path = './testResults/cellSizeControl', nameAppend = 'allDynamics'):
        def make_patch_spines_invisible(ax):
            ax.set_frame_on(True)
            ax.patch.set_visible(False)
            for sp in ax.spines.values():
                sp.set_visible(False)

        if baselineThreshold == 'auto':
            baselineThreshold = [self.getBaselineSignal(whichSignal = 'Calcium'), self.getBaselineSignal(whichSignal = 'IP3'), 0.0, 0.0]
        else:
            baselineThreshold = baselineThreshold

        signals = ['Calcium', 'IP3', 'ER Calcium', 'IP3R']
        signaling = {key: 0 for key in signals}
        nonSignaling = {key: 0 for key in signals}
        a = {key: 0 for key in signals}
        b = {key: 0 for key in signals}

        for i in range(0, len(signals)):
            signaling[signals[i]] = self.getSignalCells(baselineThreshold[i], signals[i]) # Signaling Cells
            nonSignaling[signals[i]] = self.getSignalCells(baselineThreshold[i], signals[i], wantSignal = False) # Non-Spiking Cells
            a[signals[i]] = pd.DataFrame(signaling[signals[i]].iloc[:,25:-1].mean().transpose()) # Plotting dataframes. 2:-1 selects time steps but not the VPLC value
            a[signals[i]] = a[signals[i]].rename(columns={0: 'signal'+signals[i]})
            b[signals[i]] = pd.DataFrame(nonSignaling[signals[i]].iloc[:,25:-1].mean().transpose()) # Plotting dataframes. 2:-1 selects time steps but not the VPLC value
            b[signals[i]] = b[signals[i]].rename(columns={0: 'nosignal'+signals[i]})

        timedf = b['Calcium'].index.to_frame()*0.2
        concat = pd.concat([timedf,
                           a['Calcium'],
                           a['IP3'],
                           a['ER Calcium'],
                           a['IP3R'],
                           b['Calcium'],
                           b['IP3'],
                           b['ER Calcium'],
                           b['IP3R'],], axis=1)
        concat = concat.rename(columns={0: 'Time (s)'})

        fig, host = plt.subplots()
        fig.subplots_adjust(right=0.75)

        par1 = host.twinx()
        par2 = host.twinx()
        par3 = host.twinx()

        par2.spines["right"].set_position(("axes", 1.2))
        par3.spines["right"].set_position(("axes", 1.4))
        make_patch_spines_invisible(par2)
        par2.spines["right"].set_visible(True)

        p1, = host.plot(concat['Time (s)'], a['Calcium'], "b-", label="Calcium Intensity (a.u.)")
        p2, = par1.plot(concat['Time (s)'], a['IP3'], "r-", label="IP3 Intensity (a.u.)")
        p3, = par2.plot(concat['Time (s)'], a['ER Calcium'], "g-", label="ER Calcium Intensity (a.u.)")
        p4, = par3.plot(concat['Time (s)'], a['IP3R'], "m-", label="Fraction of inactive IP3 Receptors")

        host.set_xlabel("Time (s)")
        host.set_ylabel("Calcium Intensity (a.u.)")
        par1.set_ylabel("IP3 Intensity (a.u.)")
        par2.set_ylabel("ER Calcium Intensity (a.u.)")
        par3.set_ylabel("Fraction of inactive IP3 Receptors")

        host.yaxis.label.set_color(p1.get_color())
        par1.yaxis.label.set_color(p2.get_color())
        par2.yaxis.label.set_color(p3.get_color())
        par3.yaxis.label.set_color(p4.get_color())

        tkw = dict(size=4, width=1.5)
        host.tick_params(axis='y', colors=p1.get_color(), **tkw)
        par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
        par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
        par3.tick_params(axis='y', colors=p4.get_color(), **tkw)
        host.tick_params(axis='x', **tkw)

        lines = [p1, p2, p3, p4]

        host.legend(lines, [l.get_label() for l in lines], loc=2, bbox_to_anchor=(1.6, 0.8))

        plt.show()

        if saveFig:
            if path!=None:
                Path(path).mkdir(parents=True, exist_ok=True)
                output_file = Path(path) / f"{typeActivity}_{self.n_cells}Cells_{nameAppend}.pdf"
                host.get_figure().savefig(output_file, transparent=True, bbox_inches="tight")
            else:
                print("Please provide a path for saving images")

    def signalHeatMap(self, whichSignal = 'Calcium', lowerLim = 'auto', upperLim = 'auto', path=None, saveFig = False, hideFig = True, typeActivity = 'Spike'):
        if whichSignal == 'Calcium':
            selector = 0
        elif whichSignal == 'IP3':
            selector = 1
        elif whichSignal == 'ER Calcium':
            selector = 2
        elif whichSignal == 'IP3R':
            selector = 3
        else:
            selector = 0

        summedDynamics = pd.DataFrame(self.disc_dynamics[:,selector,:]).sum(axis=1)

        if isinstance(lowerLim, str):
            minValue = summedDynamics.min()
        else:
            minValue = lowerLim

        if isinstance(upperLim, str):
            maxValue = summedDynamics.max()
        else:
            maxValue = upperLim

        colormap = plt.cm.inferno
        normalize = matplotlib.colors.Normalize(vmin=minValue, vmax=maxValue)
        with sns.axes_style("white"):
                fig=plt.figure(figsize=(18,10))
                ax = fig.add_subplot(1,1,1)
                ax.axis('off')
                fig.patch.set_alpha(0.)
                sm = plt.cm.ScalarMappable(cmap=colormap, norm=normalize)
                sm._A = []
                cbar=fig.colorbar(sm, ax=ax)
                cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=80,fontweight="bold" )

                for cell in self.new_vertices:
                    ax.plot(cell[:,0],cell[:,1], linewidth=1.0, color='k')

                for k in range(len(self.new_vertices)):
                        cell=self.new_vertices[k]
                        c=colors.to_hex(colormap(normalize(summedDynamics.iloc[k])), keep_alpha=False)
                        ax.fill(cell[:,0],cell[:,1], c)

        if saveFig:
            if path!=None:
                Path(path).mkdir(parents=True, exist_ok=True)
                output_file = Path(path) / f"{whichSignal}_{self.n_cells}Cells_heatMap_{typeActivity}.png"
                plt.savefig(output_file, transparent=True, bbox_inches="tight")
            else:
                print("Please provide a path for saving images")
        if hideFig:
            plt.close(fig)


def download_geometry_files():
    """Download and extract geometry files if they don't exist"""
    geometry_dir = Path("./geometry")
    
    if geometry_dir.exists() and list(geometry_dir.glob("*.npy")):
        print("Geometry files already exist.")
        return True
    
    print("Downloading geometry files...")
    try:
        # Create geometry directory
        geometry_dir.mkdir(exist_ok=True)
        
        # Download the geometry zip file from GitHub
        url = "https://github.com/MulticellularSystemsLab/MSELab_Calcium_Cartography_2021/raw/master/geometry.7z"
        zip_path = "./geometry.7z"
        
        # Try to use urllib to download the file
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete. Extracting files...")
        
        # Check if 7z is available, otherwise try to use a different approach
        try:
            # First try to use 7zip if available
            import py7zr
            with py7zr.SevenZipFile(zip_path, mode='r') as z:
                z.extractall(path="./")
            print("Extraction complete.")
        except ImportError:
            print("py7zr not found. Installing it...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "py7zr"])
            import py7zr
            with py7zr.SevenZipFile(zip_path, mode='r') as z:
                z.extractall(path="./")
            print("Extraction complete.")
        
        # Clean up the zip file
        os.remove(zip_path)
        return True
        
    except Exception as e:
        print(f"Error downloading or extracting geometry files: {e}")
        print("\nPlease manually download the geometry files:")
        print("1. Visit: https://github.com/MulticellularSystemsLab/MSELab_Calcium_Cartography_2021")
        print("2. Download the geometry.7z file")
        print("3. Extract it using 7zip to create a 'geometry' folder")
        print("4. Place this folder in the same directory as this script")
        return False


def setup_matplotlib_for_windows():
    """Configure matplotlib for better Windows performance"""
    # Use Agg backend for better performance on Windows
    matplotlib.use('Agg')
    
    # Reduce figure DPI for better performance
    plt.rcParams['figure.dpi'] = 100
    
    # Use a TkAgg backend for interactive plots if available
    try:
        matplotlib.use('TkAgg', force=True)
        print("Using TkAgg backend for interactive plots")
    except:
        pass


def main():
    # Configure matplotlib for Windows
    setup_matplotlib_for_windows()
    
    # Check for geometry files
    if not download_geometry_files():
        print("Cannot proceed without geometry files.")
        return
    
    # Choose simulation type through user input
    print("\nChoose a simulation type:")
    print("1. Single cell spikes")
    print("2. Intercellular transients")
    print("3. Intercellular waves")
    print("4. Fluttering")
    
    choice = ""
    while choice not in ["1", "2", "3", "4"]:
        choice = input("Enter option (1-4): ")
    
    if choice == "1":
        desired_simulation = "Single cell spikes"
    elif choice == "2":
        desired_simulation = "Intercellular transients"
    elif choice == "3":
        desired_simulation = "Intercellular waves"
    else:
        desired_simulation = "Fluttering"
    
    # Establish simulation parameters based on user input
    if desired_simulation == "Single cell spikes":
        lower_VPLC = 0.1
        upper_VPLC = 0.5
        save_Name = "Spikes"
    elif desired_simulation == "Intercellular transients":
        lower_VPLC = 0.25
        upper_VPLC = 0.6
        save_Name = "ICT"
    elif desired_simulation == "Intercellular waves":
        lower_VPLC = 0.4
        upper_VPLC = 0.8
        save_Name = "ICW"
    elif desired_simulation == "Fluttering":
        lower_VPLC = 1.4
        upper_VPLC = 1.5
        save_Name = "Fluttering"
    else:  # Default to ICW simulation
        lower_VPLC = 0.4
        upper_VPLC = 0.8
        save_Name = "ICW"
    
    print(f"\nRunning {desired_simulation} simulation...")
    
    sim_params = {
        'K_PLC': 0.2,
        'K_5': 0.66,
        'k_1': 1.11,
        'k_a': 0.08,
        'k_p': 0.13,
        'k_2': 0.0203,
        'V_SERCA': 0.9,
        'K_SERCA': 0.1,
        'c_tot': 2,
        'beta': .185,
        'k_i': 0.4,
        'D_p': 0.005,
        'tau_max': 800,
        'k_tau': 1.5,
        'lower': lower_VPLC,
        'upper': upper_VPLC,
        'frac': 0.007680491551459293,
        'D_c_ratio': 0.1
    }

    # Ask user for pouch size
    print("\nChoose pouch size:")
    print("1. X-Small (faster but less detailed)")
    print("2. Small")
    print("3. Medium")
    print("4. Large (slower but more detailed)")
    
    size_choice = ""
    while size_choice not in ["1", "2", "3", "4"]:
        size_choice = input("Enter option (1-4): ")
    
    if size_choice == "1":
        pouch_size = "xsmall"
    elif size_choice == "2":
        pouch_size = "small"
    elif size_choice == "3":
        pouch_size = "medium"
    else:
        pouch_size = "large"
    
    # Create folder for simulation results
    saveFolderName = f'./simulationResults/Pouch_Visualization_{save_Name}'
    Path(saveFolderName).mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {saveFolderName}")

    # Create and run the simulation
    print("\nCreating simulation...")
    p_Simulation = Pouch(params=sim_params, size=pouch_size, sim_number=12345, save=True, saveName=f'Demo_{save_Name}')
    
    # Run simulation
    p_Simulation.simulate()
    
    # Generate output files
    p_Simulation.draw_profile(saveFolderName)
    p_Simulation.draw_kymograph(saveFolderName)
    
    # Ask user if they want to create the animation (as it's time-consuming)
    make_animation = input("\nGenerate animation? This may take 10-15 minutes (y/n): ").lower()
    if make_animation == 'y':
        p_Simulation.make_animation(saveFolderName)
    
    print(f"\nSimulation complete! Results saved to {saveFolderName}")
    
    # Open the output folder
    try:
        # Windows specific - open the folder in Explorer
        os.startfile(os.path.abspath(saveFolderName))
    except:
        print(f"Please check the results in: {os.path.abspath(saveFolderName)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSimulation cancelled by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nPress Enter to exit...")
    input()