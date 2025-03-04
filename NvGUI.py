#!/usr/bin/env python3
# NVIDIA GPU-Accelerated Calcium Signaling Simulation - GUI version

import os
import sys
import subprocess
import urllib.request
import traceback
import random
from pathlib import Path
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import queue
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import colors
from matplotlib import animation

# Check and install required packages
required_packages = ['numpy', 'pandas', 'matplotlib', 'seaborn', 'peakutils', 'py7zr']
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Installing required package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Try to import CuPy for GPU acceleration
HAS_GPU = False
try:
    # First attempt to import
    try:
        import cupy as cp
        
        # Now verify CUDA runtime is working by creating a test array
        try:
            test_array = cp.array([1, 2, 3], dtype=cp.float32)
            test_result = cp.sum(test_array)  # Try a simple operation
            HAS_GPU = True
            print("GPU acceleration enabled with CuPy")
        except RuntimeError as e:
            print(f"WARNING: CuPy loaded but CUDA runtime error occurred: {e}")
            print("This is likely due to missing NVIDIA CUDA libraries or incompatible versions.")
            print("Make sure CUDA toolkit is installed and compatible with your CuPy version.")
            print("Falling back to CPU with NumPy")
            cp = np
    except ImportError:
        print("CuPy not found. Attempting to install...")
        try:
            # Try to determine the correct CuPy package based on system
            if sys.platform == 'win32':
                subprocess.check_call([sys.executable, "-m", "pip", "install", "cupy"])
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "cupy-cuda11x"])
            
            # Try importing again after installation
            import cupy as cp
            # Test if it works
            test_array = cp.array([1, 2, 3], dtype=cp.float32)
            test_result = cp.sum(test_array)
            HAS_GPU = True
            print("Successfully installed and enabled GPU acceleration with CuPy")
        except Exception as e:
            print(f"Failed to install or use CuPy: {e}")
            print("Falling back to CPU with NumPy")
            cp = np
except Exception as e:
    print(f"WARNING: Error setting up GPU acceleration: {e}")
    print("Falling back to CPU with NumPy")
    cp = np

# Now import other required packages
import peakutils

# Import the original Pouch class and utilities from the original script
def import_simulation_module():
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from calcium_signaling_simulation import Pouch, download_geometry_files, setup_matplotlib_for_windows
    return Pouch, download_geometry_files, setup_matplotlib_for_windows

class RedirectText:
    """Redirects console output to tkinter Text widget"""
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.queue = queue.Queue()
        self.updating = True
        threading.Thread(target=self._update_text_widget, daemon=True).start()
    
    def write(self, string):
        self.queue.put(string)
    
    def _update_text_widget(self):
        while self.updating:
            try:
                while True:  # Process all available messages
                    string = self.queue.get_nowait()
                    self.text_widget.configure(state="normal")
                    self.text_widget.insert("end", string)
                    self.text_widget.see("end")
                    self.text_widget.configure(state="disabled")
                    self.queue.task_done()
            except queue.Empty:
                time.sleep(0.1)  # Wait a bit for new messages
    
    def flush(self):
        pass
    
    def stop(self):
        self.updating = False

class GpuPouch:
    """GPU-accelerated version of the Pouch class for calcium signaling simulation"""
    
    def __init__(self, params=None, size='xsmall', sim_number=0, save=False, saveName='default'):
        """Class implementing GPU-accelerated pouch structure and simulating Calcium signaling.
        
        This class mirrors the original Pouch class but uses CuPy for GPU acceleration 
        and uses float32 precision for better performance.
        """
        # Create characteristics of the pouch object
        self.size = size
        self.saveName = saveName
        self.sim_number = sim_number
        self.save = save
        self.param_dict = params
        
        # If parameters are not set, then use baseline values
        if self.param_dict is None:
            self.param_dict = {
                'K_PLC': 0.2, 'K_5': 0.66, 'k_1': 1.11, 'k_a': 0.08, 'k_p': 0.13, 'k_2': 0.0203, 
                'V_SERCA': 0.9, 'K_SERCA': 0.1, 'c_tot': 2, 'beta': .185, 'k_i': 0.4, 'D_p': 0.005, 
                'tau_max': 800, 'k_tau': 1.5, 'lower': 0.5, 'upper': 0.7, 
                'frac': 0.007680491551459293, 'D_c_ratio': 0.1
            }
        
        # If a dictionary is given, assure all parameters are provided
        expected_params = [
            'D_c_ratio', 'D_p', 'K_5', 'K_PLC', 'K_SERCA', 'V_SERCA', 'beta', 'c_tot', 'frac',
            'k_1', 'k_2', 'k_a', 'k_i', 'k_p', 'k_tau', 'lower', 'tau_max', 'upper'
        ]
        if sorted([r for r in self.param_dict]) != sorted(expected_params):
            print("Improper parameter input, please assure all parameters are specified")
            return
        
        # Load geometry files using numpy first
        geometry_dir = Path("./geometry")
        try:
            disc_vertices = np.load(geometry_dir / "disc_vertices.npy", allow_pickle=True).item()  # Vertices
            disc_laplacians = np.load(geometry_dir / "disc_sizes_laplacian.npy", allow_pickle=True).item()  # Laplacian Matrix
            disc_adjs = np.load(geometry_dir / "disc_sizes_adj.npy", allow_pickle=True).item()  # Adjacency matrix
        except FileNotFoundError as e:
            print(f"Error loading geometry files: {e}")
            print("Make sure geometry files are in the correct location.")
            raise
        
        # Convert numpy arrays to CuPy arrays with float32 precision
        self.adj_matrix = cp.array(disc_adjs[self.size], dtype=cp.float32)  # Adjacency Matrix
        self.laplacian_matrix = cp.array(disc_laplacians[size], dtype=cp.float32)  # Laplacian Matrix
        self.new_vertices = disc_vertices[size]  # Keep vertices on CPU for visualization
        
        # Establish characteristics of the pouch for simulations
        self.n_cells = self.adj_matrix.shape[0]  # Number of cells in the pouch
        self.dt = 0.2  # Time step for ODE approximations
        self.T = int(3600 / self.dt)  # Simulation to run for 3600 seconds (1 hour)
        
        # Convert all parameters to float32 for better GPU performance
        for key, value in self.param_dict.items():
            self.param_dict[key] = float(value)
        
        # Establish baseline parameter values for the simulation (all as float32)
        self.K_PLC = cp.float32(self.param_dict['K_PLC'])
        self.K_5 = cp.float32(self.param_dict['K_5'])
        self.k_1 = cp.float32(self.param_dict['k_1'])
        self.k_a = cp.float32(self.param_dict['k_a'])
        self.k_p = cp.float32(self.param_dict['k_p'])
        self.k_2 = cp.float32(self.param_dict['k_2'])
        self.V_SERCA = cp.float32(self.param_dict['V_SERCA'])
        self.K_SERCA = cp.float32(self.param_dict['K_SERCA'])
        self.c_tot = cp.float32(self.param_dict['c_tot'])
        self.beta = cp.float32(self.param_dict['beta'])
        self.k_i = cp.float32(self.param_dict['k_i'])
        self.D_p = cp.float32(self.param_dict['D_p'])
        self.D_c = cp.float32(self.param_dict['D_c_ratio']) * cp.float32(self.param_dict['D_p'])
        self.tau_max = cp.float32(self.param_dict['tau_max'])
        self.k_tau = cp.float32(self.param_dict['k_tau'])
        self.lower = cp.float32(self.param_dict['lower'])
        self.upper = cp.float32(self.param_dict['upper'])
        self.frac = cp.float32(self.param_dict['frac'])
        
        # Initialize disc_dynamics with float32 precision on GPU
        self.disc_dynamics = cp.zeros((self.n_cells, 4, self.T), dtype=cp.float32)
        self.VPLC_state = cp.zeros((self.n_cells, 1), dtype=cp.float32)
    
    def simulate(self):
        """GPU-accelerated simulation of calcium dynamics"""
        # Set random seed for reproducibility
        cp.random.seed(self.sim_number)
        if HAS_GPU:
            np.random.seed(self.sim_number)  # Also set NumPy seed for CPU operations
        
        # Initialize simulation variables
        self.disc_dynamics[:, 2, 0] = (self.c_tot - self.disc_dynamics[:, 0, 0]) / self.beta
        
        # Initialize using random values (keeping these operations on CPU with numpy if using CuPy)
        if HAS_GPU:
            # Generate random values on CPU, then transfer to GPU
            random_values = np.random.uniform(0.5, 0.7, size=(self.n_cells, 1))
            self.disc_dynamics[:, 3, 0] = cp.array(random_values.T, dtype=cp.float32)
            
            # Initialize VPLC_state with random values
            random_vplc = np.random.uniform(float(self.lower), float(self.upper), (self.n_cells, 1))
            self.VPLC_state = cp.array(random_vplc, dtype=cp.float32)
            
            # Select random initiator cells
            n_initiators = int(float(self.frac) * self.n_cells)
            stimulated_cell_idxs = np.random.choice(self.n_cells, n_initiators)
            
            # Set random VPLC values for initiator cells
            random_initiator_vplc = np.random.uniform(1.3, 1.5, len(stimulated_cell_idxs))
            
            # Update values on GPU
            for i, idx in enumerate(stimulated_cell_idxs):
                self.VPLC_state[idx, 0] = cp.float32(random_initiator_vplc[i])
        else:
            # If no GPU, use numpy directly
            self.disc_dynamics[:, 3, 0] = cp.random.uniform(0.5, 0.7, size=(self.n_cells, 1)).T
            self.VPLC_state = cp.random.uniform(self.lower, self.upper, (self.n_cells, 1))
            stimulated_cell_idxs = cp.random.choice(self.n_cells, int(self.frac * self.n_cells))
            self.VPLC_state[stimulated_cell_idxs, 0] = cp.random.uniform(1.3, 1.5, len(stimulated_cell_idxs))
        
        # Reshape VPLC_state for ODE calculation
        V_PLC = self.VPLC_state.reshape((self.n_cells, 1))
        
        # Simulation progress tracking
        print("Starting simulation...")
        total_steps = self.T - 1
        update_interval = max(1, total_steps // 20)  # Update progress every 5%
        start_time = time.time()
        
        # ODE approximation solving - main computation loop
        for step in range(1, self.T):
            # Progress update
            if step % update_interval == 0:
                progress = step / total_steps * 100
                elapsed = time.time() - start_time
                eta = (elapsed / step) * (total_steps - step)
                print(f"Progress: {progress:.1f}% | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s", end="\r")
            
            # ARRAY REFORMATTING - keep everything on GPU
            ca = self.disc_dynamics[:, 0, step-1].reshape(-1, 1)
            ipt = self.disc_dynamics[:, 1, step-1].reshape(-1, 1)
            s = self.disc_dynamics[:, 2, step-1].reshape(-1, 1)
            r = self.disc_dynamics[:, 3, step-1].reshape(-1, 1)
            
            # Diffusion terms - key area for GPU acceleration
            ca_laplacian = self.D_c * cp.dot(self.laplacian_matrix, ca)
            ipt_laplacian = self.D_p * cp.dot(self.laplacian_matrix, ipt)
            
            # Core ODE EQUATIONS - optimized for GPU with element-wise operations
            # Calcium dynamics
            calcium_release = (self.k_1 * ((r * ca * ipt) / ((self.k_a + ca) * (self.k_p + ipt)))**3 + self.k_2) * (s - ca)
            calcium_uptake = self.V_SERCA * (ca**2) / (ca**2 + self.K_SERCA**2)
            self.disc_dynamics[:, 0, step] = (ca + self.dt * (ca_laplacian + calcium_release - calcium_uptake)).T
            
            # IP3 dynamics
            ip3_production = cp.multiply(V_PLC, ca**2 / (ca**2 + self.K_PLC**2))
            ip3_degradation = self.K_5 * ipt
            self.disc_dynamics[:, 1, step] = (ipt + self.dt * (ipt_laplacian + ip3_production - ip3_degradation)).T
            
            # ER calcium dynamics
            self.disc_dynamics[:, 2, step] = ((self.c_tot - ca) / self.beta).T
            
            # IP3R inactivation dynamics
            ip3r_term1 = (self.k_tau**4 + ca**4) / (self.tau_max * self.k_tau**4)
            ip3r_term2 = (1 - r * (self.k_i + ca) / self.k_i)
            self.disc_dynamics[:, 3, step] = (r + self.dt * ip3r_term1 * ip3r_term2).T
        
        print(f"\nSimulation completed in {time.time() - start_time:.2f} seconds.")
    
    def make_animation(self, path=None):
        """Creation of calcium video - Similar to original but with CPU transfer"""
        print("Generating animation (this may take several minutes)...")
        start_time = time.time()
        
        # Transfer data from GPU to CPU for visualization
        if HAS_GPU:
            cpu_dynamics = self.disc_dynamics.get()
        else:
            cpu_dynamics = self.disc_dynamics
        
        colormap = plt.cm.Greens
        normalize = matplotlib.colors.Normalize(
            vmin=np.min(cpu_dynamics[:, 0, :]), 
            vmax=max(np.max(cpu_dynamics[:, 0, :]), 1)
        )
        
        with sns.axes_style("white"):
            fig = plt.figure(figsize=(25, 15))
            fig.patch.set_alpha(0.)
            ax = fig.add_subplot(1, 1, 1)
            ax.axis('off')
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=normalize)
            sm._A = []
            cbar = fig.colorbar(sm, ax=ax)
            cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=15, fontweight="bold")
            
            for cell in self.new_vertices:
                ax.plot(cell[:, 0], cell[:, 1], linewidth=0.0, color='w', alpha=0.0)
            
            patches = [matplotlib.patches.Polygon(verts) for verts in self.new_vertices]
            
            def time_stamp_gen(n):
                j = 0
                while j < n:  # 0.2 sec interval to 1 hour time lapse
                    yield "Elapsed time: " + '{0:02.0f}:{1:02.0f}'.format(*divmod(j*self.dt, 60))
                    j += 50
            
            time_stamps = time_stamp_gen(self.T)
            
            def init():
                return [ax.add_patch(p) for p in patches]
            
            def animate(frame, time_stamps):
                for j in range(len(patches)):
                    c = colors.to_hex(colormap(normalize(frame[j])), keep_alpha=False)
                    patches[j].set_facecolor(c)
                ax.set_title(next(time_stamps), fontsize=50, fontweight="bold")
                return patches
            
            # Use original frame count as in the notebook
            frames = cpu_dynamics[:, 0, ::50].T  # Keep original step of 50
            print(f"Generating animation with {frames.shape[0]} frames...")
            
            anim = animation.FuncAnimation(
                fig, animate, init_func=init, frames=frames,
                fargs=(time_stamps,), interval=70, blit=True
            )
            
        if self.save:
            if path is not None:
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
    
    def draw_profile(self, path=None):
        """Draw the VPLC Profile for the simulation - Similar to original but with CPU transfer"""
        print("Drawing VPLC profile...")
        
        # Transfer data from GPU to CPU for visualization
        if HAS_GPU:
            cpu_vplc = self.VPLC_state.get()
        else:
            cpu_vplc = self.VPLC_state
        
        colormap = plt.cm.Blues
        normalize = matplotlib.colors.Normalize(vmin=0.0, vmax=1.5)
        
        with sns.axes_style("white"):
            fig = plt.figure(figsize=(18, 10))
            ax = fig.add_subplot(1, 1, 1)
            ax.axis('off')
            fig.patch.set_alpha(0.)
            
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=normalize)
            sm._A = []
            cbar = fig.colorbar(sm, ax=ax)
            cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=80, fontweight="bold")
            
            for cell in self.new_vertices:
                ax.plot(cell[:, 0], cell[:, 1], linewidth=1.0, color='k')
            
            for k in range(len(self.new_vertices)):
                cell = self.new_vertices[k]
                c = colors.to_hex(colormap(normalize(cpu_vplc[k]))[0], keep_alpha=False)
                ax.fill(cell[:, 0], cell[:, 1], c)
        
        if self.save:
            if path is not None:
                Path(path).mkdir(parents=True, exist_ok=True)
                output_file = Path(path) / f"{self.size}Disc_VPLCProfile_{self.sim_number}_{self.saveName}.png"
                fig.savefig(output_file, transparent=True, bbox_inches="tight")
                print(f"VPLC profile saved to {output_file}")
            else:
                print("Please provide a path for saving images")
        
        plt.close(fig)  # Close the figure to free memory
    
    def draw_kymograph(self, path=None):
        """Draw the calcium Kymograph for the simulation - Similar to original but with CPU transfer"""
        print("Drawing kymograph...")
        
        # Transfer data from GPU to CPU for visualization and computation
        if HAS_GPU:
            cpu_dynamics = self.disc_dynamics.get()
        else:
            cpu_dynamics = self.disc_dynamics
        
        with sns.axes_style("white"):
            centeriods = np.zeros((self.adj_matrix.shape[0], 2))
            for j in range(self.adj_matrix.shape[0]):
                x_center, y_center = self.new_vertices[j].mean(axis=0)
                centeriods[j, 0], centeriods[j, 1] = x_center, y_center
            
            y_axis = centeriods[:, 1]
            kymograp_index = np.where((y_axis < (-490)) & (y_axis > (-510)))  # Location of where to draw the kymograph line
            
            colormap = plt.cm.Greens
            normalize = matplotlib.colors.Normalize(
                vmin=np.min(cpu_dynamics[:, 0, :]), 
                vmax=max(1, np.max(cpu_dynamics[:, 0, :]))
            )
            
            fig = plt.figure(figsize=(30, 10))
            kymograph = cpu_dynamics[kymograp_index, 0, :][0][:, ::2]
            kymograph = np.repeat(kymograph, 60, axis=0)
            
            plt.imshow(kymograph.T, cmap=colormap, norm=normalize)
            ax = plt.gca()
            plt.yticks(np.arange(0, self.T/2, 1498), [0, 10, 20, 30, 40, 50, 60], fontsize=30, fontweight="bold")
            plt.xticks([])
            plt.ylabel('Time (min)', fontsize=30, fontweight='bold')
            
            if self.size == 'xsmall':
                plt.xlabel('Position', fontsize=20, fontweight='bold')
            else:
                plt.xlabel('Position', fontsize=30, fontweight='bold')
            
            if self.save:
                if path is not None:
                    Path(path).mkdir(parents=True, exist_ok=True)
                    output_file = Path(path) / f"{self.size}Disc_Kymograph_{self.sim_number}_{self.saveName}.png"
                    fig.savefig(output_file, transparent=True, bbox_inches="tight")
                    print(f"Kymograph saved to {output_file}")
                else:
                    print("Please provide a path for saving images")
        
        plt.close(fig)  # Close the figure to free memory
        del kymograph
    
    def get_cpu_data(self):
        """Return simulation data transferred to CPU for analysis and visualization"""
        if HAS_GPU:
            return {
                'disc_dynamics': self.disc_dynamics.get(),
                'VPLC_state': self.VPLC_state.get()
            }
        else:
            return {
                'disc_dynamics': self.disc_dynamics,
                'VPLC_state': self.VPLC_state
            }
    
    def cleanup(self):
        """Release GPU memory"""
        if HAS_GPU:
            try:
                mempool = cp.get_default_memory_pool()
                pinned_mempool = cp.get_default_pinned_memory_pool()
                mempool.free_all_blocks()
                pinned_mempool.free_all_blocks()
            except Exception as e:
                print(f"Warning: Error during GPU memory cleanup: {e}")


class SimulationRunner:
    """Handles the simulation execution with GPU acceleration"""
    
    def __init__(self, update_progress_callback=None, log_output_callback=None):
        self.update_progress_callback = update_progress_callback
        self.log_output_callback = log_output_callback
        self.simulation_running = False
        self.batch_running = False
        self.pouch = None
        self.current_batch_sim = 0
        self.total_batch_sims = 0
        
        # Parameter ranges with validation (min, max, default, description)
        self.parameter_ranges = {
            'K_PLC': (0.01, 1.0, 0.2, "IP3 receptor calcium binding coefficient"),
            'K_5': (0.1, 2.0, 0.66, "IP3 degradation rate"),
            'k_1': (0.1, 5.0, 1.11, "IP3 receptor calcium release rate"),
            'k_a': (0.01, 0.5, 0.08, "Activating calcium binding coefficient"),
            'k_p': (0.01, 0.5, 0.13, "IP3 binding coefficient"),
            'k_2': (0.001, 0.1, 0.0203, "Calcium leak from ER"),
            'V_SERCA': (0.1, 2.0, 0.9, "Maximum SERCA pump rate"),
            'K_SERCA': (0.01, 0.5, 0.1, "SERCA pump binding coefficient"),
            'c_tot': (0.5, 5.0, 2.0, "Total calcium concentration"),
            'beta': (0.05, 0.5, 0.185, "ER/cytosol volume ratio"),
            'k_i': (0.1, 1.0, 0.4, "Inhibiting calcium binding coefficient"),
            'D_p': (0.001, 0.02, 0.005, "IP3 diffusion coefficient"),
            'tau_max': (100, 2000, 800, "Maximum time constant for IP3R inactivation"),
            'k_tau': (0.5, 5.0, 1.5, "IP3R inactivation calcium threshold"),
            'lower': (0.05, 2.0, 0.4, "Lower bound of standby cell VPLCs"),
            'upper': (0.1, 2.0, 0.8, "Upper bound of standby cell VPLCs"),
            'frac': (0.001, 0.05, 0.007680491551459293, "Fraction of initiator cells"),
            'D_c_ratio': (0.01, 0.5, 0.1, "Calcium/IP3 diffusion ratio")
        }
        
        # Simulation types
        self.simulation_types = {
            "Single cell spikes": {"lower_VPLC": 0.1, "upper_VPLC": 0.5, "save_name": "Spikes"},
            "Intercellular transients": {"lower_VPLC": 0.25, "upper_VPLC": 0.6, "save_name": "ICT"},
            "Intercellular waves": {"lower_VPLC": 0.4, "upper_VPLC": 0.8, "save_name": "ICW"},
            "Fluttering": {"lower_VPLC": 1.4, "upper_VPLC": 1.5, "save_name": "Fluttering"}
        }
    
    def log(self, message):
        """Log a message through the callback if available"""
        if self.log_output_callback:
            self.log_output_callback(message)
        else:
            print(message)
    
    def update_progress(self, value, status=None):
        """Update progress through the callback if available"""
        if self.update_progress_callback:
            self.update_progress_callback(value, status)
    
    def validate_parameters(self, params):
        """Validate parameters against their allowed ranges"""
        for param, value in params.items():
            min_val, max_val, _, _ = self.parameter_ranges[param]
            if value < min_val or value > max_val:
                return False, f"Parameter {param} value {value} is outside allowed range [{min_val}, {max_val}]"
        
        # Additional validation: upper should be greater than lower for VPLC
        if params['upper'] <= params['lower']:
            return False, f"Upper VPLC value ({params['upper']}) must be greater than lower VPLC value ({params['lower']})"
        
        return True, "Parameters valid"
    
    def generate_random_parameters(self, base_params, random_params, min_values, max_values):
        """Generate random parameters within specified ranges for batch simulation"""
        params = base_params.copy()
        
        for param, randomize in random_params.items():
            if randomize:
                # Use random value within the specified min/max range
                min_val = min_values[param]
                max_val = max_values[param]
                params[param] = random.uniform(min_val, max_val)
        
        return params
    
    def check_geometry_files(self):
        """Check if geometry files are available and download if needed"""
        try:
            # We need to import the download function here
            _, download_geometry_files, _ = import_simulation_module()
            
            self.log("Checking for geometry files...")
            success = download_geometry_files()
            if success:
                self.log("Geometry files ready.\n")
                return True
            else:
                self.log("ERROR: Geometry files not available. Please download them manually.\n")
                return False
        except Exception as e:
            self.log(f"Error checking geometry files: {str(e)}\n")
            traceback.print_exc()
            return False
    
    def run_single_simulation(self, params, pouch_size, output_folder, sim_name=None, sim_number=None, generate_animation=False):
        """Run a single simulation with the given parameters on GPU"""
        if self.simulation_running:
            self.log("A simulation is already running. Please wait for it to finish.")
            return False
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Validate parameters
        valid, message = self.validate_parameters(params)
        if not valid:
            self.log(f"Parameter validation error: {message}\n")
            return False
        
        # Use current time as simulation number if not provided
        if sim_number is None:
            sim_number = int(time.time()) % 100000
        
        # Use default save name if not provided
        if sim_name is None:
            save_name = f"Demo_{sim_number}"
        else:
            save_name = sim_name
        
        # Log parameters
        self.log(f"\n{'='*80}\n")
        mode_str = "GPU-accelerated" if HAS_GPU else "CPU"
        self.log(f"Running {mode_str} simulation with {pouch_size} pouch size\n")
        self.log(f"Output folder: {output_folder}\n")
        self.log("Parameters:\n")
        for param, value in params.items():
            self.log(f"  {param}: {value}\n")
        self.log("\n")
        
        # Set running flag
        self.simulation_running = True
        
        # Run simulation in a thread
        thread = threading.Thread(
            target=self._simulation_thread,
            args=(params, pouch_size, save_name, output_folder, sim_number, generate_animation, False, None),
            daemon=True
        )
        thread.start()
        
        return True
    
    def run_batch_simulations(self, base_params, pouch_size, output_folder, batch_count, 
                             random_params, min_values, max_values, 
                             prefix="Batch", generate_animations=False, 
                             batch_complete_callback=None):
        """Run multiple simulations with randomized parameters on GPU"""
        if self.simulation_running or self.batch_running:
            self.log("A simulation is already running. Please wait for it to finish.")
            return False
        
        # Check for valid batch count
        if batch_count <= 0:
            self.log("Number of simulations must be greater than 0")
            return False
        
        # Check that at least one parameter is set to be randomized
        any_randomized = False
        for param, randomize in random_params.items():
            if randomize:
                any_randomized = True
                break
        
        if not any_randomized:
            self.log("No parameters are selected for randomization. Please select at least one parameter to randomize.")
            return False
        
        # Set batch running flags
        self.batch_running = True
        self.current_batch_sim = 0
        self.total_batch_sims = batch_count
        
        # Start batch processing
        mode_str = "GPU-accelerated" if HAS_GPU else "CPU"
        self.log(f"Starting batch of {batch_count} {mode_str} simulations\n")
        self.log(f"Randomized parameters: ")
        for param, randomize in random_params.items():
            if randomize:
                min_val = min_values[param]
                max_val = max_values[param]
                self.log(f"{param} [{min_val:.4f}-{max_val:.4f}], ")
        self.log("\n\n")
        
        # Process the first simulation
        self._process_next_batch_simulation(
            base_params, pouch_size, output_folder, 
            random_params, min_values, max_values,
            prefix, generate_animations, batch_complete_callback
        )
        
        return True
    
    def _process_next_batch_simulation(self, base_params, pouch_size, output_folder, 
                                     random_params, min_values, max_values,
                                     prefix, generate_animations, batch_complete_callback):
        """Process the next simulation in the batch"""
        if not self.batch_running:
            return
        
        self.current_batch_sim += 1
        
        if self.current_batch_sim <= self.total_batch_sims:
            # Generate random parameters
            batch_params = self.generate_random_parameters(
                base_params, random_params, min_values, max_values
            )
            
            # Update progress
            batch_progress = (self.current_batch_sim - 1) / self.total_batch_sims * 100
            self.update_progress(batch_progress, f"Batch simulation {self.current_batch_sim}/{self.total_batch_sims}")
            
            # Create sim name with prefix and index
            sim_name = f"{prefix}_{self.current_batch_sim}"
            
            # Set running flag
            self.simulation_running = True
            
            # Run simulation in a thread
            thread = threading.Thread(
                target=self._simulation_thread,
                args=(
                    batch_params, pouch_size, sim_name, output_folder, 
                    self.current_batch_sim, generate_animations, True,
                    lambda: self._process_next_batch_simulation(
                        base_params, pouch_size, output_folder, 
                        random_params, min_values, max_values,
                        prefix, generate_animations, batch_complete_callback
                    )
                ),
                daemon=True
            )
            thread.start()
        else:
            # All simulations complete
            self.batch_running = False
            self.update_progress(100, "Batch simulations complete")
            self.log("\nAll batch simulations completed!\n")
            
            # Free GPU memory
            if HAS_GPU and self.pouch:
                self.pouch.cleanup()
            
            # Call batch complete callback if provided
            if batch_complete_callback:
                batch_complete_callback()
    
    def _simulation_thread(self, sim_params, pouch_size, save_name, output_folder, 
                         sim_number, generate_animation, batch_mode, next_batch_callback):
        """Run the GPU-accelerated simulation in a background thread"""
        try:
            # Create GpuPouch object
            self.log("Creating simulation object...\n")
            self.pouch = GpuPouch(params=sim_params, size=pouch_size, sim_number=sim_number, save=True, saveName=save_name)
            
            # Run simulation with progress updates
            mode_str = "GPU-accelerated" if HAS_GPU else "CPU"
            self.log(f"Running {mode_str} simulation...\n")
            
            # Track simulation time
            start_time = time.time()
            
            # Run the simulation on GPU
            self.pouch.simulate()
            
            # Check if simulation should continue
            if self.simulation_running and (self.batch_running if batch_mode else True):
                # Generate outputs
                self.log("\nGenerating VPLC profile...\n")
                if batch_mode:
                    batch_progress = ((self.current_batch_sim - 1) + 0.9) / self.total_batch_sims * 100
                    self.update_progress(batch_progress, f"Batch {self.current_batch_sim}/{self.total_batch_sims} - Generating VPLC profile...")
                else:
                    self.update_progress(90, "Generating VPLC profile...")
                
                self.pouch.draw_profile(output_folder)
                
                self.log("Generating kymograph...\n")
                if batch_mode:
                    batch_progress = ((self.current_batch_sim - 1) + 0.95) / self.total_batch_sims * 100
                    self.update_progress(batch_progress, f"Batch {self.current_batch_sim}/{self.total_batch_sims} - Generating kymograph...")
                else:
                    self.update_progress(95, "Generating kymograph...")
                
                self.pouch.draw_kymograph(output_folder)
                
                # Generate animation if requested
                if generate_animation:
                    self.log("Generating animation...\n")
                    if batch_mode:
                        batch_progress = ((self.current_batch_sim - 1) + 0.99) / self.total_batch_sims * 100
                        self.update_progress(batch_progress, f"Batch {self.current_batch_sim}/{self.total_batch_sims} - Generating animation...")
                    else:
                        self.update_progress(98, "Generating animation...")
                    
                    self.pouch.make_animation(output_folder)
                
                self.log("\nSimulation complete!\n")
                self.log(f"Results saved to: {output_folder}\n")
                
                if not batch_mode:
                    self.update_progress(100, "Simulation complete")
            
        except Exception as e:
            self.log(f"\nError during simulation: {str(e)}\n")
            traceback.print_exc()
        finally:
            self.simulation_running = False
            
            # Clean up GPU memory between simulations
            if not batch_mode and HAS_GPU and self.pouch:
                self.pouch.cleanup()
            
            if batch_mode and next_batch_callback:
                # Continue with next batch simulation
                next_batch_callback()
    
    def generate_animation(self, output_folder, animation_complete_callback=None):
        """Generate animation from the simulation results"""
        if self.pouch is None:
            self.log("No simulation results available. Run a simulation first.")
            return False
        
        # Run animation generation in a thread
        thread = threading.Thread(
            target=self._animation_thread,
            args=(output_folder, animation_complete_callback),
            daemon=True
        )
        thread.start()
        
        return True
    
    def _animation_thread(self, output_folder, animation_complete_callback):
        """Generate animation in a background thread"""
        try:
            self.log("\nGenerating animation (this may take several minutes)...\n")
            self.update_progress(0, "Starting animation generation...")
            
            self.pouch.make_animation(output_folder)
            
            self.log("\nAnimation generation complete!\n")
            self.update_progress(100, "Animation complete")
            
        except Exception as e:
            self.log(f"\nError generating animation: {str(e)}\n")
            traceback.print_exc()
        finally:
            if animation_complete_callback:
                animation_complete_callback()
    
    def stop_simulation(self):
        """Stop the running simulation"""
        if self.simulation_running:
            self.simulation_running = False
            self.batch_running = False
            self.log("\nStopping simulation...\n")
            
            # Clean up GPU memory when stopping
            if HAS_GPU and self.pouch:
                self.pouch.cleanup()
            
            return True
        return False


class CalciumSignalingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NVIDIA GPU-Accelerated Calcium Signaling Simulation")
        self.root.geometry("1200x900")
        self.root.minsize(1000, 800)
        
        self.message_queue = queue.Queue()
        self.save_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulationResults")
        
        # Create the simulation runner with callbacks
        self.simulation_runner = SimulationRunner(
            update_progress_callback=self.update_progress,
            log_output_callback=self.log_output
        )
        
        # Set up the simulation types and their parameters
        self.simulation_types = self.simulation_runner.simulation_types
        self.parameter_ranges = self.simulation_runner.parameter_ranges
        
        # Create parameter variables to store the current values
        self.param_vars = {}
        self.param_min_vars = {}
        self.param_max_vars = {}
        self.param_random_vars = {}
        
        for param, (min_val, max_val, default, _) in self.parameter_ranges.items():
            self.param_vars[param] = tk.DoubleVar(value=default)
            self.param_min_vars[param] = tk.DoubleVar(value=min_val)
            self.param_max_vars[param] = tk.DoubleVar(value=max_val)
            self.param_random_vars[param] = tk.BooleanVar(value=False)
        
        self.create_widgets()
        
        # Check for geometry files on startup
        threading.Thread(target=self.check_geometry_files, daemon=True).start()
        
        # Check for GPU capabilities
        if not HAS_GPU:
            messagebox.showwarning(
                "GPU Acceleration Not Available", 
                "NVIDIA GPU acceleration is not available. The simulation will run on CPU which will be significantly slower.\n\n"
                "Possible solutions:\n"
                "1. Make sure you have an NVIDIA GPU\n"
                "2. Install CUDA Toolkit 11.x\n"
                "3. Install cupy with: pip install cupy-cuda11x\n"
                "4. Ensure CUDA DLLs are in your PATH"
            )
            self.log_output("WARNING: Running in CPU mode. Install CUDA and CuPy for GPU acceleration.\n")
            self.log_output("CUDA installation guide: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/\n")
    
    def create_widgets(self):
        # Main frame to hold everything
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top section - simulation type and pouch size
        top_frame = ttk.LabelFrame(main_frame, text="Simulation Settings", padding="10")
        top_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # GPU indicator
        gpu_status = "ENABLED" if HAS_GPU else "DISABLED"
        gpu_color = "green" if HAS_GPU else "red"
        gpu_label = ttk.Label(top_frame, text=f"GPU Acceleration: {gpu_status}", foreground=gpu_color, font=("Arial", 10, "bold"))
        gpu_label.grid(row=0, column=4, padx=5, pady=5)
        
        # Simulation type
        ttk.Label(top_frame, text="Simulation Type:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.sim_type_var = tk.StringVar(value="Intercellular waves")
        sim_type_combo = ttk.Combobox(top_frame, textvariable=self.sim_type_var, 
                                     values=list(self.simulation_types.keys()),
                                     state="readonly", width=25)
        sim_type_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        sim_type_combo.bind("<<ComboboxSelected>>", self.on_sim_type_change)
        
        # Pouch size
        ttk.Label(top_frame, text="Pouch Size:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.pouch_size_var = tk.StringVar(value="large")
        pouch_size_combo = ttk.Combobox(top_frame, textvariable=self.pouch_size_var, 
                                       values=["xsmall", "small", "medium", "large"],
                                       state="readonly", width=15)
        pouch_size_combo.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Output folder
        ttk.Label(top_frame, text="Output Folder:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.output_folder_var = tk.StringVar(value=self.save_folder)
        output_entry = ttk.Entry(top_frame, textvariable=self.output_folder_var, width=50)
        output_entry.grid(row=1, column=1, columnspan=3, sticky=tk.W+tk.E, padx=5, pady=5)
        browse_btn = ttk.Button(top_frame, text="Browse...", command=self.browse_output_folder)
        browse_btn.grid(row=1, column=4, sticky=tk.W, padx=5, pady=5)
        
        # Creating a notebook for parameters, batch, and output
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Parameters tab
        params_frame = ttk.Frame(notebook, padding="10")
        notebook.add(params_frame, text="Parameters")
        
        # Batch tab
        batch_frame = ttk.Frame(notebook, padding="10")
        notebook.add(batch_frame, text="Batch Generation")
        
        # Output tab
        output_frame = ttk.Frame(notebook)
        notebook.add(output_frame, text="Output")
        
        # Create a canvas with scrollbar for parameters
        canvas_frame = ttk.Frame(params_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(canvas_frame)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add parameter sliders to the scrollable frame
        header_frame = ttk.Frame(scrollable_frame)
        header_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(header_frame, text="Parameter", width=15).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(header_frame, text="Value", width=30).grid(row=0, column=1)
        ttk.Label(header_frame, text="Current", width=8).grid(row=0, column=2)
        ttk.Label(header_frame, text="Description", width=40).grid(row=0, column=3, sticky=tk.W)
        
        # Group parameters
        param_frame = ttk.Frame(scrollable_frame)
        param_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add sliders for each parameter
        for row, param in enumerate(sorted(self.parameter_ranges.keys())):
            min_val, max_val, default, desc = self.parameter_ranges[param]
            
            # Parameter label
            param_label = ttk.Label(param_frame, text=f"{param}:")
            param_label.grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            
            # Parameter slider
            slider = ttk.Scale(param_frame, from_=min_val, to=max_val, 
                             variable=self.param_vars[param], 
                             orient=tk.HORIZONTAL, length=300)
            slider.grid(row=row, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
            
            # Parameter value display
            value_label = ttk.Label(param_frame, textvariable=self.param_vars[param], width=8)
            value_label.grid(row=row, column=2, sticky=tk.W, padx=5, pady=2)
            
            # Parameter description
            desc_label = ttk.Label(param_frame, text=desc, wraplength=300)
            desc_label.grid(row=row, column=3, sticky=tk.W, padx=5, pady=2)
        
        # Add preset buttons
        preset_frame = ttk.Frame(params_frame)
        preset_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Label(preset_frame, text="Load Presets:").pack(side=tk.LEFT, padx=5)
        
        for preset_name in self.simulation_types.keys():
            preset_btn = ttk.Button(preset_frame, text=preset_name,
                                  command=lambda p=preset_name: self.load_preset(p))
            preset_btn.pack(side=tk.LEFT, padx=5)
        
        # Create batch tab content
        batch_top_frame = ttk.Frame(batch_frame)
        batch_top_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(batch_top_frame, text="Number of Simulations:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.batch_count_var = tk.IntVar(value=5)
        batch_count_spinbox = ttk.Spinbox(batch_top_frame, from_=1, to=100, textvariable=self.batch_count_var, width=5)
        batch_count_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(batch_top_frame, text="Prefix for Saved Files:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.batch_prefix_var = tk.StringVar(value="Batch")
        batch_prefix_entry = ttk.Entry(batch_top_frame, textvariable=self.batch_prefix_var, width=15)
        batch_prefix_entry.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Auto-generate animations checkbox
        self.batch_animations_var = tk.BooleanVar(value=False)
        batch_animations_check = ttk.Checkbutton(batch_top_frame, text="Generate animations", 
                                              variable=self.batch_animations_var)
        batch_animations_check.grid(row=0, column=4, sticky=tk.W, padx=5, pady=5)
        
        # Create parameter range settings for batch generation
        batch_params_frame = ttk.Frame(batch_frame)
        batch_params_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Headers
        ttk.Label(batch_params_frame, text="Parameter", width=15).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(batch_params_frame, text="Randomize", width=10).grid(row=0, column=1)
        ttk.Label(batch_params_frame, text="Min Value", width=10).grid(row=0, column=2)
        ttk.Label(batch_params_frame, text="Max Value", width=10).grid(row=0, column=3)
        ttk.Label(batch_params_frame, text="Description", width=40).grid(row=0, column=4, sticky=tk.W)
        
        # Add parameter range controls
        for row, param in enumerate(sorted(self.parameter_ranges.keys()), 1):
            min_val, max_val, default, desc = self.parameter_ranges[param]
            
            # Parameter label
            param_label = ttk.Label(batch_params_frame, text=f"{param}:")
            param_label.grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
            
            # Randomize checkbox
            random_check = ttk.Checkbutton(batch_params_frame, variable=self.param_random_vars[param])
            random_check.grid(row=row, column=1, padx=5, pady=2)
            
            # Min value entry
            min_entry = ttk.Entry(batch_params_frame, textvariable=self.param_min_vars[param], width=10)
            min_entry.grid(row=row, column=2, padx=5, pady=2)
            
            # Max value entry
            max_entry = ttk.Entry(batch_params_frame, textvariable=self.param_max_vars[param], width=10)
            max_entry.grid(row=row, column=3, padx=5, pady=2)
            
            # Parameter description
            desc_label = ttk.Label(batch_params_frame, text=desc, wraplength=300)
            desc_label.grid(row=row, column=4, sticky=tk.W, padx=5, pady=2)
        
        # Add parameter preset buttons for batch
        batch_preset_frame = ttk.Frame(batch_frame)
        batch_preset_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Label(batch_preset_frame, text="Randomize Selected Parameters For:").pack(side=tk.LEFT, padx=5)
        
        for preset_name in self.simulation_types.keys():
            preset_btn = ttk.Button(batch_preset_frame, text=preset_name,
                                  command=lambda p=preset_name: self.set_batch_preset(p))
            preset_btn.pack(side=tk.LEFT, padx=5)
        
        # Add batch run button
        batch_run_btn = ttk.Button(batch_frame, text="Run Batch Simulations", 
                                 command=self.run_batch_simulations)
        batch_run_btn.pack(pady=10)
        
        # Create text widget for output
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=20)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.output_text.configure(state="disabled")
        
        # Bottom action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # Adjust button text based on GPU availability
        run_text = "Run GPU Simulation" if HAS_GPU else "Run Simulation (CPU mode)"
        self.run_button = ttk.Button(button_frame, text=run_text, command=self.run_simulation)
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_simulation, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.animation_button = ttk.Button(button_frame, text="Generate Animation", command=self.generate_animation, state=tk.DISABLED)
        self.animation_button.pack(side=tk.LEFT, padx=5)
        
        self.open_folder_button = ttk.Button(button_frame, text="Open Output Folder", command=self.open_output_folder)
        self.open_folder_button.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=5)
    
    def on_sim_type_change(self, event):
        """Update parameter values when simulation type changes"""
        sim_type = self.sim_type_var.get()
        params = self.simulation_types[sim_type]
        
        # Update lower and upper VPLC values
        self.param_vars['lower'].set(params['lower_VPLC'])
        self.param_vars['upper'].set(params['upper_VPLC'])
        
        # Update save folder
        save_name = params['save_name']
        base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulationResults")
        self.save_folder = os.path.join(base_folder, f"Pouch_Visualization_{save_name}")
        self.output_folder_var.set(self.save_folder)
    
    def load_preset(self, preset_name):
        """Load parameter presets for a specific simulation type"""
        if preset_name in self.simulation_types:
            # Set the simulation type combobox
            self.sim_type_var.set(preset_name)
            
            # Update the parameters
            params = self.simulation_types[preset_name]
            self.param_vars['lower'].set(params['lower_VPLC'])
            self.param_vars['upper'].set(params['upper_VPLC'])
            
            # Reset other parameters to defaults
            for param, (min_val, max_val, default, _) in self.parameter_ranges.items():
                if param not in ['lower', 'upper']:
                    self.param_vars[param].set(default)
            
            # Update save folder
            save_name = params['save_name']
            base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulationResults")
            self.save_folder = os.path.join(base_folder, f"Pouch_Visualization_{save_name}")
            self.output_folder_var.set(self.save_folder)
            
            messagebox.showinfo("Preset Loaded", f"Loaded parameters for '{preset_name}' simulation")
    
    def set_batch_preset(self, preset_name):
        """Set batch randomization presets for a specific simulation type"""
        if preset_name in self.simulation_types:
            # Reset all randomization flags to False
            for param in self.param_random_vars:
                self.param_random_vars[param].set(False)
            
            # Set min/max values to the parameter ranges
            for param, (min_val, max_val, _, _) in self.parameter_ranges.items():
                self.param_min_vars[param].set(min_val)
                self.param_max_vars[param].set(max_val)
            
            # Set specific parameters based on the simulation type
            if preset_name == "Single cell spikes":
                # For single cell spikes, randomize calcium-related parameters
                for param in ['K_PLC', 'K_5', 'k_1', 'k_a', 'k_p', 'k_2']:
                    self.param_random_vars[param].set(True)
                
            elif preset_name == "Intercellular transients":
                # For ICT, randomize diffusion and cell-to-cell communication
                for param in ['D_p', 'D_c_ratio', 'frac']:
                    self.param_random_vars[param].set(True)
                
            elif preset_name == "Intercellular waves":
                # For ICW, randomize wave propagation parameters
                for param in ['lower', 'upper', 'D_p', 'D_c_ratio']:
                    self.param_random_vars[param].set(True)
                
            elif preset_name == "Fluttering":
                # For fluttering, randomize ER-related parameters
                for param in ['V_SERCA', 'K_SERCA', 'c_tot', 'beta']:
                    self.param_random_vars[param].set(True)
            
            # Always randomize these parameters for variety
            self.param_random_vars['frac'].set(True)
            
            messagebox.showinfo("Batch Preset Applied", 
                              f"Randomization settings updated for '{preset_name}' simulation. " +
                              f"You can adjust which parameters to randomize in the table.")
    
    def browse_output_folder(self):
        """Open a dialog to select output folder"""
        folder = filedialog.askdirectory(initialdir=self.save_folder)
        if folder:
            self.save_folder = folder
            self.output_folder_var.set(folder)
    
    def open_output_folder(self):
        """Open the output folder in file explorer"""
        folder = self.output_folder_var.get()
        if os.path.exists(folder):
            if sys.platform == 'win32':
                os.startfile(folder)
            elif sys.platform == 'darwin':  # macOS
                subprocess.call(['open', folder])
            else:  # Linux
                subprocess.call(['xdg-open', folder])
        else:
            messagebox.showinfo("Folder Not Found", "The output folder does not exist yet. Run a simulation first.")
    
    def check_geometry_files(self):
        """Check if geometry files are available and download if needed"""
        success = self.simulation_runner.check_geometry_files()
        if not success:
            messagebox.showerror("Geometry Files Missing", 
                              "Geometry files required for simulation are missing and couldn't be downloaded automatically. " + 
                              "Please download them manually according to the instructions in the output window.")
    
    def log_output(self, message):
        """Add a message to the output text widget"""
        self.message_queue.put(message)
        self.root.after(10, self.process_message_queue)
    
    def process_message_queue(self):
        """Process messages in the queue and update the output text widget"""
        try:
            while True:
                message = self.message_queue.get_nowait()
                self.output_text.configure(state="normal")
                self.output_text.insert(tk.END, message)
                self.output_text.see(tk.END)
                self.output_text.configure(state="disabled")
                self.message_queue.task_done()
        except queue.Empty:
            pass
    
    def update_progress(self, value, status=None):
        """Update the progress bar and status text"""
        self.progress_var.set(value)
        if status:
            self.status_var.set(status)
        self.root.update_idletasks()
    
    def create_parameter_dict(self):
        """Create a parameter dictionary from current GUI values"""
        params = {}
        for param, var in self.param_vars.items():
            params[param] = var.get()
        return params
    
    def create_random_parameters_dict(self):
        """Create dictionary indicating which parameters should be randomized"""
        random_params = {}
        for param, var in self.param_random_vars.items():
            random_params[param] = var.get()
        return random_params
    
    def create_min_values_dict(self):
        """Create dictionary with minimum values for randomized parameters"""
        min_values = {}
        for param, var in self.param_min_vars.items():
            min_values[param] = var.get()
        return min_values
    
    def create_max_values_dict(self):
        """Create dictionary with maximum values for randomized parameters"""
        max_values = {}
        for param, var in self.param_max_vars.items():
            max_values[param] = var.get()
        return max_values
    
    def run_simulation(self):
        """Run single GPU-accelerated simulation with current parameters"""
        # Get current parameters
        params = self.create_parameter_dict()
        pouch_size = self.pouch_size_var.get()
        output_folder = self.output_folder_var.get()
        
        # Update UI before starting
        self.run_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.animation_button.configure(state=tk.DISABLED)
        
        # Clear output window
        self.output_text.configure(state="normal")
        self.output_text.delete(1.0, tk.END)
        self.output_text.configure(state="disabled")
        
        # Run the simulation
        sim_type = self.sim_type_var.get()
        save_name = f"Demo_{self.simulation_types[sim_type]['save_name']}"
        
        # If GPU acceleration is available, show a notice
        if HAS_GPU:
            self.log_output("Running with NVIDIA GPU acceleration (float32 precision)\n")
        else:
            self.log_output("WARNING: Running without GPU acceleration. This will be much slower.\n")
            self.log_output("To enable GPU acceleration, install CUDA and CuPy:\n")
            self.log_output("1. Download and install CUDA Toolkit 11.x from NVIDIA website\n")
            self.log_output("2. Install CuPy with: pip install cupy-cuda11x\n")
            self.log_output("3. Make sure NVIDIA DLLs are in your PATH\n\n")
        
        success = self.simulation_runner.run_single_simulation(
            params, pouch_size, output_folder, save_name,
            generate_animation=False
        )
        
        if not success:
            # If simulation couldn't start, reset UI
            self.run_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)
    
    def run_batch_simulations(self):
        """Run batch GPU-accelerated simulations with randomized parameters"""
        # Get current parameters and settings
        base_params = self.create_parameter_dict()
        random_params = self.create_random_parameters_dict()
        min_values = self.create_min_values_dict()
        max_values = self.create_max_values_dict()
        
        batch_count = self.batch_count_var.get()
        prefix = self.batch_prefix_var.get()
        generate_animations = self.batch_animations_var.get()
        
        pouch_size = self.pouch_size_var.get()
        output_folder = self.output_folder_var.get()
        
        # Update UI before starting
        self.run_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.animation_button.configure(state=tk.DISABLED)
        
        # Clear output window
        self.output_text.configure(state="normal")
        self.output_text.delete(1.0, tk.END)
        self.output_text.configure(state="disabled")
        
        # If GPU acceleration is available, show a notice
        if HAS_GPU:
            self.log_output("Running batch simulations with NVIDIA GPU acceleration (float32 precision)\n")
        else:
            self.log_output("WARNING: Running batch without GPU acceleration. This will be much slower.\n")
            self.log_output("To enable GPU acceleration, install CUDA and CuPy:\n")
            self.log_output("1. Download and install CUDA Toolkit 11.x from NVIDIA website\n")
            self.log_output("2. Install CuPy with: pip install cupy-cuda11x\n")
            self.log_output("3. Make sure NVIDIA DLLs are in your PATH\n\n")
        
        # Run the batch simulation
        success = self.simulation_runner.run_batch_simulations(
            base_params, pouch_size, output_folder, batch_count,
            random_params, min_values, max_values, 
            prefix, generate_animations,
            self.batch_complete
        )
        
        if not success:
            # If batch couldn't start, reset UI
            self.run_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)
    
    def batch_complete(self):
        """Callback when batch simulation is complete"""
        self.run_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        if self.simulation_runner.pouch is not None:
            self.animation_button.configure(state=tk.NORMAL)
    
    def stop_simulation(self):
        """Stop the running simulation"""
        success = self.simulation_runner.stop_simulation()
        if success:
            self.status_var.set("Stopping simulation...")
    
    def generate_animation(self):
        """Generate animation from the simulation results"""
        if self.simulation_runner.pouch is None:
            messagebox.showinfo("No Simulation", "No simulation results available. Run a simulation first.")
            return
        
        # Update UI before starting
        self.run_button.configure(state=tk.DISABLED)
        self.animation_button.configure(state=tk.DISABLED)
        
        # Generate the animation
        output_folder = self.output_folder_var.get()
        self.simulation_runner.generate_animation(output_folder, self.animation_complete)
    
    def animation_complete(self):
        """Callback when animation generation is complete"""
        self.run_button.configure(state=tk.NORMAL)
        self.animation_button.configure(state=tk.NORMAL)


def main():
    # Create the GPU-accelerated GUI
    root = tk.Tk()
    app = CalciumSignalingGUI(root)
    
    # Set up icon if available
    try:
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calcium_icon.ico")
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
    except:
        pass
    
    # Start the main loop
    root.mainloop()


if __name__ == "__main__":
    main()