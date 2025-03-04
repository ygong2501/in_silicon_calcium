#!/usr/bin/env python3
# Calcium Signaling Simulation - GUI version

import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import subprocess
import urllib.request
import traceback
from pathlib import Path
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import queue

# Import the Pouch class and utilities from the original script
# Import at module level would normally be preferred, but putting it in a function
# to avoid import errors if calcium_signaling_simulation.py has issues
def import_simulation_module():
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from calcium_signaling_simulation import Pouch, download_geometry_files, setup_matplotlib_for_windows
    return Pouch, download_geometry_files, setup_matplotlib_for_windows

# Check and install required packages
required_packages = ['peakutils', 'py7zr', 'numpy', 'pandas', 'matplotlib', 'seaborn']
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Installing required package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# After ensuring they're installed
import peakutils

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


class CalciumSignalingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Calcium Signaling Simulation")
        self.root.geometry("1200x900")
        self.root.minsize(1000, 800)
        
        self.message_queue = queue.Queue()
        self.simulation_running = False
        self.pouch = None
        self.save_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulationResults")
        
        # Set up the simulation types and their parameters
        self.simulation_types = {
            "Single cell spikes": {"lower_VPLC": 0.1, "upper_VPLC": 0.5, "save_name": "Spikes"},
            "Intercellular transients": {"lower_VPLC": 0.25, "upper_VPLC": 0.6, "save_name": "ICT"},
            "Intercellular waves": {"lower_VPLC": 0.4, "upper_VPLC": 0.8, "save_name": "ICW"},
            "Fluttering": {"lower_VPLC": 1.4, "upper_VPLC": 1.5, "save_name": "Fluttering"}
        }
        
        # Default parameters
        self.default_params = {
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
            'lower': 0.4,  # Default to ICW
            'upper': 0.8,  # Default to ICW
            'frac': 0.007680491551459293,
            'D_c_ratio': 0.1
        }
        
        # Parameter ranges for sliders (min, max, default, description)
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
        
        # Create parameter variables to store the current values
        self.param_vars = {}
        for param, (min_val, max_val, default) in self.parameter_ranges.items():
            self.param_vars[param] = tk.DoubleVar(value=default)
        
        self.create_widgets()
        
        # Check for geometry files on startup
        threading.Thread(target=self.check_geometry_files, daemon=True).start()
    
    def create_widgets(self):
        # Main frame to hold everything
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top section - simulation type and pouch size
        top_frame = ttk.LabelFrame(main_frame, text="Simulation Settings", padding="10")
        top_frame.pack(fill=tk.X, padx=5, pady=5)
        
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
        output_entry.grid(row=1, column=1, columnspan=2, sticky=tk.W+tk.E, padx=5, pady=5)
        browse_btn = ttk.Button(top_frame, text="Browse...", command=self.browse_output_folder)
        browse_btn.grid(row=1, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Creating a notebook for parameters and output
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Parameters tab
        params_frame = ttk.Frame(notebook, padding="10")
        notebook.add(params_frame, text="Parameters")
        
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
        row = 0
        # Group parameters in three columns
        param_groups = []
        current_group = []
        for i, param in enumerate(sorted(self.parameter_ranges.keys())):
            current_group.append(param)
            if len(current_group) == 6 or i == len(self.parameter_ranges) - 1:
                param_groups.append(current_group)
                current_group = []
        
        # Create frames for each parameter group
        param_frames = []
        for group in param_groups:
            frame = ttk.Frame(scrollable_frame)
            frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
            param_frames.append(frame)
        
        # Add sliders to each parameter group frame
        for frame_idx, (frame, group) in enumerate(zip(param_frames, param_groups)):
            for row, param in enumerate(group):
                min_val, max_val, default = self.parameter_ranges[param]
                
                # Parameter label
                param_label = ttk.Label(frame, text=f"{param}:")
                param_label.grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
                
                # Parameter slider
                slider = ttk.Scale(frame, from_=min_val, to=max_val, 
                                  variable=self.param_vars[param], 
                                  orient=tk.HORIZONTAL, length=200)
                slider.grid(row=row, column=1, sticky=tk.W+tk.E, padx=5, pady=2)
                
                # Parameter value display
                value_label = ttk.Label(frame, textvariable=self.param_vars[param], width=8)
                value_label.grid(row=row, column=2, sticky=tk.W, padx=5, pady=2)
        
        # Add preset buttons
        preset_frame = ttk.Frame(params_frame)
        preset_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Label(preset_frame, text="Load Presets:").pack(side=tk.LEFT, padx=5)
        
        for preset_name in self.simulation_types.keys():
            preset_btn = ttk.Button(preset_frame, text=preset_name,
                                   command=lambda p=preset_name: self.load_preset(p))
            preset_btn.pack(side=tk.LEFT, padx=5)
        
        # Output tab
        output_frame = ttk.Frame(notebook)
        notebook.add(output_frame, text="Output")
        
        # Create text widget for output
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=20)
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.output_text.configure(state="disabled")
        
        # Bottom action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.run_button = ttk.Button(button_frame, text="Run Simulation", command=self.run_simulation)
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
            for param, (min_val, max_val, default) in self.parameter_ranges.items():
                if param not in ['lower', 'upper']:
                    self.param_vars[param].set(default)
            
            # Update save folder
            save_name = params['save_name']
            base_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulationResults")
            self.save_folder = os.path.join(base_folder, f"Pouch_Visualization_{save_name}")
            self.output_folder_var.set(self.save_folder)
            
            messagebox.showinfo("Preset Loaded", f"Loaded parameters for '{preset_name}' simulation")
    
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
        try:
            # We need to import the download function here
            _, download_geometry_files, _ = import_simulation_module()
            
            self.log_output("Checking for geometry files...")
            success = download_geometry_files()
            if success:
                self.log_output("Geometry files ready.\n")
            else:
                self.log_output("ERROR: Geometry files not available. Please download them manually.\n")
                messagebox.showerror("Geometry Files Missing", 
                                   "Geometry files required for simulation are missing and couldn't be downloaded automatically. " + 
                                   "Please download them manually according to the instructions in the output window.")
        except Exception as e:
            self.log_output(f"Error checking geometry files: {str(e)}\n")
            traceback.print_exc()
    
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
    
    def run_simulation(self):
        """Run the calcium signaling simulation"""
        if self.simulation_running:
            messagebox.showinfo("Simulation Running", "A simulation is already running. Please wait for it to finish.")
            return
        
        # Create output folder if it doesn't exist
        output_folder = self.output_folder_var.get()
        os.makedirs(output_folder, exist_ok=True)
        
        # Get simulation parameters
        sim_params = self.create_parameter_dict()
        pouch_size = self.pouch_size_var.get()
        sim_type = self.sim_type_var.get()
        save_name = self.simulation_types[sim_type]['save_name']
        
        # Update UI
        self.run_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.animation_button.configure(state=tk.DISABLED)
        self.update_progress(0, "Starting simulation...")
        
        # Clear output window
        self.output_text.configure(state="normal")
        self.output_text.delete(1.0, tk.END)
        self.output_text.configure(state="disabled")
        
        # Log parameters
        self.log_output(f"Running {sim_type} simulation with {pouch_size} pouch size\n")
        self.log_output(f"Output folder: {output_folder}\n")
        self.log_output("Parameters:\n")
        for param, value in sim_params.items():
            self.log_output(f"  {param}: {value}\n")
        self.log_output("\n")
        
        # Start simulation in a separate thread
        self.simulation_running = True
        threading.Thread(target=self.simulation_thread, 
                        args=(sim_params, pouch_size, sim_type, save_name, output_folder),
                        daemon=True).start()
    
    def simulation_thread(self, sim_params, pouch_size, sim_type, save_name, output_folder):
        """Run the simulation in a background thread"""
        try:
            # Import simulation class
            Pouch, _, setup_matplotlib_for_windows = import_simulation_module()
            
            # Set up matplotlib for Windows
            setup_matplotlib_for_windows()
            
            # Redirect stdout to capture output
            original_stdout = sys.stdout
            redirect = RedirectText(self.output_text)
            sys.stdout = redirect
            
            # Create Pouch object
            self.log_output("Creating simulation object...\n")
            self.pouch = Pouch(params=sim_params, size=pouch_size, sim_number=12345, save=True, saveName=f'Demo_{save_name}')
            
            # Run simulation with progress updates
            self.log_output("Running simulation...\n")
            
            # Override simulate method to track progress
            original_simulate = self.pouch.simulate
            
            def progress_simulate(*args, **kwargs):
                np.random.seed(self.pouch.sim_number)
                
                self.pouch.disc_dynamics[:,2,0] = (self.pouch.c_tot-self.pouch.disc_dynamics[:,0,0])/self.pouch.beta
                self.pouch.disc_dynamics[:,3,0] = np.random.uniform(.5,.7,size=(self.pouch.n_cells,1)).T
                self.pouch.VPLC_state = np.random.uniform(self.pouch.lower, self.pouch.upper, (self.pouch.n_cells,1))
                stimulated_cell_idxs = np.random.choice(self.pouch.n_cells, int(self.pouch.frac*self.pouch.n_cells))
                self.pouch.VPLC_state[stimulated_cell_idxs,0] = np.random.uniform(1.3,1.5,len(stimulated_cell_idxs))
                
                V_PLC = self.pouch.VPLC_state.reshape((self.pouch.n_cells,1))
                
                print("Starting simulation...")
                total_steps = self.pouch.T - 1
                update_interval = max(1, total_steps // 100)  # Update progress more often
                start_time = time.time()
                
                # ODE approximation solving
                for step in range(1, self.pouch.T):
                    if not self.simulation_running:
                        print("\nSimulation stopped by user.")
                        return
                    
                    # Progress update
                    if step % update_interval == 0:
                        progress = step / total_steps * 100
                        elapsed = time.time() - start_time
                        eta = (elapsed / step) * (total_steps - step)
                        status = f"Progress: {progress:.1f}% | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s"
                        print(status, end="\r")
                        self.root.after(0, lambda p=progress, s=status: self.update_progress(p, s))
                    
                    # ARRAY REFORMATTING
                    ca = self.pouch.disc_dynamics[:,0,step-1].reshape(-1,1)
                    ipt = self.pouch.disc_dynamics[:,1,step-1].reshape(-1,1)
                    s = self.pouch.disc_dynamics[:,2,step-1].reshape(-1,1)
                    r = self.pouch.disc_dynamics[:,3,step-1].reshape(-1,1)
                    ca_laplacian = self.pouch.D_c*np.dot(self.pouch.laplacian_matrix,ca)
                    ipt_laplacian = self.pouch.D_p*np.dot(self.pouch.laplacian_matrix,ipt)
                    
                    # ODE EQUATIONS
                    self.pouch.disc_dynamics[:,0,step]=(ca+self.pouch.dt*(ca_laplacian+(self.pouch.k_1*(np.divide(np.divide(r*np.multiply(ca,ipt),(self.pouch.k_a+ca)),(self.pouch.k_p+ipt)))**3 +self.pouch.k_2)*(s-ca)-self.pouch.V_SERCA*(ca**2)/(ca**2+self.pouch.K_SERCA**2))).T
                    self.pouch.disc_dynamics[:,1,step]=(ipt+self.pouch.dt*(ipt_laplacian+np.multiply(V_PLC,np.divide(ca**2,(ca**2+self.pouch.K_PLC**2)))-self.pouch.K_5*ipt)).T
                    self.pouch.disc_dynamics[:,2,step]=((self.pouch.c_tot-ca)/self.pouch.beta).T
                    self.pouch.disc_dynamics[:,3,step]=(r+self.pouch.dt*((self.pouch.k_tau**4+ca**4)/(self.pouch.tau_max*self.pouch.k_tau**4))*((1-r*(self.pouch.k_i+ca)/self.pouch.k_i))).T
                
                print(f"\nSimulation completed in {time.time() - start_time:.2f} seconds.")
            
            # Replace the simulate method with our progress-tracking version
            self.pouch.simulate = progress_simulate
            
            # Run simulation
            self.pouch.simulate()
            
            if self.simulation_running:  # Only continue if not stopped
                # Generate outputs
                self.log_output("\nGenerating VPLC profile...\n")
                self.update_progress(90, "Generating VPLC profile...")
                self.pouch.draw_profile(output_folder)
                
                self.log_output("Generating kymograph...\n")
                self.update_progress(95, "Generating kymograph...")
                self.pouch.draw_kymograph(output_folder)
                
                self.log_output("\nSimulation complete!\n")
                self.log_output(f"Results saved to: {output_folder}\n")
                self.update_progress(100, "Simulation complete")
            
            # Reset stdout
            sys.stdout = original_stdout
            redirect.stop()
            
        except Exception as e:
            self.log_output(f"\nError during simulation: {str(e)}\n")
            traceback.print_exc()
        finally:
            self.simulation_running = False
            self.root.after(0, self.simulation_finished)
    
    def simulation_finished(self):
        """Update UI after simulation is finished"""
        self.run_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        if self.pouch is not None:
            self.animation_button.configure(state=tk.NORMAL)
    
    def stop_simulation(self):
        """Stop the running simulation"""
        if self.simulation_running:
            self.simulation_running = False
            self.log_output("\nStopping simulation...\n")
            self.status_var.set("Stopping simulation...")
    
    def generate_animation(self):
        """Generate animation from the simulation results"""
        if self.pouch is None:
            messagebox.showinfo("No Simulation", "No simulation results available. Run a simulation first.")
            return
        
        # Run animation generation in a separate thread
        self.run_button.configure(state=tk.DISABLED)
        self.animation_button.configure(state=tk.DISABLED)
        self.update_progress(0, "Starting animation generation...")
        
        threading.Thread(target=self.animation_thread, daemon=True).start()
    
    def animation_thread(self):
        """Generate animation in a background thread"""
        try:
            output_folder = self.output_folder_var.get()
            
            # Redirect stdout to capture output
            original_stdout = sys.stdout
            redirect = RedirectText(self.output_text)
            sys.stdout = redirect
            
            self.log_output("\nGenerating animation (this may take several minutes)...\n")
            self.pouch.make_animation(output_folder)
            
            self.log_output("\nAnimation generation complete!\n")
            self.update_progress(100, "Animation complete")
            
            # Reset stdout
            sys.stdout = original_stdout
            redirect.stop()
            
        except Exception as e:
            self.log_output(f"\nError generating animation: {str(e)}\n")
            traceback.print_exc()
        finally:
            self.root.after(0, lambda: self.run_button.configure(state=tk.NORMAL))
            self.root.after(0, lambda: self.animation_button.configure(state=tk.NORMAL))


def main():
    # Create the GUI
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