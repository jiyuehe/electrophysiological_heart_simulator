# This code had been tested in MacOS and Ubuntu.
# Use Visual Studio Code as the IDE to run this code.
# Edit the "settings.json" file inside the hidden folder ".vscode" to your own python path.

# Dependencies: 
# pip install numpy
# pip install matplotlib 
# pip install plotly
# pip install scipy
# pip install numba

# %%
import codes
import os
from pathlib import Path
import numpy as np # pip install numpy
import matplotlib.pyplot as plt # pip install matplotlib
import plotly.io as pio
pio.renderers.default = "browser" # simulation result display in internet browser

script_dir = os.path.dirname(os.path.abspath(__file__)) # get the path of the current script
os.chdir(script_dir) # change the working directory
script_dir = Path(script_dir)

# %% 
# load geometry data file
# --------------------------------------------------
data_path = script_dir / 'data'
output = codes.load_geometry_data.execute(data_path)
voxel = output['voxel']
neighbor_id_2d = output['neighbor_id_2d']
Delta = output['Delta']
voxel_for_each_vertex = output['voxel_for_each_vertex']
vertex_for_each_voxel = output['vertex_for_each_voxel']
vertex = output['vertex']
face = output['face']
n_voxel = voxel.shape[0] 

# %% 
# simulation parameters
# --------------------------------------------------
dt = 0.05 # ms. if dt is not small enough, simulation will result nan. Generally, if c <= 1.0, can use dt = 0.05
t_final = 1000 # ms. NOTE: need to be at least long enough to have two pacings (pacing_cycle_length), or cannot compute activation phase
arrhythmia_flag = 0 # 0: focal arrhythmia. 1: rotor arrhythmia
compute_electrogram_flag = 1 # 1: compute electrogram. 0: do not compute electrogram
electrode_id = [0, 5000, 10000] # electrode locations for computing electrograms

# Mitchell-Schaeffer heart model parameters
c = 1 # diffusion coefficient
v_gate = 0.13 # gating variable threshold
heart_model_parameter = {
    'tau_in_voxel': np.ones(n_voxel) * 0.3, # tau_in
    'tau_out_voxel': np.ones(n_voxel) * 6, # tau_out
    'tau_open_voxel': np.ones(n_voxel) * 120, # tau_open
    'tau_close_voxel': np.ones(n_voxel) * 80, # tau_close
    'v_gate_voxel': np.ones(n_voxel) * v_gate, # gating variable threshold
    'c_voxel': c * np.ones(n_voxel), # diffusion coefficient
}

# rotor arrhythmia parameters
arrhythmia_parameters = {
    'pacing_start_time': 10, # ms
    'pacing_cycle_length': 250, # ms
    "s1_pacing_voxel_id": 23403, # location of s1 pacing site
    "s1_t": 0, # ms. time of s1 pacing
    "s1_s2_delta_t": 205 / dt, # ms. time interval between s1 and s2 pacing
    "ap_min": 0.004, # a threshold value of action potential 
    "ap_max": 0.030, # a threshold value of action potential 
    "h_min": 0.200, # a threshold value of gating variable
    "h_max": 0.300, # a threshold value of gating variable
    "s2_region_size_factor": 0.8, # a less than 1 multiplication factor to reduce s2 pacing region size
}

# %% 
# compute simulation
# --------------------------------------------------
# fiber orientations
D0 = codes.load_fiber.execute(n_voxel)

# compute heart model equation parts
P_2d = codes.compute_equation_part.execute(n_voxel, D0, neighbor_id_2d, heart_model_parameter)

# solve differential equations
action_potential, h = codes.compute_simulation.execute_CPU_parallel(neighbor_id_2d, n_voxel, dt, t_final, P_2d, Delta, arrhythmia_flag, arrhythmia_parameters)

# compute unipolar electrogram
if compute_electrogram_flag == 1:
    electrode_xyz = voxel[electrode_id, :]
    electrogram_unipolar = codes.compute_unipolar_electrogram.execute_CPU_parallel(electrode_xyz, voxel, D0, heart_model_parameter['c_voxel'], action_potential, Delta, neighbor_id_2d)

# save simulation results
simulation_results = {}
simulation_results['action_potential'] = action_potential
simulation_results['h'] = h
if compute_electrogram_flag == 1:
    simulation_results['electrogram_unipolar'] = electrogram_unipolar

# save the dictionary variable
np.savez(script_dir / 'result' / 'simulation_results.npz', **simulation_results)

#%%
# display result
# --------------------------------------------------
debug_plot = 1
if debug_plot == 1: 
    # show some action potentials and electrograms
    fig, axes = plt.subplots(
        nrows=3, ncols=2, figsize=(12, 8), sharex='col', sharey=False
    )

    for i, eid in enumerate(electrode_id):
        # left column: action potentials
        axes[i, 0].plot(action_potential[eid, :])
        axes[i, 0].set_title(f'Action Potential at Location {eid}')
        axes[i, 0].set_ylabel('Voltage (scaled)')
        axes[i, 0].set_xlabel('Time (ms)')

        # right column: unipolar electrograms
        axes[i, 1].plot(electrogram_unipolar[i, :])
        axes[i, 1].set_title(f'Unipolar Electrogram at Location {electrode_id[i]}')
        axes[i, 1].set_ylabel('Voltage (scaled)')
        axes[i, 1].set_xlabel('Time (ms)')

    plt.tight_layout()
    plt.savefig('result/AP_and_EGM.png', dpi=300)
    plt.show()

os._exit(0) # ensures the kernel dies. or even after the visual studio code is closed, there will still be heavy python process in CPU cause computer to heat up.

#%%