# %%
import codes
import os
from pathlib import Path
import numpy as np # pip install numpy
import matplotlib.pyplot as plt # pip install matplotlib
import plotly.graph_objects as go # pip install plotly. For 3D interactive plot: triangular mesh, and activation movie
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
t_final = 800 # ms. NOTE: need to be at least long enough to have two pacings (pacing_cycle_length), or cannot compute activation phase
pacing_start_time = 0 # ms
pacing_cycle_length = 250 # ms
rotor_flag = 1 # 0: focal arrhythmia. 1: rotor arrhythmia via s1-s2 pacing
compute_electrogram_flag = 0 # 1: compute electrogram. 0: do not compute electrogram

# Mitchell-Schaeffer heart model parameters
c = 1 # diffusion coefficient
v_gate = 0.13
heart_model_parameter = {
    'tau_in_voxel': np.ones(n_voxel) * 0.3,
    'tau_out_voxel': np.ones(n_voxel) * 6,
    'tau_open_voxel': np.ones(n_voxel) * 120,
    'tau_close_voxel': np.ones(n_voxel) * 80,
    'c_voxel': c * np.ones(n_voxel),
    'v_gate_voxel': np.ones(n_voxel) * v_gate
}

# %% 
# compute simulation
# --------------------------------------------------
electrode_id = [0, 5000, 10000] # electrode locations for computing electrograms

# fiber orientations
D0 = codes.simulation.fibers.execute(n_voxel)

# compute heart model equation parts
P_2d = codes.compute_equation_parts.execute(n_voxel, D0, neighbor_id_2d, heart_model_parameter)

# rotor arrhythmia parameters
rotor_parameters = {
    "s1_pacing_voxel_id": 23403, # location of s1 pacing site
    "s1_t": 0, # ms. time of s1 pacing
    "s1_s2_delta_t": 205 / dt, # ms. time interval between s1 and s2
    "ap_min": 0.004, # a threshold value of action potential 
    "ap_max": 0.030, # a threshold value of action potential 
    "h_min": 0.200, # a threshold value of gating variable
    "h_max": 0.300, # a threshold value of gating variable
    "s2_region_size_factor": 0.5 # a less than 1 multiplication factor to reduce s2 pacing region size
}

# compute simulation
action_potential, h = codes.compute_simulation.execute_CPU_parallel(neighbor_id_2d, voxel_flag, n_voxel, dt, t_final, P_2d, Delta, model_flag, rotor_flag, rotor_parameters)
np.save('result/action_potential.npy', action_potential)
np.save('result/h.npy', h)

# compute unipolar electrogram
if compute_electrogram_flag == 1:
    electrode_xyz = voxel[electrode_id, :]
    electrogram_unipolar = codes.compute_unipolar_electrogram.execute_CPU_parallel(electrode_xyz, voxel, D0, heart_model_parameter['c_voxel'], action_potential, Delta, neighbor_id_2d)
    np.save('result/electrogram_unipolar.npy', electrogram_unipolar)

# create phase from action potential
action_potential_phase = np.zeros_like(action_potential)
activation_phase = np.zeros_like(action_potential)
for id in range(action_potential.shape[0]):
    if ((id+1) % (action_potential.shape[0]//5)) == 0:
        print(f'compute phase {(id+1)/action_potential.shape[0]*100:.1f}%')
    action_potential_phase[id,:], activation_phase[id,:] = codes.create_phase.execute(action_potential[id,:], v_gate)
np.save('result/action_potential_phase.npy', action_potential_phase)

#%%
# display result
# --------------------------------------------------
debug_plot = 0
if debug_plot == 1:
    # action potential
    plt.figure()
    plt.plot(action_potential[electrode_id, :].T)
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (scaled)')
    plt.title('examples of simulated action potential')
    plt.savefig('result/action_potential.png')

    # unipolar electrogram
    plt.figure()
    # e_id = 0
    # plt.plot(electrogram_unipolar[e_id,:], 'b')
    plt.plot(electrogram_unipolar.T)
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (scaled)')
    plt.title('examples of simulated unipolar electrogram')
    plt.savefig('result/unipolar_electrogram.png')

    # action potential phase, for activation movie display
    voxel_id = 2000
    ap = action_potential[voxel_id,:]
    ap_phase = action_potential_phase[voxel_id,:]
    plt.figure()
    plt.plot(ap, 'b')
    plt.plot(ap_phase, 'g')

    plt.show()

# activation phase movie using matplotlib, with option to save as mp4
do_flag = 1
if do_flag == 1: 
    save_flag = 0 # 1: save movie as mp4. 0: do not save movie
    starting_time = 190 # 0 # ms
    ending_time = 400 # t_final # ms
    movie_data = action_potential_phase[voxel_for_each_vertex, starting_time:ending_time] # display on vertices
    # movie_data = action_potential_phase[:, starting_time:] # display on voxels
    codes.display_activation_movie.execute_on_voxel_save_as_mp4(save_flag, movie_data, vertex)

# activation activation movie display using plotly, display using a browser, has a time frame slider
# NOTE: if data is too large, it will not display. for example, 1000 ms simulation will display only a blank page
do_flag = 0
if do_flag == 1:
    option = 1 # 1: display on mesh. 2: display on voxle

    if option == 1:
        movie_data = action_potential_phase[voxel_for_each_vertex,:] # grab vertex data
    elif option == 2:
        time_range = range(0,50) # range(start, stop) generates numbers from start up to (but not including) stop.
        movie_data = action_potential_phase[:,time_range] # if data is too large, will not display, thus trim the data
    
    data_min = np.min(movie_data)
    data_max = np.max(movie_data)
    data_threshold = data_min
    map_color = {}
    n_time = movie_data.shape[1]
    for n in range(n_time):
        if (n % (n_time//5)) == 0:
            print(f'compute color map {n/n_time*100:.1f}%')
        data = movie_data[:, n]
        color = codes.convert_data_to_color.execute(data, data_min, data_max, data_threshold)
        map_color[n] = color

    if option == 1:
        codes.display_activation_movie.execute_on_mesh(vertex, face, map_color)
    elif option == 2:
        codes.display_activation_movie.execute_on_volume(voxel, map_color)

debug_plot = 0
if debug_plot == 1: # local activation time map
    # compute local activation time map
    action_potential_mesh = action_potential[voxel_for_each_vertex,:]
    t_start = 380
    cycle_length_percentage = 1 # use a number <1 when at a time instance, multiple cycles overlap
    lat, cl = codes.calculate_local_activation_time.execute_on_action_potential(t_start, action_potential_mesh, v_gate, cycle_length_percentage)
    # lat.shape = (number of mesh vertices,). 
    # cl.shape = (n x number of mesh vertices, ) where n depends on how many cycle length is there in the action potential

    debug_plot = 0
    if debug_plot == 1: # cycle length histogram
        plt.figure()
        plt.hist(cl, bins=10)
        plt.xlabel('cycle length, ms')
        plt.ylabel('counts')
        plt.title('cycle length histogram')
        plt.show()

    # convert local activation time into color
    data = lat
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    data_threshold = data_min-0.1 # a little small than data_min, so that places with value of data_min will have color
    color = codes.convert_data_to_color.execute(data, data_min, data_max, data_threshold)
    fig = go.Figure(
        data = [
            go.Mesh3d(
                x = vertex[:, 0], y = vertex[:, 1], z = vertex[:, 2],
                i = face[:, 0], j = face[:, 1], k = face[:, 2],
                vertexcolor = color)
        ]
    )
    fig.show()

os._exit(0) # ensures the kernel dies. or even after the visual studio code is closed, there will still be heavy python process in CPU cause computer to heat up.
#%%