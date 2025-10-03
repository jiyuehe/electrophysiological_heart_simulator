# %%
# load libraries
# --------------------------------------------------
import codes
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go # pip install plotly. For 3D interactive plot: triangular mesh, and activation movie
import plotly.io as pio
pio.renderers.default = "browser" # simulation result mesh display in internet browser

script_dir = os.path.dirname(os.path.abspath(__file__)) # get the path of the current script
os.chdir(script_dir) # change the working directory

# %% 
# load the .mat data file
# --------------------------------------------------
data_path = script_dir + "/data/"
voxel, neighbor_id_2d, Delta, voxel_for_each_vertex, vertex_for_each_voxel, vertex, face, vertex_flag = codes.processing.prepare_geometry.execute(data_path)
voxel_flag = vertex_flag[vertex_for_each_voxel]

# %% 
# simulation parameters
# --------------------------------------------------
dt = 0.05 # ms. if dt is not small enough, simulation will result nan. Generally, if c <= 1.0, can use dt = 0.05
t_final = 1000 # ms. NOTE: need to be at least long enough to have two pacings, or cannot compute phase from action potential
pacing_start_time = 1 # ms
pacing_cycle_length = 250 # ms

# parameters of the heart model
n_voxel = voxel.shape[0] 

model_flag = 1 # 1: Mitchell-Schaeffer, 2: Aliev–Panfilov
if model_flag == 1: # Mitchell-Schaeffer
    parameter = {}
    parameter['tau_in_voxel'] = np.ones(n_voxel) * 0.3
    parameter['tau_out_voxel'] = np.ones(n_voxel) * 6
    parameter['tau_open_voxel'] = np.ones(n_voxel) * 120
    parameter['tau_close_voxel'] = np.ones(n_voxel) * 80
    c = 1 # diffusion coefficient. c = 1 is good for atrium
    parameter['c_voxel'] = c * np.ones(n_voxel)
    v_gate = 0.13
    parameter['v_gate_voxel'] = np.ones(n_voxel) * v_gate
elif model_flag == 2: # Aliev–Panfilov
    parameter = {}
    c = 0.1 # diffusion coefficient
    parameter['c_voxel'] = c * np.ones(n_voxel)
    v_gate = 0.13
    parameter['v_gate_voxel'] = np.ones(n_voxel) * v_gate
#%%
rotor_flag = 1 # 0: focal arrhythmia. 1: rotor arrhythmia via s1-s2 pacing

# %% 
# compute simulation
# --------------------------------------------------
electrode_id = [0, 5000, 10000] # electrode locations for computing electrograms

do_flag = 1 # 1: compute, 0: load existing result
if do_flag == 1:
    # fiber orientations
    D0 = codes.simulation.fibers.execute(n_voxel)

    # compute heart model equation parts
    P_2d = codes.compute_equation_parts.execute(n_voxel, D0, neighbor_id_2d, parameter, model_flag)

    # create the pacing signal
    pacing_signal = codes.create_pacing_signal.execute(dt, t_final, pacing_start_time, pacing_cycle_length, model_flag)

    # compute simulation
    action_potential, h = codes.compute_simulation.execute_CPU_parallel(neighbor_id_2d, voxel_flag, n_voxel, dt, t_final, pacing_signal, P_2d, Delta, model_flag, rotor_flag)
    np.save('result/action_potential.npy', action_potential)

    # compute unipolar electrogram    
    electrode_xyz = voxel[electrode_id, :]
    electrogram_unipolar = codes.compute_unipolar_electrogram.execute_CPU_parallel(electrode_xyz, voxel, D0, parameter['c_voxel'], action_potential, Delta, neighbor_id_2d)
    np.save('result/electrogram_unipolar.npy', electrogram_unipolar)

    # create phase from action potential
    action_potential_phase = np.zeros_like(action_potential)
    activation_phase = np.zeros_like(action_potential)
    for id in range(action_potential.shape[0]):
        do_flag = 1
        if do_flag == 1 and (id % (action_potential.shape[0]//5)) == 0:
            print(f'compute phase {id/action_potential.shape[0]*100:.1f}%')

        action_potential_phase[id,:], activation_phase[id,:] = codes.create_phase.execute(action_potential[id,:], v_gate)
    np.save('result/action_potential_phase.npy', action_potential_phase)
elif do_flag == 0:
    action_potential = np.load('result/action_potential.npy')
    electrogram_unipolar = np.load('result/electrogram_unipolar.npy')
    action_potential_phase = np.load('result/action_potential_phase.npy')

# compute local activation time map
action_potential_mesh = action_potential[voxel_for_each_vertex,:]
t_start = 1
cycle_length_percentage = 1 # use a number <1 when at a time instance, multiple cycles overlap
lat, cl = codes.calculate_local_activation_time.execute_on_action_potential(t_start, action_potential_mesh, v_gate, cycle_length_percentage)
# lat.shape = (number of mesh vertices,). 
# cl.shape = (n x number of mesh vertices, ) where n depends on how many cycle length is there in the action potential

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

    plt.show()

#%%
# display result
# --------------------------------------------------
debug_plot = 0
if debug_plot == 1:
    voxel_id = 2000
    ap = action_potential[voxel_id,:]
    ap_phase = action_potential_phase[voxel_id,:]
    plt.figure()
    plt.plot(ap, 'b')
    plt.plot(ap_phase, 'g')
    plt.show()

# %%
# activation activation movie display using plotly, display using a browser
# NOTE: if simulation is too long, it will not display, for example, 1000 ms simulation will display a blank page
do_flag = 0
if do_flag == 1:
    # activation activation movie display on mesh using plotly
    movie_data = action_potential_phase[voxel_for_each_vertex,:]
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
    codes.display_activation_movie.execute_on_mesh(vertex, face, map_color)

# activation movie display on volume using plotly, display using a browser
do_flag = 0
if do_flag == 1:
    movie_data = action_potential_phase[:,0:50] # if too large of data, will not display
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
    codes.display_activation_movie.execute_on_volume(voxel, map_color)

# activation phase movie display on vertex using matplotlib, with option to save as mp4
do_flag = 1
if do_flag == 1: 
    save_flag = 0 # 1: save movie as mp4. 0: do not save movie
    action_potential_phase_vertex = action_potential_phase[voxel_for_each_vertex,180:]
    codes.display_activation_movie.execute_on_vertex_save_as_mp4(save_flag, action_potential_phase_vertex, vertex)

# activation phase movie display on volume using matplotlib, with option to save as mp4
do_flag = 0
if do_flag == 1: 
    save_flag = 1 # 1: save movie as mp4. 0: do not save movie
    codes.display_activation_movie.execute_on_voxel_save_as_mp4(save_flag, action_potential_phase, voxel)

debug_plot = 0
if debug_plot == 1: # cycle length histogram
    plt.figure()
    plt.hist(cl, bins=10)
    plt.xlabel('cycle length, ms')
    plt.ylabel('counts')
    plt.title('cycle length histogram')
    plt.show()

debug_plot = 0
if debug_plot == 1: # local activation time map
    # convert local activation time into color
    data = lat
    data_min = np.min(data)
    data_max = np.max(data)
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
