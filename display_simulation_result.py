#%%
import os
from pathlib import Path
script_dir = os.path.dirname(os.path.abspath(__file__)) # get the path of the current script
os.chdir(script_dir) # change the working directory
script_dir = Path(script_dir)

import codes
import numpy as np

#%%
save_movie_flag = 0 # 1: save movie. 0: do not save movie
starting_time = 400 # 0 # ms
ending_time = 600 # ms
simulation_results_file_name = 'simulation_results.npz'

# load geometry data file
data_path = script_dir / 'data'
loaded = codes.load_geometry_data.execute(data_path)
geometry_data = {
    'voxel': loaded['voxel'],
    'neighbor_id_2d': loaded['neighbor_id_2d'],
    'Delta': loaded['Delta'],
    'voxel_for_each_vertex': loaded['voxel_for_each_vertex'],
    'vertex_for_each_voxel': loaded['vertex_for_each_voxel'],
    'vertex': loaded['vertex'],
    'face': loaded['face'],
}

vertex = geometry_data['vertex']
face = geometry_data['face']
voxel_for_each_vertex = geometry_data['voxel_for_each_vertex']

node = geometry_data['voxel']

# simulation data
result_dir = script_dir / 'result'
simulation_results = np.load(result_dir / simulation_results_file_name)
action_potential = simulation_results['action_potential']
t = np.arange(action_potential.shape[1])

start_id = np.argmin(np.abs(t - starting_time)) # find index of closest value
end_id = np.argmin(np.abs(t - ending_time)) # find index of closest value
t = t[start_id:end_id]
action_potential = action_potential[:,start_id:end_id]

v_gate = 0.13

# activation phase movie using matplotlib, with option to save as gif
do_flag = 1
if do_flag == 1: 
    movie_data = action_potential
    geometry_flag = 2 # 3D atrium
    codes.display_activation_movie.execute_on_voxel_save_as_mp4(save_movie_flag, movie_data, node, t, geometry_flag)

print('done')