#%%
import os
from pathlib import Path
script_dir = os.path.dirname(os.path.abspath(__file__)) # get the path of the current script
os.chdir(script_dir) # change the working directory
script_dir = Path(script_dir)

import codes
import numpy as np # pip install numpy

#%%
save_movie_flag = 0 # 1: save movie. 0: do not save movie
starting_time = 0 # 0 # ms
ending_time = 500 # ms
geometry_flag = 0 # 0: 2D sheet, 1: 3D slab, 2: patient 3D atrium
simulation_results_file_name = 'simulation_results_0.npz'

# load data
if geometry_flag == 0: # 2D sheet
    file_name = script_dir.parent / 'data' / 'sheet.obj'
elif geometry_flag == 1: # 3D slab
    file_name = script_dir.parent / 'data' / 'slab.obj'
elif geometry_flag == 2: # patient 3D atrium
    file_name = script_dir.parent / 'data' / '49_2-LA_edited.obj'
elif geometry_flag == 3: # long slab for computing conduction velocity
    file_name = script_dir.parent / 'data' / 'long_slab.obj'

loaded = np.load(str(file_name)[0:-4] + '.npz')
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

if geometry_flag == 2:
    node = geometry_data['voxel'][voxel_for_each_vertex,:]
elif geometry_flag in [0, 1, 3]:
    node = geometry_data['voxel']

# simulation data
result_dir = script_dir.parent / 'result'
simulation_results = np.load(result_dir / simulation_results_file_name)
action_potential = simulation_results['action_potential']
t = simulation_results['physical_time']

start_id = np.argmin(np.abs(t - starting_time)) # find index of closest value
end_id = np.argmin(np.abs(t - ending_time)) # find index of closest value
t = t[start_id:end_id]
action_potential = action_potential[:,start_id:end_id]

v_gate = 0.13

#%%
# activation phase movie using matplotlib, with option to save as mp4
do_flag = 1
if do_flag == 1: 
    if str(file_name)[-11:-4] == '_edited':
        movie_data = action_potential[voxel_for_each_vertex, :] # display on vertices
    elif str(file_name)[-8:-4] == 'slab' or str(file_name)[-9:-4] == 'sheet':
        movie_data = action_potential
    
    codes.display_activation_movie.execute_on_voxel_save_as_mp4(save_movie_flag, movie_data, node, t, geometry_flag)

print('done')
