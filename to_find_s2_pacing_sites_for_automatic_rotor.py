#%%
import codes
import os
import numpy as np
import matplotlib.pyplot as plt

#%%
# NOTE: 
# use ui_select_vertices.py to manually find out the s2 pacing sites that will generate a rotor
# to find out u and h values of the s2 pacing sites, RUN ONLY the s1 pacing simulation and save the u and h
# because if use the s1s2 u and h, the u already has the stimulus added to the s2 pacing sites, so the u and h vaules are not correct

# load simulation 
script_dir = os.path.dirname(os.path.abspath(__file__)) # get the path of the current script
os.chdir(script_dir) # change the working directory

data_path = script_dir + "/data/"
voxel, neighbor_id_2d, Delta, voxel_for_each_vertex, vertex_for_each_voxel, vertex, face, vertex_flag = codes.processing.prepare_geometry.execute(data_path)
voxel_flag = vertex_flag[vertex_for_each_voxel]

action_potential = np.load('result/action_potential.npy')
h = np.load('result/h.npy')

# the manually assigned pacing sites
s1_pacing_voxel_id = np.where(voxel_flag == 1)[0]
s2_pacing_voxel_id = np.where(voxel_flag == 2)[0]

neighbor_id = neighbor_id_2d[s1_pacing_voxel_id, :] # add all the neighbors of the pacing voxel to be paced
neighbor_id = neighbor_id[neighbor_id != -1] # remove the -1s, which means no neighbors
s1_pacing_voxel_id = np.concatenate([s1_pacing_voxel_id, neighbor_id])
s1_pacing_voxel_id = np.unique(s1_pacing_voxel_id)

neighbor_id = neighbor_id_2d[s2_pacing_voxel_id, :] # add all the neighbors of the pacing voxel to be paced
neighbor_id = neighbor_id[neighbor_id != -1] # remove the -1s, which means no neighbors
s2_pacing_voxel_id = np.concatenate([s2_pacing_voxel_id, neighbor_id])
s2_pacing_voxel_id = np.unique(s2_pacing_voxel_id)

s2_t = 205

action_potential_s2 = action_potential[s2_pacing_voxel_id,s2_t]
h_s2 = h[s2_pacing_voxel_id,s2_t]

print(
    f"action potential min max: {np.min(action_potential_s2)} {np.max(action_potential_s2)}\n"
    f"h min max: {np.min(h_s2)} {np.max(h_s2)}"
)

# plot the values of action_potential_s2 and h_s2
plt.figure()
plt.plot(action_potential_s2,'b')
plt.plot(h_s2,'g')
plt.xlabel('voxels')
plt.ylabel('')
plt.title('b:action potential, g:h')

# plot the manually assigned pacing sites
codes.debug_display_of_s1s2_pacing_sites.execute(voxel, s1_pacing_voxel_id, s2_pacing_voxel_id)

#%%
# automatically find s2 pacing voxels
action_potential_s2_t = action_potential[:,s2_t]
id1 = np.where((action_potential_s2_t >= np.min(action_potential_s2)) & (action_potential_s2_t <= np.max(action_potential_s2)))[0]

h_s2_t = h[:,s2_t]
id2 = np.where((h_s2_t >= np.min(h_s2)) & (h_s2_t <= np.max(h_s2)))[0] # elements in id2 is more than id1

common_ids = np.intersect1d(id1, id2)

# plot the automatically assigned pacing sites
codes.debug_display_of_s1s2_pacing_sites.execute(voxel, s1_pacing_voxel_id, common_ids)

#%%
# automatically find s2 pacing sites
ap_min = 0.01
ap_max = 0.02
h_min = 0.23
h_max = 0.30
id1 = np.where((action_potential_s2_t >= ap_min) & (action_potential_s2_t <= ap_max))[0]
id2 = np.where((h_s2_t >= h_min) & (h_s2_t <= h_max))[0]
s2_pacing_voxel_id_auto = np.intersect1d(id1, id2)

# plot the automatically assigned pacing sites
codes.debug_display_of_s1s2_pacing_sites.execute(voxel, s1_pacing_voxel_id, s2_pacing_voxel_id_auto)

#%%
# check the automatic pacing sites
pacing_site = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

codes.debug_display_of_s1s2_pacing_sites.execute(voxel, s1_pacing_voxel_id, pacing_site)
