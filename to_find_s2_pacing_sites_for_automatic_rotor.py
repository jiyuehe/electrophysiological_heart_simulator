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

# plot the regions on voxels
codes.debug_display_of_s1s2_pacing_sites.execute(voxel, s1_pacing_voxel_id, s2_pacing_voxel_id)

# automatically find s2 pacing voxels
action_potential_s2_t = action_potential[:,s2_t]
id1 = np.where((action_potential_s2_t >= np.min(action_potential_s2)) & (action_potential_s2_t <= np.max(action_potential_s2)))[0]

h_s2_t = h[:,s2_t]
id2 = np.where((h_s2_t >= np.min(h_s2)) & (h_s2_t <= np.max(h_s2)))[0] # elements in id2 is more than id1

common_ids = np.intersect1d(id1, id2)

# plot the regions on voxels
codes.debug_display_of_s1s2_pacing_sites.execute(voxel, s1_pacing_voxel_id, common_ids)

# automatically find s2 pacing sites
ap_min = 0.002058
ap_max = 0.026245
h_min = 0.221531
h_max = 0.335100
id1 = np.where((action_potential_s2_t >= ap_min) & (action_potential_s2_t <= ap_max))[0]
id2 = np.where((h_s2_t >= h_min) & (h_s2_t <= h_max))[0]
s2_pacing_voxel_id_auto = np.intersect1d(id1, id2)

# plot the regions on voxels
codes.debug_display_of_s1s2_pacing_sites.execute(voxel, s1_pacing_voxel_id, s2_pacing_voxel_id_auto)

# check the automatic pacing sites
pacing_site = np.array([43508, 43512, 43513, 43514, 44055, 44056, 44057, 44060, 44061, 44062,
 44066, 44067, 44611, 44612, 44613, 44617, 44618, 44619, 44622, 44623,
 44624, 44627, 44628, 44633, 45175, 45176, 45177, 45181, 45182, 45183,
 45186, 45187, 45188, 45191, 45192, 45198, 45729, 45736, 45737, 45738,
 45742, 45743, 45744, 45748, 45749, 45750, 45754, 45755, 45756, 45757,
 45762, 45763, 46302, 46303, 46304, 46308, 46309, 46310, 46314, 46315,
 46316, 46320, 46321, 46322, 46323, 46327, 46328, 46329, 46884, 46885,
 46886, 46890, 46891, 46892, 46893, 46897, 46898, 46899, 46903, 46904,
 46905, 46906, 46910, 46911, 46912, 46917, 46918, 47465, 47471, 47472,
 47473, 47477, 47478, 47479, 47483, 47484, 47485, 47489, 47490, 47491,
 47497, 48056, 48057, 48063, 48064, 48065, 48069, 48070, 48071, 48075,
 48076, 48077, 48082, 48083, 48089, 48658, 48665, 48672, 48673, 48674,
 48678, 48679, 48680, 48685, 48686, 48692, 49313, 49314, 49325, 49326,
 49327, 49335, 49336, 49337, 49345, 50003, 50020, 50035, 50036, 50037,
 50053, 50762, 50763, 50782, 50783, 51486, 51510, 52209, 52252, 52982,
 53001, 53022, 53872, 53889, 53890, 53907, 53908, 54842, 54858, 54859,
 54874, 54875, 55877, 55903, 55904, 55943, 55944, 55945, 55983, 56991,
 57032, 57033, 57073, 57074, 57075, 57113, 58260, 58294, 58295])

codes.debug_display_of_s1s2_pacing_sites.execute(voxel, s1_pacing_voxel_id, pacing_site)