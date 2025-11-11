import codes
import numpy as np
from numba import njit, prange

# CPU paralleled computation
# --------------------------------------------------
@njit(parallel=True)
def compute_voxel(u_current, h_current, P_2d, neighbor_id_2d_2, J_stim, dt, Delta):
    n_voxel = u_current.shape[0]
    u_next = np.empty_like(u_current)
    h_next = np.empty_like(h_current)
    diffusion_term = np.zeros(n_voxel)
    
    for n in prange(n_voxel): # parallel loop
        # compute diffusion term
        diffusion_term[n] = P_2d[n, 20] / (4*Delta**2) * \
        ( \
            P_2d[n, 0] * (u_current[neighbor_id_2d_2[n, 0]] - u_current[n]) + \
            P_2d[n, 1] * (u_current[neighbor_id_2d_2[n, 1]] - u_current[n]) + \
            P_2d[n, 2] * (u_current[neighbor_id_2d_2[n, 2]] - u_current[n]) + \
            P_2d[n, 3] * (u_current[neighbor_id_2d_2[n, 3]] - u_current[n]) + \
            P_2d[n, 4] * (u_current[neighbor_id_2d_2[n, 4]] - u_current[n]) + \
            P_2d[n, 5] * (u_current[neighbor_id_2d_2[n, 5]] - u_current[n]) + \
            P_2d[n, 6] * (u_current[neighbor_id_2d_2[n, 0]] - u_current[neighbor_id_2d_2[n, 1]]) + \
            P_2d[n, 7] * (u_current[neighbor_id_2d_2[n, 2]] - u_current[neighbor_id_2d_2[n, 3]]) + \
            P_2d[n, 8] * (u_current[neighbor_id_2d_2[n, 4]] - u_current[neighbor_id_2d_2[n, 5]]) + \
            P_2d[n, 9] * (u_current[neighbor_id_2d_2[n, 6]] - u_current[neighbor_id_2d_2[n, 8]]) + \
            P_2d[n, 10] * (u_current[neighbor_id_2d_2[n, 9]] - u_current[neighbor_id_2d_2[n, 7]]) + \
            P_2d[n, 11] * (u_current[neighbor_id_2d_2[n, 14]] - u_current[neighbor_id_2d_2[n, 16]]) + \
            P_2d[n, 12] * (u_current[neighbor_id_2d_2[n, 17]] - u_current[neighbor_id_2d_2[n, 15]]) + \
            P_2d[n, 13] * (u_current[neighbor_id_2d_2[n, 10]] - u_current[neighbor_id_2d_2[n, 12]]) + \
            P_2d[n, 14] * (u_current[neighbor_id_2d_2[n, 13]] - u_current[neighbor_id_2d_2[n, 11]]) \
        )
        
        # compute the next time step value of u
        u_next[n] = u_current[n] + dt * \
        ( \
            (h_current[n] * (u_current[n]**2) * (1 - u_current[n])) / P_2d[n, 17] - \
            u_current[n] / P_2d[n, 18] + \
            J_stim[n] + \
            diffusion_term[n] \
        )

        # compute the next time step value of h
        h_next_1 = ((1 - h_current[n]) / P_2d[n, 15]) * dt + h_current[n]
        h_next_2 = (-h_current[n] / P_2d[n, 16]) * dt + h_current[n]
        
        if u_current[n] < P_2d[n, 19]:
            h_next[n] = h_next_1
        elif u_current[n] >= P_2d[n, 19]:
            h_next[n] = h_next_2

    return u_next, h_next

def execute_CPU_parallel(neighbor_id_2d, n_voxel, dt, t_final, P_2d, Delta, arrhythmia_flag, arrhythmia_parameters):
    # pacing parameters
    s1_pacing_voxel_id = arrhythmia_parameters["s1_pacing_voxel_id"] 
    s1_t = arrhythmia_parameters["s1_t"] 
    s2_t = s1_t + arrhythmia_parameters["s1_s2_delta_t"] 
    ap_min = arrhythmia_parameters["ap_min"] 
    ap_max = arrhythmia_parameters["ap_max"] 
    h_min = arrhythmia_parameters["h_min"] 
    h_max = arrhythmia_parameters["h_max"] 
    s2_region_size_factor = arrhythmia_parameters["s2_region_size_factor"] 

    # s1 pacing location
    neighbor_id = neighbor_id_2d[s1_pacing_voxel_id, :] # add all the neighbors of the pacing voxel to be paced
    neighbor_id = neighbor_id[neighbor_id != -1] # remove the -1s, which means no neighbors
    if np.isscalar(s1_pacing_voxel_id): # if s1_pacing_voxel_id is just a number
        s1_pacing_voxel_id = np.array([s1_pacing_voxel_id]) # np.concatenate will not work with number, that's why convert it to 1d array
    s1_pacing_voxel_id = np.concatenate([s1_pacing_voxel_id, neighbor_id])
    s1_pacing_voxel_id = np.unique(s1_pacing_voxel_id)

    # set initial value at rest
    u_current = np.zeros(n_voxel)
    h_current = np.ones(n_voxel)

    # initialize pacing stimulus
    J_stim = np.zeros(n_voxel)

    sim_u_voxel = np.zeros((n_voxel, t_final)) # sampling frequency at 1 kHz
    sim_h_voxel = np.zeros((n_voxel, t_final)) # sampling frequency at 1 kHz

    neighbor_id_2d_2 = neighbor_id_2d.copy() # NOTE: without .copy(), changes of neighbor_id_2d_2 will also change neighbor_id_2d
    neighbor_id_2d_2[neighbor_id_2d_2 == -1] = 0 # change -1 to 0, so that it can be used as index
        # this does not matter because the corresponding P_2d will be 0, and their product will be 0
        # so these terms will be eliminated anyway

    J_stim_magnitude = 1
    pacing_duration = 5 # ms
    total_model_time_steps = int(t_final/dt) # number of simulation time steps
    id_save = 0 # simulation time step is small, do not save all of them, save at 1 kHz as the catheter electrodes
    for model_time_step in range(total_model_time_steps): 
        if ((model_time_step+1) % (total_model_time_steps//5)) == 0:
            print(f'simulating {(model_time_step+1)/total_model_time_steps*100:.1f}%')
        
        model_time = model_time_step * dt

        # s1 pacing
        J_stim.fill(0.0) # reset pacing stimulus values to 0s
        if arrhythmia_flag == 0: # focal arrhythmia, s1 pace according to cycle length setting
            t = model_time
            while t - arrhythmia_parameters['pacing_start_time'] > arrhythmia_parameters['pacing_cycle_length']:
                t = t - arrhythmia_parameters['pacing_cycle_length']

            if t >= arrhythmia_parameters['pacing_start_time'] and t <= arrhythmia_parameters['pacing_start_time'] + pacing_duration:
                # print('s1 pacing')
                J_stim[s1_pacing_voxel_id] = J_stim_magnitude

        elif arrhythmia_flag != 0: # not focal arrhythmia, s1 pace only once
            if model_time >= s1_t and model_time <= s1_t + pacing_duration:
                # print('s1 pacing')
                J_stim[s1_pacing_voxel_id] = J_stim_magnitude

        # s2 pacing
        if arrhythmia_flag != 0 and model_time >= s2_t and model_time <= s2_t + pacing_duration: 
            action_potential_s2_t = sim_u_voxel[:,int(s2_t)-1] # -1: the current values are not saved yet, so check the previous time frame
            h_s2_t = sim_h_voxel[:,int(s2_t)-1] # -1: the current values are not saved yet, so check the previous time frame

            id1 = np.where((action_potential_s2_t >= ap_min) & (action_potential_s2_t <= ap_max))[0]
            id2 = np.where((h_s2_t >= h_min) & (h_s2_t <= h_max))[0]
            s2_pacing_voxel_id_auto = np.intersect1d(id1, id2) # these voxels could have a ring-like shape, which cannot generate rotor

            # grab a portion of the shape, so it becomes like a curvy patch (instead of a ring), allow waves to rotate at the edges of the patch
            id = s2_pacing_voxel_id_auto[0] # find one voxel to start, can be any random one
            iter = 0
            while (id.size < s2_pacing_voxel_id_auto.size * s2_region_size_factor or id.size < 1000) and iter <= 20: # repeat several times to include more neighbors
                # NOTE: iter <= 10 is to prevent inifinte while loop that sometimes will happen
                neighbor_id = neighbor_id_2d[id, :] # the neighbors
                neighbor_id = neighbor_id[neighbor_id != -1] # remove the -1s, which means no neighbors
                id = np.concatenate([np.atleast_1d(id), np.atleast_1d(neighbor_id)]) # add the neighbors
                id = np.intersect1d(id, s2_pacing_voxel_id_auto) # make sure its within the original shape
                iter = iter + 1
            s2_pacing_voxel_id = id
            J_stim[s2_pacing_voxel_id] = J_stim_magnitude

        u_next, h_next = compute_voxel(u_current, h_current, P_2d, neighbor_id_2d_2, J_stim, dt, Delta)
        
        # update value
        u_current = u_next
        h_current = h_next
        
        # save value at 1 kHz
        number_of_steps_per_ms = int(1/dt)
        if (model_time_step % number_of_steps_per_ms) == 0:
            sim_u_voxel[:, id_save] = u_current
            sim_h_voxel[:, id_save] = h_current
            id_save = id_save + 1

    return sim_u_voxel, sim_h_voxel
