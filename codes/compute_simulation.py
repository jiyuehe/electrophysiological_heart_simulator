import codes
import numpy as np
from numba import njit, prange # pip install numba

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

def execute_CPU_parallel(neighbor_id_2d, n_voxel, dt, t_final, P_2d, Delta, rotor_flag, rotor_parameters):
    # pacing parameters
    s1_pacing_voxel_id = rotor_parameters["s1_pacing_voxel_id"] 
    s1_t = rotor_parameters["s1_t"] 
    s2_t = s1_t + rotor_parameters["s1_s2_delta_t"] 
    ap_min = rotor_parameters["ap_min"] 
    ap_max = rotor_parameters["ap_max"] 
    h_min = rotor_parameters["h_min"] 
    h_max = rotor_parameters["h_max"] 
    s2_region_size_factor = rotor_parameters["s2_region_size_factor"] 

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

    T = int(t_final/dt) # number of simulation time steps
    id_save = 0 # simulation time step is small, do not save all of them, save at 1 kHz as most surgical hardware (catheter electrodes)

    sim_u_voxel = np.zeros((n_voxel, t_final)) # sampling frequency at 1 kHz
    sim_h_voxel = np.zeros((n_voxel, t_final)) # sampling frequency at 1 kHz

    neighbor_id_2d_2 = neighbor_id_2d.copy() # NOTE: without .copy(), changes of neighbor_id_2d_2 will also change neighbor_id_2d
    neighbor_id_2d_2[neighbor_id_2d_2 == -1] = 0 # change -1 to 0, so that it can be used as index
        # this does not matter because the corresponding P_2d will be 0, and their product will be 0
        # so these terms will be eliminated anyway

    for t in range(T): 
        if ((t+1) % (T//5)) == 0:
            print(f'simulating {(t+1)/T*100:.1f}%')

        J_stim.fill(0.0) # reset pacing stimulus values to 0s
        
        # s1 pacing
        if t >= s1_t and t <= s1_t + 10/dt: # 10/dt is 10 ms of pacing duration
            J_stim[s1_pacing_voxel_id] = 20

        # s2 pacing
        if rotor_flag == 1 and t >= s2_t and t <= s2_t + 10 / dt: # 10/dt is 10 ms of pacing duration
            action_potential_s2_t = sim_u_voxel[:,int(s2_t*dt)-1] # -1: the current values are not saved yet, so check the previous time frame
            h_s2_t = sim_h_voxel[:,int(s2_t*dt)-1] # -1: the current values are not saved yet, so check the previous time frame

            id1 = np.where((action_potential_s2_t >= ap_min) & (action_potential_s2_t <= ap_max))[0]
            id2 = np.where((h_s2_t >= h_min) & (h_s2_t <= h_max))[0]
            s2_pacing_voxel_id_auto = np.intersect1d(id1, id2) # these voxels could have a ring-like shape, which cannot generate rotor

            # grab a portion of the shape, so it becomes like a curvy patch (instead of a ring), allow waves to rotate at the edges of the patch
            id = s2_pacing_voxel_id_auto[0] # find one voxel to start, can be any random one
            while id.size < s2_pacing_voxel_id_auto.size * s2_region_size_factor: # repeat several times to include more neighbors
                neighbor_id = neighbor_id_2d[id, :] # the neighbors
                neighbor_id = neighbor_id[neighbor_id != -1] # remove the -1s, which means no neighbors
                id = np.concatenate([np.atleast_1d(id), np.atleast_1d(neighbor_id)]) # add the neighbors
                id = np.intersect1d(id, s2_pacing_voxel_id_auto) # make sure its within the original shape

            s2_pacing_voxel_id = id
            J_stim[s2_pacing_voxel_id] = 20

        fibrillation_flag = 1
        if fibrillation_flag == 1 and t == 400 / dt:
            # update heart model parameter in the middle of rotor arrhythmia to create fibrillations
            heart_model_parameter = {
                'tau_in_voxel': np.ones(n_voxel) * 0.3, # tau_in
                'tau_out_voxel': np.ones(n_voxel) * 6, # tau_out
                'tau_open_voxel': np.ones(n_voxel) * 100, # tau_open
                'tau_close_voxel': np.ones(n_voxel) * 100, # tau_close
            }
            P_2d[:, 15] = heart_model_parameter['tau_open_voxel']
            P_2d[:, 16] = heart_model_parameter['tau_close_voxel']
            P_2d[:, 17] = heart_model_parameter['tau_in_voxel']
            P_2d[:, 18] = heart_model_parameter['tau_out_voxel']

        u_next, h_next = compute_voxel(u_current, h_current, P_2d, neighbor_id_2d_2, J_stim, dt, Delta)
        
        # update value
        u_current = u_next
        h_current = h_next
        
        # save value at 1 kHz
        if (t % int(1/dt)) == 0:
            sim_u_voxel[:, id_save] = u_current
            sim_h_voxel[:, id_save] = h_current
            id_save = id_save + 1

    return sim_u_voxel, sim_h_voxel

'''
# vectorized computation
# --------------------------------------------------
def execute_vectorized(neighbor_id_2d, pacing_voxel_id, n_voxel, dt, t_final, pacing_signal, P_2d, Delta, model_flag):
    neighbor_id = neighbor_id_2d[pacing_voxel_id, :] # add all the neighbors of the pacing voxel to be paced
    pacing_voxel_id = neighbor_id[neighbor_id != -1] # remove the -1s, which means no neighbors

    # set initial value at rest
    if model_flag == 1:
        u_current = np.zeros(n_voxel)
        h_current = np.ones(n_voxel)
    elif model_flag == 2:
        u_current = np.zeros(n_voxel)
        h_current = np.zeros(n_voxel)

    u_next = np.empty_like(u_current)
    h_next = np.empty_like(h_current)
    J_stim = np.zeros(n_voxel)
    diffusion_term = np.zeros(n_voxel)

    T = int(t_final/dt) # number of simulation time steps
    id_save = 0

    sim_u_voxel = np.zeros((n_voxel, t_final)) # sampling frequency at 1 kHz
    sim_h_voxel = np.zeros((n_voxel, t_final)) # sampling frequency at 1 kHz

    neighbor_id_2d_2 = neighbor_id_2d.copy() # NOTE: without .copy(), changes of neighbor_id_2d_2 will also change neighbor_id_2d
    neighbor_id_2d_2[neighbor_id_2d_2 == -1] = 0 # change -1 to 0, so that it can be used as index
        # this does not matter because the corresponding P_2d will be 0, 
        # so these terms will be eliminated anyway

    for t in range(T): 
        do_flag = 1
        if do_flag == 1 and (t % (T//10)) == 0:
            print(f'simulating {t/T*100:.1f}%')
        
        J_stim.fill(0.0) # reset values to 0s
        J_stim[pacing_voxel_id] = pacing_signal[t]

        # compute diffusion term
        diffusion_term = P_2d[:, 20] / (4*Delta**2) * \
        ( \
            P_2d[:, 0] * (u_current[neighbor_id_2d_2[:, 0]] - u_current) + \
            P_2d[:, 1] * (u_current[neighbor_id_2d_2[:, 1]] - u_current) + \
            P_2d[:, 2] * (u_current[neighbor_id_2d_2[:, 2]] - u_current) + \
            P_2d[:, 3] * (u_current[neighbor_id_2d_2[:, 3]] - u_current) + \
            P_2d[:, 4] * (u_current[neighbor_id_2d_2[:, 4]] - u_current) + \
            P_2d[:, 5] * (u_current[neighbor_id_2d_2[:, 5]] - u_current) + \
            P_2d[:, 6] * (u_current[neighbor_id_2d_2[:, 0]] - u_current[neighbor_id_2d_2[:, 1]]) + \
            P_2d[:, 7] * (u_current[neighbor_id_2d_2[:, 2]] - u_current[neighbor_id_2d_2[:, 3]]) + \
            P_2d[:, 8] * (u_current[neighbor_id_2d_2[:, 4]] - u_current[neighbor_id_2d_2[:, 5]]) + \
            P_2d[:, 9] * (u_current[neighbor_id_2d_2[:, 6]] - u_current[neighbor_id_2d_2[:, 8]]) + \
            P_2d[:, 10] * (u_current[neighbor_id_2d_2[:, 9]] - u_current[neighbor_id_2d_2[:, 7]]) + \
            P_2d[:, 11] * (u_current[neighbor_id_2d_2[:, 14]] - u_current[neighbor_id_2d_2[:, 16]]) + \
            P_2d[:, 12] * (u_current[neighbor_id_2d_2[:, 17]] - u_current[neighbor_id_2d_2[:, 15]]) + \
            P_2d[:, 13] * (u_current[neighbor_id_2d_2[:, 10]] - u_current[neighbor_id_2d_2[:, 12]]) + \
            P_2d[:, 14] * (u_current[neighbor_id_2d_2[:, 13]] - u_current[neighbor_id_2d_2[:, 11]]) \
        )
        diffusion_term = diffusion_term.flatten()
        
        # compute the next time step value of u        
        if model_flag == 1:
            u_next = u_current + dt * \
            ( \
                (h_current * (u_current**2) * (1 - u_current)) / P_2d[:, 17] - \
                u_current / P_2d[:, 18] + \
                J_stim + \
                diffusion_term \
            )
            
            # compute the next time step value of h
            h_next_1 = ((1 - h_current) / P_2d[:, 15]) * dt + h_current
            h_next_2 = (-h_current / P_2d[:, 16]) * dt + h_current
            
            id_1 = u_current < P_2d[:, 19]
            id_2 = u_current >= P_2d[:, 19]
            
            h_next[id_1] = h_next_1[id_1]
            h_next[id_2] = h_next_2[id_2]
        elif model_flag == 2:
            k = 1
            a = 0.15
            mu_1 = 0.2
            mu_2 = 0.3
            epsilon_0 = 0.002

            u_next = diffusion_term - \
                k * u_current * (u_current - a) * (u_current - 1) - \
                u_current * h_current + \
                J_stim
            h_next = (epsilon_0 + mu_1 * h_current / (u_current + mu_2)) * \
                (-h_current - k * u_current * (u_current - a - 1))
            
            if t>10:
                print(t)
                print(u_next)

        # update value
        u_current = u_next
        h_current = h_next
        
        # save value at 1 kHz
        if (t % int(1/dt)) == 0:
            sim_u_voxel[:, id_save] = u_current
            sim_h_voxel[:, id_save] = h_current
            id_save = id_save + 1

    return sim_u_voxel, sim_h_voxel
'''