import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

# CPU paralleled computation
# --------------------------------------------------
@njit(parallel=True)
def compute_voxel(n_voxel, n_electrode, D11_b, D12_b, D13_b, D21_b, D22_b, D23_b, D31_b, D32_b, D33_b, c_voxel, l, l_x, l_y, l_z, dvdx_b, dvdy_b, dvdz_b): 
    egm_part = np.zeros((n_voxel, n_electrode))
    for n in prange(n_voxel):
        egm_part[n,:] = c_voxel[n] / (l[n,:]**3) * (
            (D11_b[n,:] * dvdx_b[n,:] + D12_b[n,:] * dvdy_b[n,:] + D13_b[n,:] * dvdz_b[n,:]) * l_x[n,:] +
            (D21_b[n,:] * dvdx_b[n,:] + D22_b[n,:] * dvdy_b[n,:] + D23_b[n,:] * dvdz_b[n,:]) * l_y[n,:] +
            (D31_b[n,:] * dvdx_b[n,:] + D32_b[n,:] * dvdy_b[n,:] + D33_b[n,:] * dvdz_b[n,:]) * l_z[n,:]
        )

    return egm_part
        
def execute_CPU_parallel(electrode_xyz, voxel, D0, c_voxel, action_potential, Delta, neighbor_id_2d):
    n_voxel = voxel.shape[0]
    
    D11 = np.zeros(n_voxel)
    D12 = np.zeros(n_voxel)
    D13 = np.zeros(n_voxel)
    D21 = np.zeros(n_voxel)
    D22 = np.zeros(n_voxel)
    D23 = np.zeros(n_voxel)
    D31 = np.zeros(n_voxel)
    D32 = np.zeros(n_voxel)
    D33 = np.zeros(n_voxel)
    for n in range(n_voxel):
        D11[n] = D0[n][0, 0]
        D12[n] = D0[n][0, 1]
        D13[n] = D0[n][0, 2]
        D21[n] = D0[n][1, 0]
        D22[n] = D0[n][1, 1]
        D23[n] = D0[n][1, 2]
        D31[n] = D0[n][2, 0]
        D32[n] = D0[n][2, 1]
        D33[n] = D0[n][2, 2]
    
    # distance from electrode to nodes
    n_electrode = electrode_xyz.shape[0]
    l = np.zeros((n_voxel, n_electrode))
    l_x = np.zeros((n_voxel, n_electrode))
    l_y = np.zeros((n_voxel, n_electrode))
    l_z = np.zeros((n_voxel, n_electrode))
    for e_id in range(n_electrode):
        l_x[:, e_id] = voxel[:, 0] - electrode_xyz[e_id, 0]
        l_y[:, e_id] = voxel[:, 1] - electrode_xyz[e_id, 1]
        l_z[:, e_id] = voxel[:, 2] - electrode_xyz[e_id, 2]
        l[:, e_id] = np.sqrt(l_x[:, e_id]**2 + l_y[:, e_id]**2 + l_z[:, e_id]**2)
    l[l < 1] = 1  # electrode is at least 1 mm away from tissue
    
    neighbor_id_2d_copy = neighbor_id_2d.copy() # NOTE: without .copy(), changes of px will also change neighbor_id_2d

    px = neighbor_id_2d_copy[:, 0] # x +. 
    px_0_id = px == -1 # these have no neighbors
    px[px_0_id] = 0 # set to some valid index to avoid error
    
    mx = neighbor_id_2d_copy[:, 1] # x -
    mx_0_id = mx == -1 # these have no neighbors
    mx[mx_0_id] = 0 # set to some valid index to avoid error
    
    py = neighbor_id_2d_copy[:, 2] # y +
    py_0_id = py == -1 # these have no neighbors
    py[py_0_id] = 0 # set to some valid index to avoid error
    
    my = neighbor_id_2d_copy[:, 3] # y -
    my_0_id = my == -1 # these have no neighbors
    my[my_0_id] = 0 # set to some valid index to avoid error
    
    pz = neighbor_id_2d_copy[:, 4] # z +
    pz_0_id = pz == -1 # these have no neighbors
    pz[pz_0_id] = 0 # set to some valid index to avoid error
    
    mz = neighbor_id_2d_copy[:, 5] # z -
    mz_0_id = mz == -1 # these have no neighbors
    mz[mz_0_id] = 0 # set to some valid index to avoid error
    
    dvdx = (action_potential[px, :] - action_potential[mx, :]) / (2 * Delta)
    dvdx[px_0_id, :] = 0 # these have no neighbors, set gradient to zero
    dvdx[mx_0_id, :] = 0 # these have no neighbors, set gradient to zero
    
    dvdy = (action_potential[py, :] - action_potential[my, :]) / (2 * Delta)
    dvdy[py_0_id, :] = 0 # these have no neighbors, set gradient to zero
    dvdy[my_0_id, :] = 0 # these have no neighbors, set gradient to zero
    
    dvdz = (action_potential[pz, :] - action_potential[mz, :]) / (2 * Delta)
    dvdz[pz_0_id, :] = 0 # these have no neighbors, set gradient to zero
    dvdz[mz_0_id, :] = 0 # these have no neighbors, set gradient to zero
    
    D11_b = np.tile(D11[:, np.newaxis], (1, n_electrode))
    D12_b = np.tile(D12[:, np.newaxis], (1, n_electrode))
    D13_b = np.tile(D13[:, np.newaxis], (1, n_electrode))
    D21_b = np.tile(D21[:, np.newaxis], (1, n_electrode))
    D22_b = np.tile(D22[:, np.newaxis], (1, n_electrode))
    D23_b = np.tile(D23[:, np.newaxis], (1, n_electrode))
    D31_b = np.tile(D31[:, np.newaxis], (1, n_electrode))
    D32_b = np.tile(D32[:, np.newaxis], (1, n_electrode))
    D33_b = np.tile(D33[:, np.newaxis], (1, n_electrode))

    T = action_potential.shape[1]
    electrogram_unipolar = np.zeros((n_electrode, T))
    for t_id in range(T):
        if (t_id + 1) % (T // 5) == 0:
            print(f'compute electrogram {(t_id + 1) / T * 100:.1f}%')
        
        dvdx_b = np.tile(dvdx[:, t_id].reshape(-1, 1), (1, n_electrode))
        dvdy_b = np.tile(dvdy[:, t_id].reshape(-1, 1), (1, n_electrode))
        dvdz_b = np.tile(dvdz[:, t_id].reshape(-1, 1), (1, n_electrode))
        
        egm_part = compute_voxel(n_voxel, n_electrode, D11_b, D12_b, D13_b, D21_b, D22_b, D23_b, D31_b, D32_b, D33_b, c_voxel, l, l_x, l_y, l_z, dvdx_b, dvdy_b, dvdz_b)

        electrogram_unipolar[:, t_id] = np.sum(egm_part, axis=0)
    
    # adjust the electrogram magnitude
    typical_magnitude = 1  # unit: volt
    
    voltage = np.zeros(n_electrode)
    for n in range(n_electrode):
        voltage[n] = np.ptp(electrogram_unipolar[n, :])  # np.ptp is equivalent to range()
    
    voltage_mean = np.mean(voltage)
    
    scale = typical_magnitude / voltage_mean
    electrogram_unipolar = electrogram_unipolar * scale
    
    debug_plot = 0
    if debug_plot == 1:
        plt.figure()
        # e_id = 0
        # plt.plot(electrogram_unipolar[e_id,:], 'b')
        plt.plot(electrogram_unipolar.T)
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (scaled)')
        plt.title('Unipolar electrogram')
        plt.show()
    
    return electrogram_unipolar
