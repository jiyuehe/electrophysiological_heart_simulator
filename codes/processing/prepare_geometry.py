import scipy.io
import numpy as np
import plotly.graph_objects as go # pip install plotly, pip install --upgrade nbformat. For 3D interactive plot: triangular mesh, and activation movie
import plotly.io as pio
pio.renderers.default = "browser" # simulation result mesh display in internet browser
import os

def execute(data_path):
    mat_data = scipy.io.loadmat(data_path + "heart_example.mat")

    voxel = mat_data['data']['geometry'][0,0]['edited'][0,0]['volume'][0,0]['voxel'][0,0] # xyz coordinates of each voxel
    neighbor_id_2d = mat_data['data']['geometry'][0,0]['edited'][0,0]['volume'][0,0]['voxel_based_voxels'][0,0].astype(np.int32) -1 # -1 is to convert Matlab 1-based index to Python 0-based index
    Delta = mat_data['data']['geometry'][0,0]['edited'][0,0]['volume'][0,0]['delta'][0,0][0,0] # voxel spacing
    voxel_for_each_vertex = mat_data['data']['geometry'][0,0]['edited'][0,0]['voxel_for_each_vertex'][0,0].astype(np.int32) -1 # -1 is to convert Matlab 1-based index to Python 0-based index
    voxel_for_each_vertex = voxel_for_each_vertex.flatten() # convert to 1D array
    vertex_for_each_voxel = mat_data['data']['geometry'][0,0]['edited'][0,0]['vertex_for_each_voxel'][0,0].astype(np.int32) -1 # -1 is to convert Matlab 1-based index to Python 0-based index
    vertex_for_each_voxel = vertex_for_each_voxel.flatten() # convert to 1D array
    vertex = mat_data['data']['geometry'][0,0]['edited'][0,0]['vertex'][0,0] # xyz coordinates of each vertex
    face = mat_data['data']['geometry'][0,0]['edited'][0,0]['face'][0,0].astype(np.int32) -1 # -1 is to convert Matlab 1-based index to Python 0-based index

    if os.path.exists(data_path + 'vertex_flag.npy'): # file exist
        vertex_flag = np.load(data_path + 'vertex_flag.npy')
    else: # file do not exist
        vertex_flag = np.zeros(len(vertex), dtype=int)

    return voxel, neighbor_id_2d, Delta, voxel_for_each_vertex, vertex_for_each_voxel, vertex, face, vertex_flag

    debug_plot = 0
    if debug_plot == 1:
        # plot voxels
        fig = go.Figure(data=[
            go.Scatter3d(
                x=voxel[:, 0], y=voxel[:, 1], z=voxel[:, 2],
                mode='markers',
                marker=dict(size=1, color='blue')
            )
        ])
        fig.show()

        # plot the voxels near the triangular mesh
        voxel_2 = voxel[voxel_for_each_vertex,:]
        fig = go.Figure(data=[
            go.Scatter3d(
                x=voxel_2[:, 0], y=voxel_2[:, 1], z=voxel_2[:, 2],
                mode='markers',
                marker=dict(size=1, color='blue')
            )
        ])
        fig.show()

        # plot the triangular mesh
        fig = go.Figure(data=[
            go.Mesh3d(
                x=vertex[:, 0], y=vertex[:, 1], z=vertex[:, 2],
                i=face[:, 0], j=face[:, 1], k=face[:, 2],
                color='white'
            )
        ])
        fig.show()
