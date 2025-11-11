import scipy.io
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

def execute(data_path):
    geometry_data = scipy.io.loadmat(data_path / 'heart_example.mat')

    voxel = geometry_data['data']['geometry'][0,0]['edited'][0,0]['volume'][0,0]['voxel'][0,0] # xyz coordinates of each voxel

    neighbor_id_2d = geometry_data['data']['geometry'][0,0]['edited'][0,0]['volume'][0,0]['voxel_based_voxels'][0,0].astype(np.int32) -1 # -1 is to convert Matlab 1-based index to Python 0-based index

    Delta = geometry_data['data']['geometry'][0,0]['edited'][0,0]['volume'][0,0]['delta'][0,0][0,0] # voxel spacing

    voxel_for_each_vertex = geometry_data['data']['geometry'][0,0]['edited'][0,0]['voxel_for_each_vertex'][0,0].astype(np.int32) -1 # -1 is to convert Matlab 1-based index to Python 0-based index
    voxel_for_each_vertex = voxel_for_each_vertex.flatten() # convert to 1D array

    vertex_for_each_voxel = geometry_data['data']['geometry'][0,0]['edited'][0,0]['vertex_for_each_voxel'][0,0].astype(np.int32) -1 # -1 is to convert Matlab 1-based index to Python 0-based index
    vertex_for_each_voxel = vertex_for_each_voxel.flatten() # convert to 1D array

    vertex = geometry_data['data']['geometry'][0,0]['edited'][0,0]['vertex'][0,0] # xyz coordinates of each vertex

    face = geometry_data['data']['geometry'][0,0]['edited'][0,0]['face'][0,0].astype(np.int32) -1 # -1 is to convert Matlab 1-based index to Python 0-based index

    output = {
        'voxel': voxel,
        'neighbor_id_2d': neighbor_id_2d,
        'Delta': Delta,
        'voxel_for_each_vertex': voxel_for_each_vertex,
        'vertex_for_each_voxel': vertex_for_each_voxel,
        'vertex': vertex,
        'face': face
    }

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

    return output
