import codes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from matplotlib.animation import PillowWriter
import plotly.graph_objects as go # pip install plotly, pip install --upgrade nbformat. For 3D interactive plot: triangular mesh, and activation movie
import plotly.io as pio
pio.renderers.default = "browser" # simulation result mesh display in internet browser
import time

# will need to install ffmpeg for saving the movie as a mp4 file
# Ubuntu Linux: sudo apt install ffmpeg
# MacOS: brew install ffmpeg

# activation movie display on volume using matplotlib, option to save as mp4
# ------------------------------
def execute_on_voxel_save_as_mp4(save_movie_flag, movie_data, node, t, geometry_flag): 
    v_gate = 0.13

    data_min = 0 
    data_max = 1 # action potential value can be large at the pacing site, that's why cap it 1 here so that the colors are good
    data_threshold = v_gate
    map_color = {}
    n_time = movie_data.shape[1]
    for n in range(n_time):
        if ((n+1) % (n_time//5)) == 0:
            print(f'compute color map {(n+1)/n_time*100:.0f}%')
        data = movie_data[:, n]
        color = codes.convert_data_to_color.execute(data, data_min, data_max, data_threshold)
        map_color[n] = color

    fig = plt.figure(figsize=(10, 8))
    
    if geometry_flag != 0: # 3D
        ax = plt.axes(projection='3d')
        plot_handle = ax.scatter(node[:, 0], node[:, 1], node[:, 2], c=map_color[0], edgecolor='none', linewidth=0)
        plt.axis('off')
        codes.set_axes_equal.execute(ax)
    elif geometry_flag == 0: # 2D sheet
        nx = int(np.max(node[:,0]) - np.min(node[:,0])) + 1
        ny = int(np.max(node[:,1]) - np.min(node[:,1])) + 1
        color_image = map_color[0].reshape((nx, ny, 3))  # shape (30, 20, 3)
        color_image = np.swapaxes(color_image, 0, 1)  # swap to (ny, nx) -> (20,30) for imshow

        ax = plt.axes()
        plot_handle = ax.imshow(color_image, origin='lower', interpolation='nearest')
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    pause_interval = 0.001
    view_matrices = {} # dictionary to store projection matrices for each frame
    n_time = movie_data.shape[1]
    for n in range(n_time):
        if ((n+1) % (n_time//5)) == 0:
            print(f'playing movie {(n+1)/n_time*100:.0f}%')

        if geometry_flag == 0: # 2D sheet
            color_image = map_color[n].reshape((nx, ny, 3))  # shape (30, 20, 3)
            color_image = np.swapaxes(color_image, 0, 1)  # swap to (ny, nx) -> (20,30) for imshow
            plot_handle.set_data(color_image)
        elif geometry_flag != 0: # 3D
            plot_handle.set_color(map_color[n])
        
        ax.set_title(f'Time: {n}/{n_time} ms') # set title with current time step

        # capture current view angles
        if geometry_flag != 0: # 3D
            view_matrices[n] = ax.get_proj().copy() # copy to avoid overwriting

        plt.pause(pause_interval)

    # save simulation movie
    if save_movie_flag == 1:
        print("saving movie")

        def animate(n):
            if ((n+1) % (n_time//10)) == 0:
                print(f'saving movie {(n+1)/n_time*100:.0f}%')

            if geometry_flag == 0: # 2D sheet
                color_image = map_color[n].reshape((nx, ny, 3))  # shape (30, 20, 3)
                color_image = np.swapaxes(color_image, 0, 1)  # swap to (ny, nx) -> (20,30) for imshow
                plot_handle.set_data(color_image)
            elif geometry_flag != 0: # 3D
                plot_handle.set_color(map_color[n])
            
            ax.set_title(f'Time: {n}/{n_time} ms') # set title with current time step

            # restore view angle
            if geometry_flag != 0: # 3D
                R = view_matrices[n]
                ax.get_proj = lambda R=R: R # use default argument to capture current R

        frame_skip = 5
        anim = animation.FuncAnimation(fig, animate, frames=range(0, n_time, frame_skip), interval=1, blit=False, repeat=False)
        # the 'interval' parameter specifies the delay between frames in milliseconds

        # save
        writer = PillowWriter(fps=20)
        anim.save('./result/simulation movie.gif', writer=writer, dpi=60)

        print("movie is saved")

# display movie (will open in internet browser)
# ------------------------------
def execute_on_volume_display_in_browser(voxel, map_color):
    fig = go.Figure(
        data = [
            go.Scatter3d(
                x = voxel[:, 0], y = voxel[:, 1], z = voxel[:, 2],
                mode = 'markers',
                marker = dict(color = map_color[0], size = 3))
        ],

        layout = go.Layout(
            updatemenus = [{
                "type": "buttons",
                "buttons": [
                    {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 5}, "fromcurrent": True, "mode": "immediate"}]},
                    {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}, "mode": "immediate", "transition": {"duration": 0}}]}
                ]
            }],

            sliders = [{
                'active': 0,
                'currentvalue': {"prefix": "Time: "},
                'pad': {"t": 50},
                'steps': [{'label': str(t), 'method': 'animate', 'args': [[str(t)], {'frame': {'duration': 0}, 'mode': 'immediate'}]}
                for t in range(len(map_color))]
            }]
        ),

        frames = [
            go.Frame(data = [go.Scatter3d(marker = dict(color = map_color[t], size = 3))], name = str(t))
            for t in range(len(map_color))
        ]
    )
    fig.show()

# display movie (will open in internet browser)
# ------------------------------
def execute_on_mesh_display_in_browser(vertex, face, map_color):
    fig = go.Figure(
        data = [
            go.Mesh3d(
                x = vertex[:, 0], y = vertex[:, 1], z = vertex[:, 2],
                i = face[:, 0], j = face[:, 1], k = face[:, 2],
                vertexcolor = map_color[0])
        ],

        layout = go.Layout(
            updatemenus = [{
                "type": "buttons",
                "buttons": [
                    {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 5}, "fromcurrent": True, "mode": "immediate"}]},
                    {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}, "mode": "immediate", "transition": {"duration": 0}}]}
                ]
            }],

            sliders = [{
                'active': 0,
                'currentvalue': {"prefix": "Time: "},
                'pad': {"t": 50},
                'steps': [{'label': str(t), 'method': 'animate', 'args': [[str(t)], {'frame': {'duration': 0}, 'mode': 'immediate'}]}
                for t in range(len(map_color))]
            }]
        ),

        frames = [
            go.Frame(data = [go.Mesh3d(vertexcolor = map_color[t])], name = str(t))
            for t in range(len(map_color))
        ]
    )
    fig.show()
