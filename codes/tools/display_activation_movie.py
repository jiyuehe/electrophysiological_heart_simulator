import codes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import plotly.graph_objects as go # pip install plotly, pip install --upgrade nbformat. For 3D interactive plot: triangular mesh, and activation movie
import plotly.io as pio
pio.renderers.default = "browser" # simulation result mesh display in internet browser

def execute_on_volume(voxel, map_color):
    # display movie (will open in internet browser)
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

def execute_on_mesh(vertex, face, map_color):
    # display movie (will open in internet browser)
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

def execute_on_voxel_save_as_mp4(save_flag, movie_data, voxel): # activation phase movie display on volume using matplotlib, option to save as mp4
    data_min = np.min(movie_data)
    data_max = np.max(movie_data)
    data_threshold = data_min
    map_color = {}
    n_time = movie_data.shape[1]
    for n in range(n_time):
        if ((n+1) % (n_time//5)) == 0:
            print(f'compute color map {(n+1)/n_time*100:.1f}%')
        data = movie_data[:, n]
        color = codes.convert_data_to_color.execute(data, data_min, data_max, data_threshold)
        map_color[n] = color

    d_buffer = 5 
    x_min = np.min(voxel[:,0]) - d_buffer
    y_min = np.min(voxel[:,1]) - d_buffer
    z_min = np.min(voxel[:,2]) - d_buffer
    x_max = np.max(voxel[:,0]) + d_buffer
    y_max = np.max(voxel[:,1]) + d_buffer
    z_max = np.max(voxel[:,2]) + d_buffer

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    ax.view_init(elev = -50, azim = 100)
    plot_handle = ax.scatter(voxel[:, 0], voxel[:, 1], voxel[:, 2], c=map_color[0], s=2, alpha=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    codes.set_axes_equal.execute(ax)

    pause_interval = 0.001
    view_angles = {} # dictionary to store view angles for each frame
    n_time = movie_data.shape[1]
    for n in range(n_time):
        if ((n+1) % (n_time//5)) == 0:
            print(f'playing movie {(n+1)/n_time*100:.1f}%')

        plot_handle.set_facecolor(map_color[n]) # set color based on phase to each voxel
        ax.set_title(f'Time: {n}/{n_time} ms') # set title with current time step

        # capture current view angles
        elev = ax.elev # elevation angle
        azim = ax.azim # azimuth angle
        view_angles[n] = {'elev': elev, 'azim': azim}

        plt.pause(pause_interval)

    # save simulation movie as mp4
    if save_flag == 1:
        print("saving movie as mp4")

        def animate(n):
            if ((n+1) % (n_time//10)) == 0:
                print(f'saving movie {(n+1)/n_time*100:.1f}%')

            plot_handle.set_facecolor(map_color[n]) # set color based on phase to each voxel
            ax.set_title(f'Time: {n}/{n_time} ms') # set title with current time step

            ax.view_init(elev=view_angles[n]['elev'], azim=view_angles[n]['azim']) # restore view angle

        anim = animation.FuncAnimation(fig, animate, frames=n_time, interval=10, blit=False, repeat=False)
        # the interval parameter specifies the delay between frames in milliseconds

        # save
        writer = FFMpegWriter(fps=10, bitrate=1800)
        anim.save('result/simulation movie.mp4', writer=writer)

        print("movie saved as mp4")
