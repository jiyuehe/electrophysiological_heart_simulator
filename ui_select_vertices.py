import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

class MeshSelector:
    def __init__(self, vertices, faces, data_path, vertex_flag):
        self.vertices = vertices
        self.faces = faces
        self.data_path = data_path
        self.vertex_flag = vertex_flag
        
        self.selection_polygon = []
        self.is_selecting = False
        self.selection_mode = False # Toggle between rotate and select modes
        
        # Create figure and 3D axis
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Plot the mesh
        self.plot_mesh()
        
        # Connect mouse events
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Add text input box for flag value
        from matplotlib.widgets import Button, TextBox
        ax_textbox = plt.axes([0.15, 0.90, 0.1, 0.04])
        self.textbox = TextBox(ax_textbox, 'Flag Value:', initial='1')
        
        # Add save button
        ax_save = plt.axes([0.81, 0.90, 0.08, 0.04])
        self.btn_save = Button(ax_save, 'Save')
        self.btn_save.on_clicked(self.save_selection)
        
        # Instructions
        self.fig.text(0.5, 0.02, 
                     'Press "A" to toggle SELECTION mode | LEFT DRAG: Add to selection | RIGHT CLICK: Clear all',
                     ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.mode_text = self.fig.text(0.5, 0.95, 'MODE: ROTATE (press A to select)', 
                                       ha='center', fontsize=12, weight='bold',
                                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.show()
    
    def plot_mesh(self):
        # Plot the triangular mesh
        self.ax.clear()
        
        # Plot mesh faces
        self.ax.plot_trisurf(self.vertices[:, 0], 
                            self.vertices[:, 1], 
                            self.vertices[:, 2],
                            triangles=self.faces,
                            alpha=0.8,
                            color='lightblue',
                            edgecolor='gray',
                            linewidth=0.2)
        
        # Highlight selected vertices (where flag > 0) with different colors
        unique_flags = np.unique(self.vertex_flag[self.vertex_flag > 0])
        if len(unique_flags) > 0:
            # Define color map for different flags
            colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 'purple']
            
            for i, flag_value in enumerate(unique_flags):
                flag_indices = np.where(self.vertex_flag == flag_value)[0]
                selected_verts = self.vertices[flag_indices]
                color = colors[i % len(colors)]  # Cycle through colors if more than 8 flags
                
                self.ax.scatter(selected_verts[:, 0],
                              selected_verts[:, 1],
                              selected_verts[:, 2],
                              c=color, s=100, alpha=1.0, 
                              depthshade=False,
                              label=f'Flag {flag_value}: {len(flag_indices)}')
            self.ax.legend()
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('3D Mesh Vertex Selection Tool')
        
        self.fig.canvas.draw_idle()
    
    def on_press(self, event):
        # Handle mouse press events
        if event.inaxes != self.ax:
            return
        
        if event.button == 1 and self.selection_mode: # Left click in selection mode
            self.is_selecting = True
            self.selection_polygon = [(event.xdata, event.ydata)]
        elif event.button == 3: # Right click - clear all selections
            self.vertex_flag = np.zeros(len(self.vertices), dtype=int)
            self.selection_polygon = []
            self.plot_mesh()
            print("All selections cleared")
    
    def on_motion(self, event):
        # Handle mouse motion during selection
        if not self.is_selecting or event.inaxes != self.ax:
            return
        
        if event.xdata is not None and event.ydata is not None:
            self.selection_polygon.append((event.xdata, event.ydata))
            
            # Draw the selection polygon in real-time
            if len(self.selection_polygon) > 1:
                self.plot_mesh()
                poly_array = np.array(self.selection_polygon)
                self.ax.plot(poly_array[:, 0], poly_array[:, 1], 'r-', linewidth=2)
                self.fig.canvas.draw_idle()
    
    def on_release(self, event):
        # Handle mouse release - complete selection
        if not self.is_selecting or event.button != 1:
            return
        
        self.is_selecting = False
        
        if len(self.selection_polygon) < 3:
            print("Selection too small, need at least 3 points")
            self.selection_polygon = []
            return
        
        # Close the polygon
        self.selection_polygon.append(self.selection_polygon[0])
        
        # Project vertices to 2D screen coordinates and update flags
        self.select_vertices_in_polygon()
        
        # Clear polygon and redraw
        self.selection_polygon = []
        self.plot_mesh()
    
    def on_key(self, event):
        # Handle key press events
        if event.key.lower() == 'a':
            self.selection_mode = not self.selection_mode
            if self.selection_mode:
                self.mode_text.set_text('MODE: SELECTION (press A to rotate)')
                self.mode_text.set_bbox(dict(boxstyle='round', facecolor='yellow', alpha=0.8))
                # Disable 3D rotation
                self.ax.disable_mouse_rotation()
                print("Selection mode ON - Click and drag to add vertices to selection")
            else:
                self.mode_text.set_text('MODE: ROTATE (press A to select)')
                self.mode_text.set_bbox(dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
                # Enable 3D rotation
                self.ax.mouse_init()
                print("Rotation mode ON - Drag to rotate view")
            self.fig.canvas.draw_idle()
    
    def save_selection(self, event):
        # Save vertex flag array to a numpy file
        num_selected = np.sum(self.vertex_flag)
        if num_selected == 0:
            print("No vertices selected to save!")
            return
        
        filename = 'vertex_flag.npy'
        np.save(self.data_path + filename, self.vertex_flag)
        print(f"Saved vertex flags to '{filename}' ({num_selected} vertices selected)")
    
    def select_vertices_in_polygon(self):
        # Select vertices that fall within the drawn polygon (front face only)
        # and set their flags to the current flag value (accumulative)
        if len(self.selection_polygon) < 3:
            return
        
        # Get the flag value from text box
        try:
            flag_value = int(self.textbox.text)
        except ValueError:
            print(f"Invalid flag value: '{self.textbox.text}'. Using 1.")
            flag_value = 1
        
        # Get the current 3D to 2D projection matrix
        proj_matrix = self.ax.get_proj()
        
        # Transform 3D vertices to 2D display coordinates and get depths
        vertices_2d = []
        depths = []
        for vertex in self.vertices:
            vec = np.array([vertex[0], vertex[1], vertex[2], 1.0])
            proj = proj_matrix @ vec
            
            if proj[3] != 0:
                x_2d = proj[0] / proj[3]
                y_2d = proj[1] / proj[3]
                z_depth = proj[2] / proj[3]
            else:
                x_2d = proj[0]
                y_2d = proj[1]
                z_depth = proj[2]
            
            vertices_2d.append([x_2d, y_2d])
            depths.append(z_depth)
        
        vertices_2d = np.array(vertices_2d)
        depths = np.array(depths)
        
        # Create a path from the selection polygon
        path = Path(self.selection_polygon)
        
        # Check which vertices are inside the polygon
        inside = path.contains_points(vertices_2d)
        
        # For each vertex inside, check if it's visible (front-facing)
        if np.sum(inside) > 0:
            depths_inside = depths[inside]
            # Use 40th percentile as cutoff - only select closer half
            depth_threshold = np.percentile(depths_inside, 40)
            front_facing = depths <= depth_threshold
            inside = inside & front_facing
        
        # Update vertex flags for newly selected vertices
        newly_selected = np.sum(inside & (self.vertex_flag == 0))
        self.vertex_flag[inside] = flag_value
        
        total_selected = np.sum(self.vertex_flag > 0)
        print(f"Set {np.sum(inside)} vertices to flag {flag_value} (total selected: {total_selected})")
    
    def get_vertex_flag(self):
        # Return the vertex flag array
        return self.vertex_flag

if __name__ == "__main__":
    # Create a sample atrium-like mesh (ellipsoid shape)
    n_theta = 30
    n_phi = 20
    
    theta = np.linspace(0, 2*np.pi, n_theta)
    phi = np.linspace(0, np.pi, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    
    # Create ellipsoid vertices (simplified atrium shape)
    a, b, c = 2.0, 1.5, 1.8  # semi-axes
    x = a * np.sin(phi_grid) * np.cos(theta_grid)
    y = b * np.sin(phi_grid) * np.sin(theta_grid)
    z = c * np.cos(phi_grid)
    
    vertices = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
    
    # Create faces (triangulation)
    faces = []
    for i in range(n_phi - 1):
        for j in range(n_theta - 1):
            # Two triangles per quad
            v1 = i * n_theta + j
            v2 = i * n_theta + (j + 1)
            v3 = (i + 1) * n_theta + j
            v4 = (i + 1) * n_theta + (j + 1)
            
            faces.append([v1, v2, v3])
            faces.append([v2, v4, v3])
    
    faces = np.array(faces)
    
    script_dir = os.path.dirname(os.path.abspath(__file__)) # get the path of the current script
    os.chdir(script_dir) # change the working directory
    data_path = script_dir + "/data/"

    if os.path.exists(data_path + 'vertex_flag.npy'): # file exist
        vertex_flag = np.load(data_path + 'vertex_flag.npy')
    else: # file do not exist
        vertex_flag = np.zeros(len(vertices), dtype=int)

    selector = MeshSelector(vertices, faces, data_path, vertex_flag)
    
    # After closing the window, you can access vertex flags:
    # vertex_flag = selector.get_vertex_flag()