import pymeshlab
import trimesh
import numpy as np
import os
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import plotly.graph_objects as go
from werkzeug.utils import secure_filename
from sklearn.cluster import KMeans
import random


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def calculate_volume(mesh):
    bounding_box = mesh.bounding_box

    length = bounding_box.extents[0]  # Length along the X-axis
    width = bounding_box.extents[1]   # Width along the Y-axis
    height = bounding_box.extents[2]  # Height along the Z-axis
    volume = mesh.volume
    return length, width, height, volume

def generate_plotly_figure_with_texture(mesh_path, texture_path=None):
    """
    Generate a Plotly 3D mesh figure with optional texture.
    """
    # Load the mesh
    mesh = trimesh.load(mesh_path, process=False)

    # Extract vertices and faces
    vertices = mesh.vertices
    faces = mesh.faces

    # Prepare vertex colors based on texture mapping if available
    vertex_colors = None
    if texture_path and mesh.visual.uv is not None:
        uv_coords = mesh.visual.uv  # UV mapping (2D coordinates)
        texture_image = Image.open(texture_path)  # Load texture image
        texture_pixels = np.array(texture_image)

        # Map UV coordinates to pixel colors
        u = (uv_coords[:, 0] * (texture_pixels.shape[1] - 1)).astype(int)
        v = (1 - uv_coords[:, 1]) * (texture_pixels.shape[0] - 1)  # Flip vertically
        v = v.astype(int)

        # Assign colors to vertices based on UV mapping
        vertex_colors = texture_pixels[v, u, :3] / 255.0  # Normalize RGB to [0, 1]

    # Create the Plotly mesh object
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                vertexcolor=vertex_colors if vertex_colors is not None else None,
                color='lightblue' if vertex_colors is None else None,
                opacity=1.0,
            )
        ]
    )

    # Add layout for better visualization
    fig.update_layout(
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis',
        ),
        margin=dict(l=0, r=0, b=0, t=0),
    )

    return fig.to_html(full_html=False)

def generate_plotly_figure_with_plane(mesh_path, texture_path=None):
    """
    Generate a Plotly 3D mesh figure with an added YZ plane at the median X.
    """
    # Load the mesh
    mesh = trimesh.load(mesh_path, process=False)

    # Extract vertices and faces
    vertices = mesh.vertices
    faces = mesh.faces

    # Prepare vertex colors based on texture mapping if available
    vertex_colors = None
    if texture_path and mesh.visual.uv is not None:
        uv_coords = mesh.visual.uv  # UV mapping (2D coordinates)
        texture_image = Image.open(texture_path)  # Load texture image
        texture_pixels = np.array(texture_image)

        # Map UV coordinates to pixel colors
        u = (uv_coords[:, 0] * (texture_pixels.shape[1] - 1)).astype(int)
        v = (1 - uv_coords[:, 1]) * (texture_pixels.shape[0] - 1)  # Flip vertically
        v = v.astype(int)

        # Assign colors to vertices based on UV mapping
        vertex_colors = texture_pixels[v, u, :3] / 255.0  # Normalize RGB to [0, 1]

    # Create the Plotly mesh object for the original mesh
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                vertexcolor=vertex_colors if vertex_colors is not None else None,
                color='lightblue' if vertex_colors is None else None,
                opacity=1.0,
            ),
            # Add the YZ plane at the median X-coordinate
            go.Mesh3d(
                x=[0, 0, 0, 0],  # All vertices have the same X (median X)
                y=[mesh.bounds[0][1], mesh.bounds[0][1], mesh.bounds[1][1], mesh.bounds[1][1]],  # Y limits
                z=[mesh.bounds[0][2], mesh.bounds[1][2], mesh.bounds[1][2], mesh.bounds[0][2]],  # Z limits
                i=[0, 0, 0, 0],  # Indices to form the quad
                j=[1, 2, 3, 1],
                k=[2, 3, 0, 3],
                color='red',  # YZ plane color
                opacity=0.5,  # Semi-transparent
            )
        ]
    )

    fig.update_layout(
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis',
        ),
        margin=dict(l=0, r=0, b=0, t=0),
    )

    return fig.to_html(full_html=False)


def mirror_and_overlap(fig_left, fig_right):
    # Get the data from the left and right figures
    left_data = fig_left['data'][0]
    right_data = fig_right['data'][0]

    # Flip the x-coordinates of the left side vertices to mirror it
    mirrored_left_x = -left_data['x']  # Negating x-coordinates for mirroring

    # Combine the left and right side data
    combined_x = np.concatenate([mirrored_left_x, right_data['x']])
    combined_y = np.concatenate([left_data['y'], right_data['y']])
    combined_z = np.concatenate([left_data['z'], right_data['z']])
    
    # Assign colors to left and right sides: 
    # Left part in red and right part in blue (you can change these as needed)
    left_colors = np.repeat([1, 0, 0], len(left_data['x']))  # Red color for left half
    right_colors = np.repeat([0, 0, 1], len(right_data['x']))  # Blue color for right half
    
    # Combine the colors
    combined_colors = np.concatenate([left_colors, right_colors])

    # Create a new plotly figure to show the overlapped result
    fig_combined = go.Figure(data=[go.Scatter3d(
        x=combined_x,
        y=combined_y,
        z=combined_z,
        mode='markers',
        marker=dict(size=5, color=combined_colors)
    )])

    fig_combined.update_layout(title='Mirrored and Overlapped Nose', scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ))

    return fig_combined

def nose_visualize_test(file_path, texture_path):


    # Load the 3D model
    #file_path = 'Pasha_guard_head.obj'  # Replace with your OBJ file path
    mesh = trimesh.load(file_path, process=False)

    # Extract vertices and UV coordinates
    vertices = mesh.vertices
    faces = mesh.faces
    uv_coords = mesh.visual.uv  # UV coordinates for texture mapping

    # Load the texture image
    #texture_path = 'Pasha_guard_head_0.png'  # Replace with your texture file path
    texture_img = Image.open(texture_path)
    texture_img = np.array(texture_img)  # Convert the texture to a NumPy array

    # Step 1: Perform K-means clustering on the vertices
    num_clusters = 45  # Number of clusters you want
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(vertices)

    # Get the labels for each vertex (which cluster each vertex belongs to)
    labels = kmeans.labels_

    # Step 2: Define a more refined approach to target the nose
    x_median = np.median(vertices[:, 0])
    x_range = 0.05 * np.ptp(vertices[:, 0])  # Smaller X range
    z_min = np.percentile(vertices[:, 2], 20)
    z_max = np.percentile(vertices[:, 2], 40)

    clusters = {i: vertices[labels == i] for i in range(num_clusters)}

    target_cluster_index = None
    best_distance = np.inf

    for i, cluster in clusters.items():
        avg_x = np.mean(cluster[:, 0])
        avg_z = np.mean(cluster[:, 2])
        if abs(avg_x - x_median) < x_range and z_min < avg_z < z_max:
            distance = abs(avg_z - np.median(vertices[:, 2]))
            if distance < best_distance:
                best_distance = distance
                target_cluster_index = i

    if target_cluster_index is None:
        raise ValueError("No valid cluster found for the nose region.")

    cluster_vertices_indices = np.where(labels == target_cluster_index)[0]

    # Filter the faces associated with the selected vertices
    index_map = {old: new for new, old in enumerate(cluster_vertices_indices)}
    filtered_faces = []
    for face in faces:
        if all(vi in index_map for vi in face):
            filtered_faces.append([index_map[vi] for vi in face])

    filtered_faces = np.array(filtered_faces)

    # Construct a new mesh for the selected cluster (nose)
    nose_mesh = trimesh.Trimesh(vertices=vertices[cluster_vertices_indices], faces=filtered_faces)

    nose_length, nose_width, nose_height, nose_volume = calculate_volume(nose_mesh)

    # Format the nose dimensions to 2 decimal places
    nose_length = f"{nose_length:.2f}"
    nose_width = f"{nose_width:.2f}"
    nose_height = f"{nose_height:.2f}"
    nose_volume = f"{nose_volume:.2f}"


    #nose_volume = nose_mesh.volume
    
    #print(f"Nose Region Volume: {nose_volume:.2f}")

    # Visualization
    cluster_uv_coords = uv_coords[cluster_vertices_indices]
    cluster_colors = np.zeros((len(vertices), 3))

    for i, uv in enumerate(cluster_uv_coords):
        x_tex = int(uv[0] * (texture_img.shape[1] - 1))
        y_tex = int((1 - uv[1]) * (texture_img.shape[0] - 1))
        cluster_colors[cluster_vertices_indices[i]] = texture_img[y_tex, x_tex][:3]

    fig = go.Figure(data=[go.Scatter3d(
        x=vertices[cluster_vertices_indices, 0],
        y=vertices[cluster_vertices_indices, 1],
        z=vertices[cluster_vertices_indices, 2],
        mode='markers',
        marker=dict(color=cluster_colors[cluster_vertices_indices] / 255.0, size=5)
    )])

    fig.update_layout(title=f'Cluster {target_cluster_index} (Nose) with Texture', scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ))

    # Step 1: Compute X median for the nose vertices
    nose_vertices = vertices[cluster_vertices_indices]
    x_median_nose = np.median(nose_vertices[:, 0])

    # Step 2: Separate vertices into left and right halves
    left_indices = np.where(nose_vertices[:, 0] < x_median_nose)[0]
    right_indices = np.where(nose_vertices[:, 0] >= x_median_nose)[0]

    # Map old indices to new indices within each half
    left_index_map = {old: new for new, old in enumerate(left_indices)}
    right_index_map = {old: new for new, old in enumerate(right_indices)}

    # Step 3: Filter faces for the left half
    left_faces = []
    for face in filtered_faces:
        if all(vi in left_index_map for vi in face):
            left_faces.append([left_index_map[vi] for vi in face])

    # Step 4: Filter faces for the right half
    right_faces = []
    for face in filtered_faces:
        if all(vi in right_index_map for vi in face):
            right_faces.append([right_index_map[vi] for vi in face])

    # Step 5: Convert filtered faces to numpy arrays
    left_faces = np.array(left_faces)
    right_faces = np.array(right_faces)

    # Step 6: Create trimesh objects for left and right halves
    left_nose_mesh = trimesh.Trimesh(vertices=nose_vertices[left_indices], faces=left_faces)
    right_nose_mesh = trimesh.Trimesh(vertices=nose_vertices[right_indices], faces=right_faces)

    # Calculate volumes for each half
    #left_volume = left_nose_mesh.volume
    #right_volume = right_nose_mesh.volume
    left_nose_length, left_nose_width, left_nose_height, left_nose_volume = calculate_volume(left_nose_mesh)
    right_nose_length, right_nose_width, right_nose_height, right_nose_volume = calculate_volume(right_nose_mesh)

    # Format the left nose dimensions to 2 decimal places
    left_nose_length = f"{left_nose_length:.2f}"
    left_nose_width = f"{left_nose_width:.2f}"
    left_nose_height = f"{left_nose_height:.2f}"
    left_nose_volume = f"{left_nose_volume:.2f}"

    # Format the right nose dimensions to 2 decimal places
    right_nose_length = f"{right_nose_length:.2f}"
    right_nose_width = f"{right_nose_width:.2f}"
    right_nose_height = f"{right_nose_height:.2f}"
    right_nose_volume = f"{right_nose_volume:.2f}"

    #print(f"Left Nose Volume: {left_volume:.2f}")
    #print(f"Right Nose Volume: {right_volume:.2f}")

    # Step 7: Visualize both halves
    # Visualization for Left Half
    fig_left = go.Figure(data=[go.Scatter3d(
        x=nose_vertices[left_indices, 0],
        y=nose_vertices[left_indices, 1],
        z=nose_vertices[left_indices, 2],
        mode='markers',
        marker=dict(size=5, color='blue')
    )])
    fig_left.update_layout(title='Left Half of Nose', scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ))
    #fig_left.show()

    # Visualization for Right Half
    fig_right = go.Figure(data=[go.Scatter3d(
        x=nose_vertices[right_indices, 0],
        y=nose_vertices[right_indices, 1],
        z=nose_vertices[right_indices, 2],
        mode='markers',
        marker=dict(size=5, color='red')
    )])
    fig_right.update_layout(title='Right Half of Nose', scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ))
    #fig_right.show()

    # Step 1: Load the texture
    #texture_path = 'Pasha_guard_head_0.png'
    texture_img = np.array(Image.open(texture_path))  # Texture as a NumPy array

    # Function to map UV coordinates to texture colors
    def get_texture_colors(uv_coords, indices):
        colors = np.zeros((len(indices), 3))
        for i, uv in enumerate(uv_coords[indices]):
            # Map UV coordinates to texture image pixels
            x_tex = int(uv[0] * (texture_img.shape[1] - 1))
            y_tex = int((1 - uv[1]) * (texture_img.shape[0] - 1))  # Flip Y for correct mapping
            colors[i] = texture_img[y_tex, x_tex, :3]  # Get RGB values
        return colors / 255.0  # Normalize to [0, 1] for visualization

    # Step 2: Get UV coordinates and colors for each half
    left_uv_colors = get_texture_colors(uv_coords[cluster_vertices_indices], left_indices)
    right_uv_colors = get_texture_colors(uv_coords[cluster_vertices_indices], right_indices)

    # Step 3: Visualize the left half with texture
    fig_left = go.Figure(data=[go.Scatter3d(
        x=nose_vertices[left_indices, 0],
        y=nose_vertices[left_indices, 1],
        z=nose_vertices[left_indices, 2],
        mode='markers',
        marker=dict(size=5, color=left_uv_colors)
    )])
    fig_left.update_layout(title='Left Half of Nose with Texture', scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ))
    #fig_left.show()

    # Step 4: Visualize the right half with texture
    fig_right = go.Figure(data=[go.Scatter3d(
        x=nose_vertices[right_indices, 0],
        y=nose_vertices[right_indices, 1],
        z=nose_vertices[right_indices, 2],
        mode='markers',
        marker=dict(size=5, color=right_uv_colors)
    )])
    fig_right.update_layout(title='Right Half of Nose with Texture', scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ))
    #fig_right.show()

    overlap_nose = mirror_and_overlap(fig_left, fig_right)


    return fig.to_html(full_html=False),fig_left.to_html(full_html=False),fig_right.to_html(full_html=False), nose_length, nose_width, nose_height, nose_volume,left_nose_length, left_nose_width, left_nose_height, left_nose_volume,right_nose_length, right_nose_width, right_nose_height, right_nose_volume, overlap_nose.to_html(full_html=False)



def kmeans_visualize(file_path):

    mesh = trimesh.load(file_path, process=False)

    # Extract vertices (positions in 3D space)
    vertices = mesh.vertices

    # Step 1: Perform K-means clustering on the vertices
    num_clusters = 45  # You can adjust the number of clusters based on the detail you want
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(vertices)

    # Get the labels for each vertex (which cluster each vertex belongs to)
    labels = kmeans.labels_

    # Step 2: Create random colors for each cluster
    def generate_random_color():
        return f'rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})'

    # Generate a list of random colors, one for each cluster
    cluster_colors = [generate_random_color() for _ in range(num_clusters)]

    # Step 3: Assign the randomly generated colors to the vertices based on their cluster label
    colors = [cluster_colors[label] for label in labels]

    # Step 4: Visualize the clustered vertices
    fig = go.Figure(data=[go.Scatter3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        mode='markers',
        marker=dict(color=colors, size=5)
    )])

    fig.update_layout(title='K-means Clustering of 3D Mesh Vertices with Random Colors', scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ))

    return fig.to_html(full_html=False)



def process_and_slice_mesh(file_path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(file_path)

    processed_mesh_path = "./uploads/processed_mesh.obj"
    ms.save_current_mesh(processed_mesh_path)

    mesh = trimesh.load(processed_mesh_path)



    x_coords = mesh.vertices[:, 0]
    x_median = np.median(x_coords)

    plane_origin = [x_median, 0, 0]
    plane_normal = [1, 0, 0]
    plane_normal_right = [-1, 0, 0]

    left_half = mesh.copy()
    left_half = left_half.slice_plane(plane_origin=plane_origin, plane_normal=plane_normal, side='back')

    right_half = mesh.copy()
    right_half = right_half.slice_plane(plane_origin=plane_origin, plane_normal=plane_normal_right, side='front')

    left_half_path = "./uploads/left_half.obj"
    right_half_path = "./uploads/right_half.obj"
    left_half.export(left_half_path)
    right_half.export(right_half_path)

    return left_half_path, right_half_path, processed_mesh_path, mesh, left_half, right_half

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/uploader', methods=['POST'])
def uploader():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return "No file selected. Please upload a valid .obj file.", 400

        filename = secure_filename(uploaded_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(file_path)

        texture_path = None
        texture_file = request.files.get('texture')
        if texture_file:
            texture_filename = secure_filename(texture_file.filename)
            texture_path = os.path.join(app.config['UPLOAD_FOLDER'], texture_filename)
            texture_file.save(texture_path)

        try:
            left_half_path, right_half_path, processed_mesh_path, mesh, left_half, right_half = process_and_slice_mesh(file_path)

            full_mesh_length, full_mesh_width, full_mesh_height, full_mesh_volume = calculate_volume(mesh)
            left_half_length, left_half_width, left_half_height, left_half_volume = calculate_volume(left_half)
            right_half_length, right_half_width, right_half_height, right_half_volume = calculate_volume(right_half)

            # Format the full mesh dimensions to 2 decimal places
            full_mesh_length = f"{full_mesh_length:.2f}"
            full_mesh_width = f"{full_mesh_width:.2f}"
            full_mesh_height = f"{full_mesh_height:.2f}"
            full_mesh_volume = f"{full_mesh_volume:.2f}"

            # Format the left half dimensions to 2 decimal places
            left_half_length = f"{left_half_length:.2f}"
            left_half_width = f"{left_half_width:.2f}"
            left_half_height = f"{left_half_height:.2f}"
            left_half_volume = f"{left_half_volume:.2f}"

            # Format the right half dimensions to 2 decimal places
            right_half_length = f"{right_half_length:.2f}"
            right_half_width = f"{right_half_width:.2f}"
            right_half_height = f"{right_half_height:.2f}"
            right_half_volume = f"{right_half_volume:.2f}"

            # Generate figures for all four views
            plotly_figure = generate_plotly_figure_with_texture(processed_mesh_path, texture_path)
            plotly_face_with_plane_figure = generate_plotly_figure_with_plane(processed_mesh_path, texture_path)
            plotly_left_half_figure = generate_plotly_figure_with_texture(left_half_path, texture_path)
            plotly_right_half_figure = generate_plotly_figure_with_texture(right_half_path, texture_path)

            # k-Means clustering
            kmeans_image = kmeans_visualize(file_path)

            nose_image, nose_left, nose_right, nose_length, nose_width, nose_height, nose_volume,left_nose_length, left_nose_width, left_nose_height, left_nose_volume,right_nose_length, right_nose_width, right_nose_height, right_nose_volume, overlap_nose = nose_visualize_test(file_path, texture_path)

            return render_template('display.html', 
                                    overlap_nose = overlap_nose,
                                   plotly_figure=plotly_figure,
                                   plotly_face_with_plane_figure=plotly_face_with_plane_figure,
                                   plotly_left_half_figure=plotly_left_half_figure,
                                   plotly_right_half_figure=plotly_right_half_figure,
                                   nose_image = nose_image,
                                   nose_left = nose_left,
                                   nose_right = nose_right,
                                   kmeans_image = kmeans_image,
                                   full_mesh_length = full_mesh_length,
                                    full_mesh_width = full_mesh_width,
                                    full_mesh_height = full_mesh_height,
                                    full_mesh_volume = full_mesh_volume,
                                    left_half_length = left_half_length,
                                    left_half_width = left_half_width,
                                    left_half_height = left_half_height,
                                    left_half_volume = left_half_volume,
                                    right_half_length = right_half_length,
                                    right_half_width = right_half_width,
                                    right_half_height = right_half_height,
                                    right_half_volume = right_half_volume,
                                   nose_length=nose_length,
                                   nose_width=nose_width,
                                   nose_height=nose_height,
                                   nose_volume=nose_volume,
                                   left_nose_length=left_nose_length,
                                   left_nose_width=left_nose_width,
                                   left_nose_height=left_nose_height,
                                   left_nose_volume=left_nose_volume,
                                   right_nose_length=right_nose_length,
                                   right_nose_width=right_nose_width,
                                   right_nose_height=right_nose_height,
                                   right_nose_volume=right_nose_volume
                                   )
        except Exception as e:
            return f"Error processing mesh: {e}", 500

@app.route('/display')
def display_mesh():
    full_mesh_length = request.args.get('full_mesh_length', 0)
    full_mesh_width = request.args.get('full_mesh_width', 0)
    full_mesh_height = request.args.get('full_mesh_height', 0)
    full_mesh_volume = request.args.get('full_mesh_volume', 0)

    left_half_length = request.args.get('left_half_length', 0)
    left_half_width = request.args.get('left_half_width', 0)
    left_half_height = request.args.get('left_half_height', 0)
    left_half_volume = request.args.get('left_half_volume', 0)

    right_half_length = request.args.get('right_half_length', 0)
    right_half_width = request.args.get('right_half_width', 0)
    right_half_height = request.args.get('right_half_height', 0)
    right_half_volume = request.args.get('right_half_volume', 0)

    nose_length = request.args.get('nose_length', 0)
    nose_width = request.args.get('nose_width', 0)
    nose_height = request.args.get('nose_height', 0)
    nose_volume = request.args.get('nose_volume', 0)

    left_nose_length = request.args.get('left_nose_length', 0)
    left_nose_width = request.args.get('left_nose_width', 0)
    left_nose_height = request.args.get('left_nose_height', 0)
    left_nose_volume = request.args.get('left_nose_volume', 0)

    right_nose_length = request.args.get('right_nose_length', 0)
    right_nose_width = request.args.get('right_nose_width', 0)
    right_nose_height = request.args.get('right_nose_height', 0)
    right_nose_volume = request.args.get('right_nose_volume', 0)
    

    return render_template('display.html',
                           full_mesh_length = full_mesh_length,
                                    full_mesh_width = full_mesh_width,
                                    full_mesh_height = full_mesh_height,
                                    full_mesh_volume = full_mesh_volume,
                                    left_half_length = left_half_length,
                                    left_half_width = left_half_width,
                                    left_half_height = left_half_height,
                                    left_half_volume = left_half_volume,
                                    right_half_length = right_half_length,
                                    right_half_width = right_half_width,
                                    right_half_height = right_half_height,
                                    right_half_volume = right_half_volume,
                                   nose_length=nose_length,
                                   nose_width=nose_width,
                                   nose_height=nose_height,
                                   nose_volume=nose_volume,
                                   left_nose_length=left_nose_length,
                                   left_nose_width=left_nose_width,
                                   left_nose_height=left_nose_height,
                                   left_nose_volume=left_nose_volume,
                                   right_nose_length=right_nose_length,
                                   right_nose_width=right_nose_width,
                                   right_nose_height=right_nose_height,
                                   right_nose_volume=right_nose_volume
                           )

if __name__ == '__main__':
    app.run(debug=True)
