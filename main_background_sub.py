import open3d as o3d
import numpy as np

"""
Single-frame LiDAR foreground extraction using background subtraction.

This script demonstrates the complete foreground-extraction pipeline on
one scene point cloud and one static-background point cloud.

Processing stages:
1. load the scene and reference background;
2. optionally voxel-downsample both point clouds;
3. build a KD-tree from the background;
4. classify scene points according to their nearest-background distance;
5. retain points sufficiently different from the background;
6. remove isolated foreground measurements using statistical filtering;
7. cluster the remaining foreground using DBSCAN;
8. retain the largest cluster as the detected person.

Intermediate point clouds are visualized to facilitate parameter tuning.
"""

import os

import open3d as o3d
import numpy as np
import os


def select_camera_view(pcd):
    """
    Open an interactive Open3D window and let the user choose the camera view.

    Close the window when the desired viewpoint has been selected.

    Returns
    -------
    camera_params : open3d.camera.PinholeCameraParameters
        Camera parameters that can be reused for all subsequent figures.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window("Select camera view", width=1280, height=720)

    vis.add_geometry(pcd)

    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([0, 0, 0])
    render_option.point_size = 2.0

    # Adjust the view manually, then close the window.
    vis.run()

    camera_params = (
        vis.get_view_control()
        .convert_to_pinhole_camera_parameters()
    )

    vis.destroy_window()

    return camera_params


def save_pointcloud_image(
    pcd,
    filename,
    camera_params,
    width=1280,
    height=720
):
    """
    Save a point cloud image using a fixed camera viewpoint.

    Parameters
    ----------
    pcd : open3d.geometry.PointCloud
        Point cloud to render.
    filename : str
        Output PNG filename.
    camera_params : open3d.camera.PinholeCameraParameters
        Camera parameters used for all figures.
    """
    vis = o3d.visualization.Visualizer()

    vis.create_window(
        window_name="Render",
        width=width,
        height=height,
        visible=True
    )

    vis.add_geometry(pcd)

    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([0, 0, 0])
    render_option.point_size = 2.0

    # Restore exactly the same camera viewpoint.
    view_control = vis.get_view_control()

    view_control.convert_from_pinhole_camera_parameters(
        camera_params,
        allow_arbitrary=True
    )

    vis.poll_events()
    vis.update_renderer()

    vis.capture_screen_image(
        filename,
        do_render=True
    )

    vis.destroy_window()


image_folder = "background_subtraction_images"
os.makedirs(image_folder, exist_ok=True)

# ---------------------------------------------------------------------
# 1. Load scene and static-background point clouds
# ---------------------------------------------------------------------

#### Load and inspect the point clouds ####
scene_rootfolder = "lidar_example_save\\provacammino\\"
scene = o3d.io.read_point_cloud(scene_rootfolder+"LidarType_CH128S1_2026-06-26-11-27-53-929_276.pcd")
background_rootfolder = "lidar_example_save\\background_provacammino\\"
background = o3d.io.read_point_cloud(background_rootfolder+"LidarType_CH128S1_2026-06-26-11-34-43-051_1.pcd")

# Choose the camera using the complete scene. !!
camera_params = select_camera_view(scene)

save_pointcloud_image(
    scene,
    os.path.join(image_folder, "01_scene.png"),
    camera_params
)

save_pointcloud_image(
    background,
    os.path.join(image_folder, "02_background.png"),
    camera_params
)

print("Scene points:", np.asarray(scene.points).shape[0])
print("Background points:", np.asarray(background.points).shape[0])

#visualization
o3d.visualization.draw_geometries([scene], window_name="Scene with Person")
o3d.visualization.draw_geometries([background], window_name="Background Only")

# ---------------------------------------------------------------------
# 2. Downsample point clouds to reduce density and computation
# ---------------------------------------------------------------------
#### Downsample the clouds ####
voxel_size = 0.01  # 1 cm

scene_ds = scene.voxel_down_sample(voxel_size)
background_ds = background.voxel_down_sample(voxel_size)

print("Downsampled scene points:", np.asarray(scene_ds.points).shape[0])
print("Downsampled background points:", np.asarray(background_ds.points).shape[0])

# ---------------------------------------------------------------------
# 4. Remove isolated foreground measurements
# ---------------------------------------------------------------------
#### Extract foreground points using background subtraction ###

## KD tree for the background
bg_tree = o3d.geometry.KDTreeFlann(background_ds)
scene_pts = np.asarray(scene_ds.points)

distance_threshold = 0.15  # 15 cm ## threshold to eliminate noise
foreground_mask = np.zeros(len(scene_pts), dtype=bool)

for i, pt in enumerate(scene_pts):
    k, idx, d2 = bg_tree.search_knn_vector_3d(pt, 1)

    if k == 0:
        foreground_mask[i] = True
        continue

    dist = np.sqrt(d2[0])
    foreground_mask[i] = dist > distance_threshold


foreground_indices = np.where(foreground_mask)[0]
foreground = scene_ds.select_by_index(foreground_indices.tolist())
o3d.visualization.draw_geometries([foreground], window_name="Foreground Scene with Person")

save_pointcloud_image(
    foreground,
    os.path.join(image_folder, "03_foreground.png"),
    camera_params
)

print("Foreground points:", np.asarray(foreground.points).shape[0])


### filter Outlier

foreground_clean, ind = foreground.remove_statistical_outlier(
    nb_neighbors=20,
    std_ratio=2.0
)
o3d.visualization.draw_geometries([foreground_clean], window_name="Foreground Scene with Person")

save_pointcloud_image(
    foreground_clean,
    os.path.join(image_folder, "04_foreground_filtered.png"),
    camera_params
)

print("Foreground after outlier removal:", np.asarray(foreground_clean.points).shape[0])

# ---------------------------------------------------------------------
# 5. Cluster foreground and retain the largest object
# ---------------------------------------------------------------------
#### DBSCAN clustering for largest cluster ####

labels = np.array(
    foreground_clean.cluster_dbscan(
        eps=0.08,
        min_points=80,
        print_progress=False
    )
)

valid_labels = labels[labels >= 0]

if len(valid_labels) > 0:
    cluster_ids, counts = np.unique(valid_labels, return_counts=True)
    largest_cluster = cluster_ids[np.argmax(counts)]

    person_indices = np.where(labels == largest_cluster)[0]
    person = foreground_clean.select_by_index(person_indices.tolist())
else:
    person = foreground_clean

o3d.visualization.draw_geometries([person], window_name="Foreground Scene with Person")

save_pointcloud_image(
    person,
    os.path.join(image_folder, "05_person.png"),
    camera_params
)

print("Unique cluster labels:", np.unique(labels))
