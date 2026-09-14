import open3d as o3d
import numpy as np
import pandas as pd
import glob
import time
import datetime
import os
"""
Utility functions for LiDAR point-cloud processing and visualization.

This module provides functions to:
- convert LiDAR CSV data into Open3D point clouds;
- visualize PCD/CSV point-cloud sequences;
- extract acquisition timestamps and estimate frame timing;
- perform foreground extraction by subtracting a static background;
- remove isolated foreground points and optionally apply DBSCAN clustering;
- project a 3D point cloud into a 2D depth map;
- process and save a sequence of foreground point clouds.

The implementation uses the Open3D legacy geometry API
(open3d.geometry.PointCloud).
"""
def csv_to_pcd(csv_path):
    """
       Convert one LiDAR CSV frame into an Open3D point cloud.

       XYZ coordinates are used as point positions. LiDAR intensity is
       normalized to [0, 1] and represented as a grayscale point color.

       Returns
       -------
       pcd : open3d.geometry.PointCloud
           Point cloud containing XYZ coordinates and grayscale intensity.
       time_vector : np.ndarray
           Per-point timestamps expressed in seconds.
    """

    data = pd.read_csv(csv_path, delimiter=",")

    timestamp_s = np.array(data["Timestamp_s"])
    timestamp_ns =  np.array(data["Timestamp_ns"]/ 1e9) # sec

    time_vector = timestamp_s + timestamp_ns

    x = np.expand_dims(np.array(data["Points_X"]),axis=1)
    y = np.expand_dims(np.array(data["Points_Y"]),axis=1)
    z = np.expand_dims(np.array(data["Points_Z"]),axis=1)
    xyz = np.hstack((x,y,z))

    intensity = np.array(data["Intensity1"])

    # Normalize intensity → [0,1]
    intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)

    # Map intensity to grayscale color
    colors = np.stack([intensity_norm]*3, axis=1)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd, time_vector

def vis_pcd(pcd_all,fps=10):
    """
        Load and visualize a sequence of PCD files at the specified frame rate.
    """
    # --- Initialize visualizer ---
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PCD Video", width=1280, height=720)
    # --- Load first frame ---
    pcd = pcd_all[0]
    vis.add_geometry(pcd)

    # Optional: improve rendering
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.asarray([0, 0, 0])

    ctr = vis.get_view_control()

    # --- Main playback loop ---

    for new_pcd in pcd_all:

        # Update geometry (important: reuse object)
        pcd.points = new_pcd.points
        pcd.colors = new_pcd.colors

        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        step = 1.0 / fps
        time.sleep(step)

    # Keep window open
    vis.run()
    vis.destroy_window()
def read_vis_pcd(pcd_files,fps,save_folder=None):
    """
        Visualize a PCD sequence and optionally save each rendered frame.

        Extract frame numbers and acquisition timestamps from LiDAR filenames.

        The timestamps are converted to datetime objects and relative sampling
        times are calculated with respect to the first frame.
    """

    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)

    # --- Initialize visualizer ---
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PCD Video", width=1280, height=720)

    # --- Load first frame ---
    pcd = o3d.io.read_point_cloud(pcd_files[0])
    vis.add_geometry(pcd)

    # Optional: improve rendering
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.background_color = np.asarray([0, 0, 0])

    ctr = vis.get_view_control()

    # --- Main playback loop ---

    for i,file in enumerate(pcd_files):

        new_pcd = o3d.io.read_point_cloud(file)

        # Update geometry (important: reuse object)
        pcd.points = new_pcd.points
        pcd.colors = new_pcd.colors

        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        # Save the current Open3D rendering.
        if save_folder is not None:
            image_path = os.path.join(
                save_folder,
                f"frame_{i:04d}.png"
            )
            vis.capture_screen_image(
                image_path,
                do_render=True
            )


        step = 1.0 / fps
        time.sleep(step)

    # Keep window open
    vis.run()
    vis.destroy_window()

def read_files_extract_info(files):
    dt_all = []
    frames = []
    sampling_time = []

    for file in files:
        head, tail = os.path.split(file)
        parts = tail[18:-4].split('-')
        a, b =parts[-1].split("_")
        frame = b
        parts[-1] = a
        dt = datetime.datetime(
            int(parts[0]),  # year
            int(parts[1]),  # month
            int(parts[2]),  # day
            int(parts[3]),  # hour
            int(parts[4]),  # minute
            int(parts[5]),  # second
            int(parts[6]) * 1000  # ms → µs
        )
        if frame == '1':
            first_dt = dt
            sampling_time.append(0.0)
        else:
            step = dt - first_dt
            sampling_time.append(step.total_seconds())
        frames.append(frame)
        dt_all.append(dt)

    df_info = pd.DataFrame({
        "filenames": files,
        "frames": frames,
        "datetimes": dt_all,
        "sampling_time":sampling_time
    })

    return df_info

def visualize_csv_sequence(folder_path, fps=10):
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))

    # Initialize visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window("CSV PointCloud Video", 1280, 720)

    # Load first frame
    pcd,tv = csv_to_pcd(csv_files[0])
    vis.add_geometry(pcd)

    render_option = vis.get_render_option()
    render_option.point_size = 2.0

    for i, file in enumerate(csv_files):
        new_pcd,tv = csv_to_pcd(file)

        # Update geometry (FAST)
        pcd.points = new_pcd.points
        pcd.colors = new_pcd.colors

        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        time.sleep(1.0 / fps)

    vis.run()
    vis.destroy_window()

def visualize_pointcloud_with_timestamps(
    csv_files,
    window=0.05,
):
    """
    Visualize multiple CSV files using per-point timestamps.

    Parameters:
    - csv_files: list of CSV paths
    - time_column: index of timestamp column
    - fps: playback speed
    - window: time window (seconds) shown at each frame
    - colormap: matplotlib colormap
    """

    # -----------------------------
    # 1. Load all data
    # -----------------------------
    all_xyz = []
    all_intensity = []
    all_timestamps = []

    for f in sorted(csv_files):

        data = pd.read_csv(f, delimiter=",")

        timestamp_s = np.array(data["Timestamp_s"])
        timestamp_ns = np.array(data["Timestamp_ns"] / 1e9)  # sec

        timestamps = timestamp_s + timestamp_ns

        x = np.expand_dims(np.array(data["Points_X"]), axis=1)
        y = np.expand_dims(np.array(data["Points_Y"]), axis=1)
        z = np.expand_dims(np.array(data["Points_Z"]), axis=1)
        xyz = np.hstack((x, y, z))

        intensity = np.array(data["Intensity1"])

        all_xyz.append(xyz)
        all_intensity.append(intensity)
        all_timestamps.append(timestamps)

    xyz = np.vstack(all_xyz)
    intensity = np.hstack(all_intensity)
    timestamps = np.hstack(all_timestamps)

    # -----------------------------
    # 2. Sort by timestamp
    # -----------------------------
    idx = np.argsort(timestamps)
    xyz = xyz[idx]
    intensity = intensity[idx]
    timestamps = timestamps[idx]

    # Normalize time
    timestamps -= timestamps.min()
    print("Timestamp min/max:", timestamps.min(), timestamps.max())


    # -----------------------------
    # 4. Open3D setup
    # -----------------------------
    vis = o3d.visualization.Visualizer()
    vis.create_window("Timestamp-based LiDAR Visualization", 1280, 720)


    mask = (timestamps >= 0) & (timestamps < 0 + window)
    pts = xyz[mask]
    print(np.shape(pts))

    intensity_mask = intensity[mask]
    # Normalize intensity → [0,1]
    intensity_norm = (intensity_mask - intensity_mask.min()) / (intensity_mask.max() - intensity_mask.min() + 1e-8)

    # Map intensity to grayscale color
    cols = np.stack([intensity_norm] * 3, axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    vis.add_geometry(pcd)

    render_option = vis.get_render_option()
    render_option.point_size = 2.0

    # -----------------------------
    # 5. Playback loop
    # -----------------------------
    t_current = 0.0
    t_max = timestamps.max()
    step = window

    while t_current < t_max:

        mask = (timestamps >= t_current) & (timestamps < t_current + window)
        pts = xyz[mask]
        print(np.shape(pts))

        intensity_mask = intensity[mask]
        # Normalize intensity → [0,1]
        intensity_norm = (intensity_mask - intensity_mask.min()) / (intensity_mask.max() - intensity_mask.min() + 1e-8)

        # Map intensity to grayscale color
        cols = np.stack([intensity_norm] * 3, axis=1)

        if len(pts) > 0:
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(cols)

            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

        time.sleep(step)
        t_current += window

    vis.run()
    vis.destroy_window()

def extract_person_pc(scenepath,backgroundpath,visualization=False,dbscan=False):
    """
        Extract foreground points from a LiDAR scene using a static background.

        For each scene point, the nearest point in the background point cloud
        is found using a KD-tree. Scene points whose nearest-background
        distance exceeds a specified threshold are classified as foreground.

        Statistical outlier removal is subsequently applied to suppress
        isolated measurements. Optionally, DBSCAN clustering is used and
        the largest foreground cluster is retained as the person.

        Parameters
        ----------
        scenepath : str
            Path to the scene point cloud containing foreground objects.
        backgroundpath : str
            Path to the reference background point cloud.
        visualization : bool
            If True, display intermediate processing results.
        dbscan : bool
            If True, apply DBSCAN and retain the largest cluster.

        Returns
        -------
        person : open3d.geometry.PointCloud
            Extracted and filtered foreground point cloud.
        """


    #### Load and inspect the point clouds ####
    scene = o3d.io.read_point_cloud(scenepath)
    background = o3d.io.read_point_cloud(backgroundpath)

    #print("Scene points:", np.asarray(scene.points).shape[0])
    #print("Background points:", np.asarray(background.points).shape[0])

    # visualization
    if visualization:
        #o3d.visualization.draw_geometries([scene], window_name="Scene with Person")
        #o3d.visualization.draw_geometries([background], window_name="Background Only")
        print('e')

    #### Downsample the clouds ####
    #voxel_size = 0.01  # 1 cm

    scene_ds = scene#.voxel_down_sample(voxel_size)
    background_ds = background#.voxel_down_sample(voxel_size)

    #print("Downsampled scene points:", np.asarray(scene_ds.points).shape[0])
    #print("Downsampled background points:", np.asarray(background_ds.points).shape[0])

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
    if visualization:
        o3d.visualization.draw_geometries([foreground], window_name="Foreground Scene with Person")

    #print("Foreground points:", np.asarray(foreground.points).shape[0])

    ### filter Outlier

    foreground_clean, ind = foreground.remove_statistical_outlier(
        nb_neighbors=10,
        std_ratio=2.0
    )
    if visualization:
        o3d.visualization.draw_geometries([foreground_clean], window_name="Foreground Scene with Person")

    #print("Foreground after outlier removal:", np.asarray(foreground_clean.points).shape[0])

    #### DBSCAN clustering for largest cluster ####
    if dbscan:
        labels = np.array(
            foreground_clean.cluster_dbscan(
                eps=0.08,
                min_points=40,
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
        if visualization:
            o3d.visualization.draw_geometries([person], window_name="Foreground Scene with Person")
    else:
        person = foreground_clean

    #print("Unique cluster labels:", np.unique(labels))

    return person


def pcd_to_depth_map_manual(
    pcd,
    width=640,
    height=480,
    fx=100.0,
    fy=100.0,
    extrinsic=None,
    fill_value=0.0
):
    """
    Project a 3D point cloud onto a 2D depth image using a pinhole model.

    Points are first expressed in the assumed camera coordinate system.
    The current implementation remaps the LiDAR coordinates as:

        camera_x = point_x
        camera_y = -point_z
        camera_z = point_y - 3

    Valid points are projected using the focal lengths fx and fy.
    If multiple 3D points project to the same pixel, the nearest point
    is retained using a z-buffer.

    Notes
    -----
    The coordinate remapping and the '-3' translation are specific to the
    current sensor/camera geometry and should be calibrated or parameterized
    if the acquisition setup changes.
    -----
    Convert a .pcd point cloud to a 2D depth map using pinhole projection.

    Parameters
    ----------
    pcd_path : str
        Path to the .pcd file.
    width, height : int
        Output depth map size.
    fx, fy, cx, cy : float
        Camera intrinsics.
    extrinsic : np.ndarray or None
        4x4 transform to move points into camera coordinates.
        If None, assumes the point cloud is already in camera coordinates.
    fill_value : float
        Value for missing pixels.

    Returns
    -------
    depth_map : np.ndarray of shape (height, width), dtype float32
    """

    cx = (width - 1) / 2
    cy = (height - 1) / 2

    points = np.asarray(pcd.points, dtype=np.float64)

    if points.shape[0] == 0:
        return np.full((height, width), fill_value, dtype=np.float32)

    # Transform to camera coordinates if needed
    if extrinsic is not None:
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])   # (N,4)
        points = (extrinsic @ points_h.T).T[:, :3]

    x = points[:, 0]
    y = -points[:, 2]
    z = points[:, 1]-3

    # Keep only points in front of the camera
    valid = z > 0
    x, y, z = x[valid], y[valid], z[valid]

    # Project to image plane
    u = np.round(fx * x / z + cx).astype(np.int32)
    v = np.round(fy * y / z + cy).astype(np.int32)

    # Keep only pixels inside image bounds
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[valid], v[valid], z[valid]

    # Initialize depth map with +inf for z-buffering
    depth_map = np.full((height, width), np.inf, dtype=np.float32)

    # Z-buffer: nearest point wins
    for uu, vv, zz in zip(u, v, z):
        if zz < depth_map[vv, uu]:
            depth_map[vv, uu] = zz

    depth_map[np.isinf(depth_map)] = fill_value
    return depth_map


def extract_and_save_person_pc_sequence(scene_rootfolder,backgroundpath,foldersave):
    person_per_frame = []
    i = 0
    os.makedirs(foldersave, exist_ok=True)

    for scenefile in os.listdir(scene_rootfolder):
        scenepath = scene_rootfolder + "\\" + scenefile
        person = extract_person_pc(scenepath, backgroundpath, visualization=False)
        person_per_frame.append(person)
        # o3d.visualization.draw_geometries([person], window_name="Foreground Scene with Person")
        print("Processing Frame "+str(i))

        o3d.io.write_point_cloud(foldersave+"\\frame" + str(i) + ".pcd", person)

        i = i + 1