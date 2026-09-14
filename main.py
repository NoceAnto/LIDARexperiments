import os
import open3d as o3d
import numpy as np
from functions import *

"""
Visualize a recorded LiDAR point-cloud sequence using its acquisition timing.

The script:
1. finds and sorts all PCD frames in the selected directory;
2. extracts frame timestamps from the filenames;
3. estimates the acquisition frame rate from the sampling intervals;
4. displays the point-cloud sequence using Open3D.

This script is intended primarily for checking the temporal behavior and
visual quality of recorded LiDAR sequences.
"""

def intensity_to_color(intensity):
    # Normalize intensity → [0,1]
    intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)

    # Map intensity to grayscale color
    colors = np.stack([intensity_norm] * 3, axis=1)
    return colors

#192.168.1.102

# --- Load all PCD files (sorted as frames) ---
# Load the recorded LiDAR frames in chronological order.

pcd_files = sorted(glob.glob("lidar_example_save/provacammino/*.pcd"))

# Extract acquisition times encoded in the filenames.
df_info = read_files_extract_info(pcd_files)
sampling_time = np.array(df_info['sampling_time'],dtype=np.float32)

# Estimate the acquisition frame rate from the median frame interval.
fps = int(round(1/np.median(np.diff(sampling_time))))

# Preview the first ten recorded frames.
read_vis_pcd(pcd_files[0:10],fps,"renderedimages")

