import numpy as np

from functions import *
import os
import open3d as o3d
import cv2

"""
Process a LiDAR foreground sequence and generate 2D depth-map frames.

The script can perform two sequential processing stages:

1. Foreground extraction
   Each recorded LiDAR frame is compared with a static background using
   `extract_person_pc()`, and the resulting foreground point cloud is
   saved as an individual PCD file.

2. Depth-map generation
   The processed foreground PCD files are loaded sequentially and projected
   onto a 2D image plane using `pcd_to_depth_map_manual()`. Each depth map
   is normalized for visualization and displayed as a grayscale image.

The resulting depth frames are stacked into a NumPy array and saved for
subsequent image- or video-based processing.
"""

scene_rootfolder = "lidar_example_save\\provacammino"
backgroundpath = "lidar_example_save\\background_provacammino\\LidarType_CH128S1_2026-06-26-11-34-43-051_1.pcd"

processed_scene_folder = "provacammino"
os.makedirs(processed_scene_folder,exist_ok=True)

# Extract the foreground from every LiDAR frame and save it as a PCD sequence.
'''i = 0
for file in os.listdir(scene_rootfolder):
    scenepath = scene_rootfolder+"\\"+file
    person = extract_person_pc(scenepath,backgroundpath,
                               visualization=False,dbscan=False)

    # Save to PCD file
    o3d.io.write_point_cloud(processed_scene_folder+"\\frame"+str(i)+".pcd", person)
    print(i)
    i = i + 1'''


# Convert each extracted foreground point cloud into a 2D depth-map frame.
person_per_frame = []
alldepth_video = []
output_folder = "depth_images"
os.makedirs(output_folder, exist_ok=True)

for i in range(0,len(os.listdir(processed_scene_folder))):
    person = o3d.io.read_point_cloud(processed_scene_folder+"\\frame"+str(i)+".pcd")
    person_per_frame.append(person)
    persondepth = pcd_to_depth_map_manual(person)
    print('we')

    depth = persondepth.astype(np.float32)

    # Mask valid depths (ignore zeros if used as missing values)
    valid_mask = depth > 0

    if np.any(valid_mask):
        d_min = np.min(depth[valid_mask])
        d_max = np.max(depth[valid_mask])

        # Normalize only valid values to 0-255
        depth_norm = np.zeros_like(depth, dtype=np.uint8)
        depth_norm[valid_mask] = (
                255 * (depth[valid_mask] - d_min) / max(d_max - d_min, 1e-8)
        ).astype(np.uint8)



        # Save normalized depth map as an 8-bit grayscale PNG.
        cv2.imwrite(
            os.path.join(output_folder, f"frame_{i:04d}.png"),
            depth_norm
        )
    else:
        depth_norm = np.zeros_like(depth, dtype=np.uint8)
    print(i)
    # Show grayscale depth map
    cv2.imshow('window_name', depth_norm)

    key = cv2.waitKey(33) & 0xFF
    if key == ord('q'):
        break

    alldepth_video.append(np.expand_dims(depth_norm,axis=2))
cv2.destroyAllWindows()

alldepth_video = np.concat(alldepth_video,axis=2)
np.save("alldepth_video.npy",alldepth_video)
vis_pcd(person_per_frame,fps=10)