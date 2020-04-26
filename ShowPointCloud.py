import numpy as np
import open3d as o3d

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="Input file path")
args=parser.parse_args()

def ShowPointCloud(PC):
    cloud = o3d.io.read_point_cloud(PC) # Read the point cloud
    print(cloud)
    o3d.visualization.draw_geometries([cloud]) # Visualize the point cloud     

filepath=args.path
print(filepath)
ShowPointCloud(filepath)
print("Done.")

if __name__ == "__main__":
    ShowPointCloud("./rgbd_dataset_freiburg1_room/KeyFrames_Downsample_5.ply")