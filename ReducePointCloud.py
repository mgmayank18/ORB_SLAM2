import numpy as np
import open3d as o3d
import argparse

######## Written BY : MAYANK GUPTA (As part of 16-833, Spring 2020, CMU)

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="Input file path")
args=parser.parse_args()

filepath=args.path

def ReducePointCloud(PC):
    pcd = o3d.io.read_point_cloud(PC) # Read the point cloud
    print(pcd)
    voxel_size = 0.03
    print("Voxel Size :",voxel_size)
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(downpcd)
    o3d.visualization.draw_geometries([downpcd])

print(filepath)
ReducePointCloud(filepath)
print("Done.")
