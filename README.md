# Dense Reconstruction using ORB-SLAM2 with point based fusion
**Authors:** [Mayank Gupta](https://www.linkedin.com/in/mayankguptaiitd/), [Swapnil Das](http://github.com/swapnil-das/)
This work was done as part of 16-833 as the final course project.

# 1. Prerequisites
Our code is built on [ORB SLAM2](https://github.com/raulmur/ORB_SLAM2), which can be installed by following the linked instructions along with it's dependencies. We also use a modified version of the [RGBD tools](https://vision.in.tum.de/data/datasets/rgbd-dataset/tools) provided by TUM which is linked [here](https://github.com/rFalque/ORB_SLAM2/blob/master/Examples/RGB-D/scripts/generate_registered_pointcloud.py) and is authored by [Raphael Falque](https://github.com/rFalque)

# 2. Usage

We have tested our code with the TUM RGBD dataset to generate dense pointclouds with point based fusion. The fusion reduces the size of the point cloud and gives a desirable representation. The parameters for fusion can be changed in pointbasedfusion.py.

## Instructions for TUM Dataset

1. Download a sequence from http://vision.in.tum.de/data/datasets/rgbd-dataset/download and uncompress it.

2. Associate RGB images and depth images using the python script [associate.py](http://vision.in.tum.de/data/datasets/rgbd-dataset/tools). We already provide associations for some of the sequences in *Examples/RGB-D/associations/*. You can generate your own associations file executing:

  ```
  python associate.py PATH_TO_SEQUENCE/rgb.txt PATH_TO_SEQUENCE/depth.txt > associations.txt
  ```

3. Execute the following command. Change `TUMX.yaml` to TUM1.yaml,TUM2.yaml or TUM3.yaml for freiburg1, freiburg2 and freiburg3 sequences respectively. Change `PATH_TO_SEQUENCE_FOLDER`to the uncompressed sequence folder. Change `ASSOCIATIONS_FILE` to the path to the corresponding associations file.

  ```
  ./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUMX.yaml PATH_TO_SEQUENCE_FOLDER ASSOCIATIONS_FILE
  ```
4. Once the association file has been made, a fusion map can be made by running the following command from *inside the dataset folder*.

  ```
  ./PATH_TO_/fusion_map.py --nth 1 --downsample 5 ./rgb.txt ./depth.txt PATH_TO_TRAJECTORY ./output_map.ply
  ```
  Here, the nth command skips frames, so 1 means no skipping, downsample can reduce the number of points that are read and make the final pointcloud of a more manageable size. Both Camera and Keyframe trajectories can be used. Using Keyframe trajectory works just like frame skipping.

5. The final pointcloud can be visualized using the following command.

```
./PATH_TO_/ShowPointCloud.py --path PATH_TO_PLY_FILE
```

Press h for help with commands in open3D viewer.
