import cv2
import os
import numpy as np

Associations_file = "./rgbd_dataset_freiburg1_room/associations.txt"
Camera_poses = "./CameraTrajectory.txt"
rel_path = "./rgbd_dataset_freiburg1_room"

with open(Associations_file,'r') as f:
    total = 0
    for line in f:
        w = line.strip().split(" ")
        RGB = cv2.imread(os.path.join(rel_path,w[1]))
        D = cv2.imread(os.path.join(rel_path,w[3]))
        total+=np.count_nonzero(D)
    print(total)