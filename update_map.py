import numpy as np
from projMapToFrame import Pointcloud

######## Written BY : MAYANK GUPTA (As part of 16-833, Spring 2020, CMU) (Adapted from Matlab code from HW4)

def avgProjMapWithInputData(proj_map, input_data, h, w, is_use):
    input_points = input_data.points
    input_colors = input_data.colors
    input_normals = input_data.normals
    alpha = input_data.ccounts

    proj_points = proj_map.points
    proj_colors = proj_map.colors
    proj_normals = proj_map.normals
    proj_ccounts = proj_map.ccounts

    is_use_3 = np.transpose(np.tile(is_use,(3,1,1)),(1,2,0))
    
    proj_points_ = np.reshape(proj_points[is_use_3],(-1,3))
    input_points_= np.reshape(input_points[is_use_3],(-1,3))
    proj_normals_ = np.reshape(proj_normals[is_use_3],(-1,3))
    input_normals_ = np.reshape(input_normals[is_use_3],(-1,3))
    proj_ccounts_ = proj_ccounts[is_use][:,None]
    alpha_ = alpha[is_use][:,None]
    
    proj_points[is_use,:] = (proj_ccounts_ * proj_points_ + alpha_ * input_points_) / (proj_ccounts_ + alpha_)
    proj_colors[is_use_3] = input_colors[is_use_3]
    proj_normals[is_use,:] = (proj_ccounts_ * proj_normals_ + alpha_ * input_normals_) / (proj_ccounts_ + alpha_)
    proj_ccounts[is_use] = np.squeeze(proj_ccounts_ + alpha_)
    
    updated_map = Pointcloud(proj_points, proj_colors, proj_normals, proj_ccounts)
    non_zero_mask = (np.sum(proj_points == 0,2) < 3)
    updated_map.set_valid_mask(non_zero_mask)
    return updated_map

def updateFusionMapWithProjMap(fusion_map, updated_map, h, w, proj_flag):
    input_valid_mask = updated_map.valid_mask
    old_flag = np.logical_not(proj_flag)
    
    old_flag_3 = np.transpose(np.tile(old_flag,(3,1)))
    old_points = np.reshape(fusion_map.points[old_flag_3], (-1,3))
    old_colors = np.reshape(fusion_map.colors[old_flag_3], (-1,3))
    old_normals = np.reshape(fusion_map.normals[old_flag_3], (-1,3))
    old_ccounts = fusion_map.ccounts[old_flag]
    
    map_points = np.vstack((old_points, np.reshape(updated_map.points[input_valid_mask,:], (-1,3))))
    map_colors = np.vstack((old_colors, np.reshape(updated_map.colors[input_valid_mask,:], (-1,3))))
    map_normals = np.vstack((old_normals, np.reshape(updated_map.normals[input_valid_mask,:], (-1,3))))
    map_ccounts = np.concatenate((old_ccounts, updated_map.ccounts[input_valid_mask].flatten()))
    
    fusion_map = Pointcloud(map_points, map_colors, map_normals, map_ccounts)
    
    return fusion_map
