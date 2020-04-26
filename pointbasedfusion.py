import math
import numpy as np
from projMapToFrame import proj, Pointcloud
from utils import *

def fuse(fusion_map, input_data, pose, cam_param, sigma, ds_ratio, t):
    #Parameters (square of point distance threshold) ====
    dist_th = 0.05 #5cm
    dist2_th = dist_th**2
    dot_th = math.cos(20*math.pi/180)
    
    proj_map, proj_flag = proj(fusion_map, input_data.h, input_data.w, pose, cam_param)
    proj_points = proj_map.points
    proj_normals = proj_map.normals

    trans_points = input_data.points
    trans_normals = input_data.normals

    is_close = isInputCloseToProjPoints(trans_points, proj_points, dist2_th)
    is_similar = isInputSimilarToProjNormals(trans_normals, proj_normals, dot_th)
    is_first = isFirst(proj_points, input_data.h, input_data.w)
    is_use = isUsableInputPoints(is_close, is_similar, is_first)
    alpha = input_data.ccounts
    
    updated_map = avgProjMapWithInputData(proj_map, trans_points, trans_normals, alpha, h, w, is_use)
    
    #next_ref_data = getNextRefData(updated_map, ds_ratio)
    
    fusion_map = updateFusionMapWithProjMap(fusion_map, updated_map, h, w, proj_flag)
    return fusion_map