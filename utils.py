#!/usr/bin/python
import math
import numpy as np
import numpy.ma as ma
from projMapToFrame import pctransform

### By SWAPNIL DAS

focalLength = 525.0
centerX = 319.5
centerY = 239.5
scalingFactor = 5000.0

def isInputSimilarToProjNormals(input_normals, proj_normals, dot_th):
	normal_dots = np.sum(np.multiply(input_normals, proj_normals), axis=2)
	return normal_dots > dot_th

def isInputCloseToProjPoints(input_points, proj_points, dist2_th):
	diff_pts = proj_points - input_points
	dist2_pts = np.sum(np.square(diff_pts), axis=2)
	return dist2_pts < dist2_th

def isFirst(proj_points, h, w):
	return np.sum(np.square(proj_points), axis=2) == np.zeros((h, w))

def isUsableInputPoints(is_s, is_c, is_f):
	return np.logical_or(is_f, np.logical_and(is_c, is_s))

def avgProjMapWithInputData(proj_map, trans_points, trans_normals, alpha, h, w, is_use):
	input_mask_data = ma.array(input_data, is_use)
	result_data = (ccounts * proj_data + alpha * input_mask_data)/(ccounts + alpha)
	result_normals = (ccounts * proj_normals + alpha * input_normals)/(ccounts + alpha)
	ccounts = ccounts + alpha
	return result_data, result_normals




