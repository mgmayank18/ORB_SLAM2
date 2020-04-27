#!/usr/bin/python
import math
import numpy as np
from projMapToFrame import pctransform

######## Written BY : SWAPNIL DAS (As part of 16-833, Spring 2020, CMU) (Adapted from Matlab code from HW4)

focalLength = 525.0
centerX = 319.5
centerY = 239.5
scalingFactor = 5000.0

def isInputSimilarToProjNormals(input_normals, proj_normals, dot_th):
	normal_dots = np.sum(np.multiply(input_normals, proj_normals), axis=2)
	return normal_dots > dot_th

def isInputCloseToProjPoints(input_points, proj_points, dist2_th):
	diff_pts = proj_points - input_points
	null_mask = (proj_points - input_points) == 0
	diff_pts[null_mask] = 1
	dist2_pts = np.sum(np.square(diff_pts), axis=2)
	return dist2_pts < dist2_th

def isFirst(proj_points, h, w):
	return np.sum(np.square(proj_points), axis=2) == np.zeros((h, w))

def isUsableInputPoints(is_c, is_s, is_f, input_valid_mask):
	return np.logical_or(np.logical_and(is_f,input_valid_mask), np.logical_and(is_c, is_s))





