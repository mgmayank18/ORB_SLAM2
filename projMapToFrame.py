import numpy as np

class Pointcloud:
    def __init__(self, points, colors, normals, ccounts=None, sigma=0.6):
        self.points = points
        self.colors = colors
        self.normals = normals
        self.valid_mask = None
        self.h = None
        self.w = None
        self.num_points = len(points)
        if ccounts is None:
            flat_points = np.reshape(points,(-1,3))
            ccounts = np.exp(-1*np.divide(np.sum(np.square(flat_points),1),2*(sigma**2)))
            ccounts = np.reshape(ccounts,(points.shape[0],points.shape[1]))
            self.ccounts = ccounts
        else:
            self.ccounts = ccounts

    def __len__(self):
        return self.num_points

    def set_h_w(self,h,w):
        self.h = h
        self.w = w
    
    def set_valid_mask(self,valid_mask):
        self.valid_mask = valid_mask

def pctransform(fusion_map_points, R_inv, t_inv):#fusion_map
    transformed_points = np.transpose(np.add(np.matmul(R_inv, np.transpose(fusion_map_points)), t_inv))
    return transformed_points
    

def proj(fusion_map, h, w, pose, cam_param):
    fx = cam_param[0]
    fy = cam_param[1]
    cx = cam_param[2]
    cy = cam_param[3]

    K = np.array([[fx, 0, cx],[0, fy, cy],[0, 0, 1]])

    R_inv = np.transpose(pose[0:3,0:3])
    t_inv = np.transpose([-1*pose[0:3,3]])

    pose_inv = np.hstack((R_inv,t_inv))
    pose_inv = np.vstack((pose_inv,np.array([0,0,0,1])))
    
    fusion_map_cam_frame_points = pctransform(fusion_map.points, R_inv, t_inv)
    
    proj_flag = (fusion_map_cam_frame_points[:,2] > 0)
    valid_mask = np.transpose(np.tile(proj_flag,(3,1)))
    valid_points = fusion_map_cam_frame_points * valid_mask
    
    x = valid_points[valid_mask]
    projected_points = np.matmul(valid_points,np.transpose(K))
    projected_points = np.divide(projected_points, np.transpose([projected_points[:,2]]))
    np.putmask(valid_points, valid_mask, projected_points)
    
    H = valid_points[:,0]
    W = valid_points[:,1]

    in_frame_mask = np.logical_and(np.logical_and(np.logical_and(H > 0, H < h), W > 0), W < w)
    
    proj_flag = np.logical_and(proj_flag, in_frame_mask)
    valid_mask = np.transpose(np.tile(proj_flag,(3,1)))
    valid_points = np.round(valid_points)

    proj_points_ = np.reshape(fusion_map.points[valid_mask],(-1,3))
    proj_colors_ = np.reshape(fusion_map.colors[valid_mask],(-1,3))
    proj_normals_ = np.reshape(fusion_map.normals[valid_mask],(-1,3))
    proj_ccounts_ = fusion_map.ccounts[proj_flag]
    
    proj_points = np.zeros((h*w,3))
    proj_colors = np.zeros((h*w,3))
    proj_normals = np.zeros((h*w,3))
    proj_ccounts = np.zeros((h*w))
    
    loc = np.reshape(valid_points[valid_mask],(-1,3))
    ind = (1+loc[:,1]*h+loc[:,2]).astype(np.uint8)
    
    proj_points[ind, :] = proj_points_
    proj_colors[ind, :] = proj_colors_
    proj_normals[ind, :] = proj_normals_
    proj_ccounts[ind] = proj_ccounts_
    
    proj_points = np.reshape(proj_points, [h, w, 3])
    proj_colors = np.reshape(proj_colors, [h, w, 3])
    proj_normals = np.reshape(proj_normals, [h, w, 3])
    proj_ccounts = np.reshape(proj_ccounts, [h, w])
    
    proj_map = Pointcloud(proj_points, proj_colors, proj_normals, proj_ccounts)
    
    return proj_map, proj_flag
