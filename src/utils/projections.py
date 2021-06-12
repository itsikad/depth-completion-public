from typing import Union, List, Tuple, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

import numpy as np

from sklearn.neighbors import KDTree

import cv2


def transform_intrinsics(
    intrinsics: np.ndarray,
    orig_img_size: Tuple,
    cropped_img_size: Tuple,
    crop_type: str = 'bottom'
) -> np.ndarray:

    """
    Transform intrinsics matrix to compensate for cropping transofrm.

    Examples:
    Assuming original image is (375,1242)
        1. CenterCrop(352,1216):
            Compensate for one-sided width reduction of(1242-1216)/2 = 13 pixels
            Compensate for one-sided height reudction of (375-352)/2 = 11.5 pixels
        
        2. BottomCrop(352,1216):
            Compensate for one-sided width reduction of(1242-1216)/2 = 13 pixels
            Compensate for height reudction of 375-352 = 23 pixels
    
    Arguments:
        intrinsics : (3,3) intrinsics matrix

        orig_img_size : dimensions of the original image

        cropped_img_size : dimensions of the cropped image
        
        crop_type: should be either 'bottom' or 'center'
        
    Return:
        intrinsics : updated camera intrinsics matrix
    """
    
    dw = (orig_img_size[0] - cropped_img_size[0]) / 2.
    dh = orig_img_size[1] - cropped_img_size[1]
    dh = dh / 2. if crop_type == 'center' else dh

    # Note: compensate for center crop of the image
    #       that changes the optical centers, but not focal lengths
    intrinsics[0, 2] = intrinsics[0,2] - dw
    intrinsics[1, 2] = intrinsics[1,2] - dh

    return intrinsics


def scale_intrinsics(
    intrinsics: Union[Tensor, np.ndarray],
    old_dims: Tuple[int,int],
    new_dims: Tuple[int,int]
) -> Tensor:
    
    """
    Scale intrinsics matrix 
    """

    intrinsics_new = intrinsics.detach().clone()

    ratio_u = float(new_dims[1]) / old_dims[1]
    ratio_v = float(new_dims[0]) / old_dims[0]

    intrinsics_new[:,0,:] = intrinsics_new[:,0,:] * ratio_u
    intrinsics_new[:,1,:] = intrinsics_new[:,1,:] * ratio_v

    return intrinsics_new


def expand_intrinsics(
    intrinsics: Union[Tensor, np.ndarray]
) -> Tuple:

    """
    Extracts u_0, v_0, f_u, f_v from a 3x3 intransics matrix.

    Argumetns:

        intrinsics :  (3,3) for single intrinsics matrix or (B,3,3) for batched intrinsics matrices.

    Return:
        Camera intrinsics tuple (u_0, v_0, f_u, f_v), scalars for a single image and vectors for a batch.

    """

    u_0 = intrinsics[...,0,2]
    v_0 = intrinsics[...,1,2]
    f_u = intrinsics[...,0,0]
    f_v = intrinsics[...,1,1]

    return u_0, v_0, f_u, f_v


def sparse_back_projection(
    sdepth: Union[Tensor, np.ndarray],
    intrinsics: Union[Tensor, np.ndarray]
) -> Union[Tensor, np.ndarray]:

    """
    Backprojects a 2D sparse depth map to a 3D pointclouds.

    Arguments:
        depth : a (H,W) or (1,H,W) or (B,1,H,W) tensor/array containg the depth map,
               Expects invalid entries to be zero.
        
        intrinsics : a (B,3,3) intrinsics matrix (tensor/array)
    
    Return:
        pc_xyz : an (N,3) or (B,N,3) tensor/array (same as input type), points coordinate in the camera reference system

        pc_idx : an (N,) or (B,N,) tensor/array (same as input type) of linear coordinates per point, 
                 values in range [0,HW-1] to allow projection

    Examples:
        1. Collect features from dense feature map:
            pointcloud_features = torch.gather(input=features_in.view(B,C,-1), dim=2, index=pc_idx.repeat(1,C,1))  # (B,C,N)

        2. Project pointcloud features back to feature_map
            features_out = torch.zeros_like(features_in.view(B,C,-1)).scatter(dim=2, index=pc_idx.repeat(1,C,1), src= pointcloud_features).reshape(B,C,H,W)  # (B,C,H,W)
    """

    if type(sdepth) != type(intrinsics):
        raise TypeError('sparse depth and intrinsics inputs must be of the same type (np.ndarray or torch.Tensor)')

    # Frame width
    sdepth = sdepth.squeeze()  # validate depth is (H,W)
    W = sdepth.shape[-1]

    # Extract valid depth pixels U,V (pixel) coordinates
    # and converts to running index in range [0,HW-1]
    if isinstance(sdepth, torch.Tensor):
        pc_uv = torch.nonzero(sdepth, as_tuple=False)  # (N,2)
    else:
        # numpy array
        pc_uv = np.transpose(np.nonzero(sdepth))  # (N,2)

    pc_idx = pc_uv[:,0] * W + pc_uv[:,1]  # (N,)
    z = sdepth.flatten()[pc_idx]  # (N,)

    # Extract intrinsics parameters
    # U,V are horizontal and vertical pixel coordinates 
    # and correponds to X(width), Y(height) coordinates in the camera reference respectively
    u_0, v_0, f_u, f_v = expand_intrinsics(intrinsics)

    # Back projection (2D->3D)
    x = (pc_uv[:,1] - u_0) * z / f_u  # (N,)
    y = (pc_uv[:,0] - v_0) * z / f_v  # (N,)

    # Pack to point cloud
    if isinstance(sdepth, torch.Tensor):
        pc_xyz = torch.stack((x,y,z), dim=1)  # (N,3)
    else:
        # numpy array
        pc_xyz = np.stack((x,y,z), axis=1)  # (N,3)

    return pc_xyz, pc_idx


def sparse_depthmap_to_multi_scale_pointcloud(
    sdepth: Tensor,
    intrinsics: Tensor,
    first_scale: int,
    num_points: Sequence[int],
    num_neighbors: Sequence[int],
    leaf_size: int
) -> Tuple[List,...]:

    """
    Generates a multi-sclale point cloud from a sparse depth map.

    Arguments:

        sdepth : (1,H,W) tensor containing the depth image as float ,depth in meters

        intrinsics : (3,3) camera intrinsics matrix

    Return:

        A tuple (pc_idx, nbrs_idx, nbrs_disp)

            pc_idx : list of (Ns,) tensor per scale that contains the points indices, range [0,HW-1]

            nbrs_idx : list of (Ns,Ks) tensor per scale that contains the neighbors indices, range [0,HW-1]

            nbrs_disp : list of (3,Ns,Ks) tensor per scale that contain the 3d displacement vector between a point and its neighbors

    """

    # Frame dimensions
    h, w = sdepth.shape[-2:]

    # Initialize point class lists
    pc_idx = []
    nbrs_idx = []
    nbrs_disp = []

    # Iterate over scales
    num_scales = len(num_points)
    first = first_scale
    last = first + num_scales

    # Extract intrinsics paramters
    # U,V are horizontal and vertical pixel coordinates 
    # and correponds to X(width), Y(height) coordinates in the camera reference respectively
    u = torch.arange(w).view(1,-1).repeat(h,1)
    v = torch.arange(h).view(-1,1).repeat(1,w)
    dense_xyz = dense_back_projection(z=sdepth.unsqueeze(0), intrinsics=intrinsics.unsqueeze(0)).squeeze()
    
    # Iterate over scales
    for s in range(first, last):

        if s != 0:
            # Down scale, 
            # maintain original pixel coordinates for proper pointcloud construction
            sdepth, max_ind = F.max_pool2d(input=sdepth, kernel_size=(2,2), stride=(2,2), return_indices=True)  # (1,H,W)->(1,H/2,W/2)
            h, w = h//2, w//2
            u = u.flatten()[max_ind]
            v = v.flatten()[max_ind]

        # Back projection (2D->3D)
        pc_uv = torch.nonzero(sdepth, as_tuple=False)  #(M,3) where the first dimension is always 1
        pc_idx_s = pc_uv[:,1] * w + pc_uv[:,2]  # (M,)
        uu = u.flatten()[pc_idx_s]
        vv = v.flatten()[pc_idx_s]
        pc_xyz = dense_xyz[:,vv,uu].T

        # Randomly draw N points
        ns = num_points[s - first]
        sample = torch.randperm(pc_xyz.shape[0])[:ns]  # (Ns,), Ns<=Ms
        pc_xyz, pc_idx_s = pc_xyz[sample,:], pc_idx_s[sample]  # (Ns,3), (Ns,)

        # Find K nearest neighbors using KDTree
        # Notes:
        #   (1) k+1 is used since the query includes the trainig points
        #   (2) don't return distances as we're interested in displacement vectors
        #   (3) returned indices are in range [0,N-1]
        ks = num_neighbors[s-first] + 1
        nbrs_tree = KDTree(pc_xyz.cpu(), leaf_size=leaf_size)
        nbrs_idx_s = torch.from_numpy(nbrs_tree.query(pc_xyz.cpu(), k=ks, return_distance=False)[:,1:]).to(sdepth.device)  # (Ns,Ks)

        # Compute displacement vector per neighbor
        nbrs_disp_s = (pc_xyz.unsqueeze(1) - pc_xyz[nbrs_idx_s]).permute(2,0,1) # (Ns,Ks,3) change to (3,Ns,Ks)

        # Convert neighbors indices to linear indices in range [0,HW-1]
        nbrs_idx_s = pc_idx_s[nbrs_idx_s]  # (Ns,Ks)

        # Stack scales into a list
        pc_idx.append(pc_idx_s.unsqueeze(0))
        nbrs_idx.append(nbrs_idx_s.unsqueeze(0))
        nbrs_disp.append(nbrs_disp_s)

    return pc_idx, nbrs_idx, nbrs_disp


def batch_sparse_depth_to_multi_scale_point_cloud(
    sdepth: Tensor,
    intrinsics: Tensor,
    first_scale: int,
    num_points: Sequence[int],
    num_neighbors: Sequence[int],
    leaf_size: int
) -> Tuple[List,...]:

    """
    Generates a multi-sclale point cloud from a sparse depth map.

    Arguments:
        sdepth : (B,1,H,W) tensor containing the depth image as float ,depth in meters

        intrinsics : (B,3,3) intrinsics matrix

    Return:
        A tuple (pc_idx, nbrs_idx, nbrs_disp)
            pc_idx : list of (B,Ns) tensor per scale that contains the points indices, range [0,HW-1]

            nbrs_idx : list of (B,Ns,Ks) tensor per scale that contains the neighbors indices, range [0,HW-1]

            nbrs_disp : list of (B,3,Ns,Ks) tensor per scale that contain the 3d displacement vector between a point and its neighbors
    """
    
    batch_size = sdepth.shape[0]

    pc_idx_in = []
    nbrs_idx_in = []
    nbrs_disp_in = []

    for sample_idx in range(batch_size):
        pc_idx, nbrs_idx, nbrs_disp = sparse_depthmap_to_multi_scale_pointcloud(
            sdepth=sdepth[sample_idx],
            intrinsics=intrinsics[sample_idx],
            first_scale=first_scale,
            num_points=num_points,
            num_neighbors=num_neighbors,
            leaf_size=leaf_size
        )

        pc_idx_in.append(pc_idx)
        nbrs_idx_in.append(nbrs_idx)
        nbrs_disp_in.append(nbrs_disp)


    pc_idx = []
    nbrs_idx = []
    nbrs_disp = []

    for s in range(len(num_points)):
        pc_idx.append(torch.stack(list(zip(*pc_idx_in))[s]))  # [(B,1,N1),...,(B,1,Ns)] tensor per scale
        nbrs_idx.append(torch.stack(list(zip(*nbrs_idx_in))[s]))  # [(B,1,N1,K1),...,(B,1,Ns,Ks)] tensor per scale
        nbrs_disp.append(torch.stack(list(zip(*nbrs_disp_in))[s]))  # [(B,N1,K1,3),...,(B,Ns,Ks,3)] tensor per scale

    return (pc_idx, nbrs_idx, nbrs_disp)


def dense_back_projection(
    z: Tensor,
    intrinsics: Tensor
) -> Tensor:

    """
    Back projects a dense depth map to a point cloud.

    Arguments:
        z : dense depth map with dimensions (B,1,H,W)

        intrinsics : (B,3,3) camera intrinsics

    Return:
        (B,3,H,W) tensor where (B,0/1/2,H,W) are the x/y/z coordinates of each point respectively
    """

    # Extract dimensions
    B, _, H, W = z.shape

    # Extract intrinsics paramters
    # U,V are horizontal and vertical pixel coordinates 
    # and correponds to X(width), Y(height) coordinates in the camera reference respectively
    u_0, v_0, f_u, f_v = expand_intrinsics(intrinsics)  # (B,)

    # Back projection (2D->3D)
    x = torch.div(torch.arange(W, device=z.device).view(1,-1) - u_0.view(-1,1), f_u.view(-1,1)).view(B,1,1,-1) * z  # (B,1,H,W)
    y = torch.div(torch.arange(H, device=z.device).view(1,-1) - v_0.view(-1,1), f_v.view(-1,1)).view(B,1,-1,1) * z  # (B,1,H,W)

    coord = torch.cat((x,y,z), dim=1)  # (B,3,H,W)

    return coord


def pointcloud_to_image(
    pointcloud: Tensor,
    intrinsics: Tensor,
    height: int,
    width: int
) -> Tensor:
    
    u_0, v_0, f_u, f_v = expand_intrinsics(intrinsics)

    batch_size = pointcloud.size(0)
    X = pointcloud[:, 0, :, :]
    Y = pointcloud[:, 1, :, :]
    Z = pointcloud[:, 2, :, :].clamp(min=1e-3)

    # compute pixel coordinates
    u_proj = f_u.view(-1,1,1) * X / Z + u_0.view(-1,1,1)  # horizontal pixel coordinate
    v_proj = f_v.view(-1,1,1) * Y / Z + v_0.view(-1,1,1)  # vertical pixel coordinate

    # normalization to [-1, 1], required by torch.nn.functional.grid_sample
    u_proj_normalized = (2 * u_proj / (width - 1) - 1).view(batch_size, -1)
    v_proj_normalized = (2 * v_proj / (height - 1) - 1).view(batch_size, -1)

    # This was important since PyTorch didn't do as it claimed for points out of boundary
    # See https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
    # Might not be necessary any more
    u_proj_mask = ((u_proj_normalized > 1) + (u_proj_normalized < -1))
    u_proj_normalized[u_proj_mask] = 2
    v_proj_mask = ((v_proj_normalized > 1) + (v_proj_normalized < -1))
    v_proj_normalized[v_proj_mask] = 2

    pixel_coords = torch.stack([u_proj_normalized, v_proj_normalized],dim=2)  # [B, H*W, 2]

    return pixel_coords.view(batch_size, height, width, 2)


def transform_curr_to_near(
    pointcloud_curr: Tensor,
    rotation: Tensor,
    translation: Tensor,
    height: int,
    width: int
) -> Tensor:

    # translation and rotmat represent the transformation from tgt pose to src pose
    batch_size = pointcloud_curr.size(0)
    XYZ_ = torch.bmm(rotation, pointcloud_curr.view(batch_size, 3, -1))

    X = (XYZ_[:,0,:] + translation[:,0].unsqueeze(1)).view(-1, 1, height, width)
    Y = (XYZ_[:,1,:] + translation[:,1].unsqueeze(1)).view(-1, 1, height, width)
    Z = (XYZ_[:,2,:] + translation[:,2].unsqueeze(1)).view(-1, 1, height, width)

    pointcloud_near = torch.cat((X, Y, Z), dim=1)

    return pointcloud_near


def homography_from(
    rgb_near: Tensor,
    pred: Tensor,
    rotation: Tensor,
    translation: Tensor,
    intrinsics: Tensor
) -> Tensor:

    # inverse warp the RGB image from the nearby frame to the current frame
    # taken from: https://github.com/fangchangma/self-supervised-depth-completion

    height = pred.shape[2]
    width = pred.shape[3]

    # Verify dimensions
    rotation = rotation.view(-1, 3, 3)
    translation = translation.view(-1, 3)

    # Compute source pixel coordinate
    pointcloud_curr = dense_back_projection(pred, intrinsics)

    pointcloud_near = transform_curr_to_near(
        pointcloud_curr=pointcloud_curr,
        rotation=rotation,
        translation=translation,
        height=height,
        width=width
    )

    pixel_coords_near = pointcloud_to_image(
        pointcloud=pointcloud_near,
        intrinsics=intrinsics,
        height=height,
        width=width
    )

    # Warping
    warped = F.grid_sample(rgb_near, pixel_coords_near, align_corners=False)

    return warped


def feature_match(img1, img2):
    
    """
    Find features on both images and match them pairwise
    """ 

    max_n_features = 1000

    detector = cv2.SIFT_create(max_n_features)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    if (des1 is None) or (des2 is None):
        return [], []

    des1 = des1.astype(np.float32)
    des2 = des2.astype(np.float32)

    matcher = cv2.DescriptorMatcher().create('BruteForce')
    matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []
    
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    return pts1, pts2


def get_pose_pnp(
    gray_curr: Tensor,
    gray_near: Tensor,
    depth_curr: Tensor,
    intrinsics: Tensor
) -> Tuple:

    # feature matching
    pts2d_curr, pts2d_near = feature_match(gray_curr,
                                           gray_near)

    # dilation of depth
    kernel = np.ones((4, 4), np.uint8)
    depth_curr_dilated = cv2.dilate(src=depth_curr, kernel=kernel)

    # backproject 3d pts
    u_0, v_0, f_u, f_v = expand_intrinsics(intrinsics)
    z = depth_curr_dilated[pts2d_curr[:,1], pts2d_curr[:,0]]  #(N,)
    x = (pts2d_curr[:,0] - u_0) * z / f_u  # (N,)
    y = (pts2d_curr[:,1] - v_0) * z / f_v  # (N,)
    pc_xyz = np.stack((x,y,z), axis=1)

    # keep only feature points with depth in the current frame
    valid = z > 0.
    pts3d_curr = np.expand_dims(pc_xyz[valid,:], axis=1).astype(np.float32)
    pts2d_near_filtered = np.expand_dims(pts2d_near[valid,:], axis=1).astype(np.float32)

    # the minimal number of points accepted by solvePnP is 6:
    # if len(pts3d_curr) >= 6 and len(pts2d_near_filtered) >= 6:
    try:
        # ransac
        ret = cv2.solvePnPRansac(pts3d_curr,
                                 pts2d_near_filtered,
                                 intrinsics,
                                 distCoeffs=None)
        success = ret[0]
        rotation = ret[1]
        translation = ret[2].squeeze()
    except:
        success = False

    # discard if translation is too small
    translation_threshold = 0.1
    success = success and np.linalg.norm(translation) > translation_threshold

    if success:
        rotation, _ = cv2.Rodrigues(rotation)  # convert to rotation matrix
        result = (success, rotation, translation)
    else:
        # return the same image and no motion when PnP fails
        rotation = np.eye(3)
        translation = np.zeros(3)
        result = (success, rotation, translation)
    
    return result
