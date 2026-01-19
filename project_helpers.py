import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

def get_dataset_info(dataset):
    """
    Datasets 1-5 are relatively easy, since they have little lens distortion and
    no dominant scene plane. You should get reasonably good reconstructions for
    these datasets using RANSAC to estimate E only (CE2 in assignment 4).

    Datasets 6-9 are much more difficult, since the scene is dominated by a near
    planar structure. Also the lens distortion is larger for the used camera,
    hence feel free to adjust the pixel_threshold if necessary.
    """
    if dataset == 1:
        img_names = ["data/1/kronan1.JPG", "data/1/kronan2.JPG"]
        im_width = 1936
        im_height = 1296
        focal_length_35mm = 45.0 # from the EXIF data
        init_pair = [0, 1]
        pixel_threshold = 1.0
    elif dataset == 2:
        # Corner of a courtyard
        img_names = ["data/2/DSC_0025.JPG", "data/2/DSC_0026.JPG", "data/2/DSC_0027.JPG", "data/2/DSC_0028.JPG", "data/2/DSC_0029.JPG", "data/2/DSC_0030.JPG", "data/2/DSC_0031.JPG", "data/2/DSC_0032.JPG", "data/2/DSC_0033.JPG"]
        im_width = 1936
        im_height = 1296
        focal_length_35mm = 43.0 # from the EXIF data
        init_pair = [0, 8]
        pixel_threshold = 1.0
    elif dataset == 3:
        # Smaller gate of a cathetral
        img_names = ["data/3/DSC_0001.JPG", "data/3/DSC_0002.JPG", "data/3/DSC_0003.JPG", "data/3/DSC_0004.JPG", "data/3/DSC_0005.JPG", "data/3/DSC_0006.JPG", "data/3/DSC_0007.JPG", "data/3/DSC_0008.JPG", "data/3/DSC_0009.JPG", "data/3/DSC_0010.JPG", "data/3/DSC_0011.JPG", "data/3/DSC_0012.JPG"]
        im_width = 1936
        im_height = 1296
        focal_length_35mm = 43.0 # from the EXIF data
        init_pair = [4, 7]
        pixel_threshold = 1.0
    elif dataset == 4:
        # Fountain
        img_names = ["data/4/DSC_0480.JPG", "data/4/DSC_0481.JPG", "data/4/DSC_0482.JPG", "data/4/DSC_0483.JPG", "data/4/DSC_0484.JPG", "data/4/DSC_0485.JPG", "data/4/DSC_0486.JPG", "data/4/DSC_0487.JPG", "data/4/DSC_0488.JPG", "data/4/DSC_0489.JPG", "data/4/DSC_0490.JPG", "data/4/DSC_0491.JPG", "data/4/DSC_0492.JPG", "data/4/DSC_0493.JPG"]
        im_width = 1936
        im_height = 1296
        focal_length_35mm = 43.0 # from the EXIF data
        init_pair = [4, 9]
        pixel_threshold = 1.0
    elif dataset == 5:
        # Golden statue
        img_names = ["data/5/DSC_0336.JPG", "data/5/DSC_0337.JPG", "data/5/DSC_0338.JPG", "data/5/DSC_0339.JPG", "data/5/DSC_0340.JPG", "data/5/DSC_0341.JPG", "data/5/DSC_0342.JPG", "data/5/DSC_0343.JPG", "data/5/DSC_0344.JPG", "data/5/DSC_0345.JPG"]
        im_width = 1936
       
        im_height = 1296
        focal_length_35mm = 45.0 # from the EXIF data
        init_pair = [2, 6]
        pixel_threshold = 1.0
    elif dataset == 6:
        # Detail of the Landhaus in Graz.
        img_names = ["data/6/DSCN2115.JPG", "data/6/DSCN2116.JPG", "data/6/DSCN2117.JPG", "data/6/DSCN2118.JPG", "data/6/DSCN2119.JPG", "data/6/DSCN2120.JPG", "data/6/DSCN2121.JPG", "data/6/DSCN2122.JPG"]
        im_width = 2272
       
        im_height = 1704
        focal_length_35mm = 38.0 # from the EXIF data
        init_pair = [1, 3]
        pixel_threshold = 1.0
    elif dataset == 7:
        # Building in Heidelberg.
        img_names = ["data/7/DSCN7409.JPG", "data/7/DSCN7410.JPG", "data/7/DSCN7411.JPG", "data/7/DSCN7412.JPG", "data/7/DSCN7413.JPG", "data/7/DSCN7414.JPG", "data/7/DSCN7415.JPG"]
        im_width = 2272
        im_height = 1704
        focal_length_35mm = 38.0 # from the EXIF data
        init_pair = [0, 6]
        pixel_threshold = 1.0
    elif dataset == 8:
        # Relief
        img_names = ["data/8/DSCN5540.JPG", "data/8/DSCN5541.JPG", "data/8/DSCN5542.JPG", "data/8/DSCN5543.JPG", "data/8/DSCN5544.JPG", "data/8/DSCN5545.JPG", "data/8/DSCN5546.JPG", "data/8/DSCN5547.JPG", "data/8/DSCN5548.JPG", "data/8/DSCN5549.JPG", "data/8/DSCN5550.JPG"]
        im_width = 2272
        im_height = 1704
        focal_length_35mm = 38.0 # from the EXIF data
        init_pair = [3, 6]
        pixel_threshold = 1.0
    elif dataset == 9:
        # Triceratops model on a poster.
        img_names = ["data/9/DSCN5184.JPG", "data/9/DSCN5185.JPG", "data/9/DSCN5186.JPG", "data/9/DSCN5187.JPG", "data/9/DSCN5188.JPG", "data/9/DSCN5189.JPG", "data/9/DSCN5191.JPG", "data/9/DSCN5192.JPG", "data/9/DSCN5193.JPG"]
        im_width = 2272
        im_height = 1704
        focal_length_35mm = 38.0 # from the EXIF data
        init_pair = [3, 5]
        pixel_threshold = 1.0
    elif dataset == 10:
        # Add your optional datasets...
        pass

    focal_length = max(im_width, im_height) * focal_length_35mm / 35.0
    K = np.array([[focal_length, 0, im_width/2], [0, focal_length, im_height/2], [0, 0, 1]])
    return K, img_names, init_pair, pixel_threshold

def correct_H_sign(H, x1, x2):
    N = x1.shape[1]
    if x1.shape[0] != 3:
        x1 = np.vstack([x1, np.ones_like(x1[:1,:])])
    if x2.shape[0] != 3:
        x2 = np.vstack([x2, np.ones_like(x2[:1,:])])

    positives = sum((sum(x2 * (H @ x1), 0)) > 0)
    if positives < N/2:
        H *= -1
    
    return H

def homography_to_RT(H):
    
    def unitize(a,b):
        denom = 1.0 / (a**2+b**2)**(0.5)
        ra = a * denom
        rb = b * denom
        return ra, rb

    [U,S,Vt] = np.linalg.svd(H)
    s1 = S[0] / S[1]
    s3 = S[2] / S[1]
    a1 = (1 - s3**2)**(0.5)
    b1 = (s1**2 - 1)**(0.5)
    [a,b] = unitize(a1, b1)
    [c,d] = unitize(1+s1*s3, a1*b1 )
    [e,f] = unitize(-b/s1, -a/s3 )
    v1 = Vt.T[:,0]
    v3 = Vt.T[:,2]
    n1 = b * v1 - a * v3
    n2 = b * v1 + a * v3
    R1 = U @ np.array([[c,0,d], [0,1,0], [-d,0,c]]) @ Vt
    R2 = U @ np.array([[c,0,-d], [0,1,0], [d,0,c]]) @ Vt
    t1 = (e * v1 + f * v3).reshape(-1,1)
    t2 = (e * v1 - f * v3).reshape(-1,1)
  
    if n1[2] < 0:
        t1 = -t1
        n1 = -n1

    if n2[2] < 0:
        t2 = -t2
        n2 = -n2

    t1 = R1 @ t1
    t2 = R2 @ t2

    RT = np.zeros((2,3,4))
    RT[0] = np.hstack([R1, t1])
    RT[1] = np.hstack([R2, t2])
    return RT

# Helpers I imported

def pflat(x):
    # Normalizes (m,n)-array x of n homogeneous points
    # to last coordinate 1.
    y = x / x[-1,:]
    return y

def triangulate_3D_point_DLT(x1, x2, P1, P2):
    u1 = x1[0]
    v1 = x1[1]
    u2 = x2[0]
    v2 = x2[1]

    M = np.zeros((4, 4))

    M[0] = u1 * P1[2] - P1[0]
    M[1] = v1 * P1[2] - P1[1]
    M[2] = u2 * P2[2] - P2[0]
    M[3] = v2 * P2[2] - P2[1]

    U, S, Vt = np.linalg.svd(M)

    X = Vt[-1]

    return X/X[-1]

def camera_center_and_axis(P):
    # The camera center can be found by taking the null space of the camera matrix
    camera_center = pflat(sp.linalg.null_space(P))[:3]

    principal_axis = P[-1, :3]
    principal_axis = principal_axis / np.linalg.norm(principal_axis)

    return camera_center, principal_axis

def plot_camera(camera_matrix, scale, ax=None):
    if ax is None:
        ax = plt.axes(projection='3d')
    (camera_center, principal_direction) = camera_center_and_axis(camera_matrix)

    ax.scatter3D(camera_center[0], camera_center[1], camera_center[2], c='g')
    dir = principal_direction * scale
    ax.quiver(camera_center[0], camera_center[1], camera_center[2], dir[0], dir[1], dir[2], color='r')
    
def enforce_essential(E_approx):
    '''
    E_approx - Approximate Essential matrix (3x3)
    '''
    U, S, Vt = np.linalg.svd(E_approx)
    
    E = U @ np.diag([1, 1, 0]) @ Vt
    
    return E
    
def compute_epipolar_errors(F, x1s, x2s):
    '''
    x1s and x2s contain matching points
    x1s - 2D image points in the first image in homogenous coordinates (3xN)
    x2s - 2D image points in the second image in homogenous coordinates (3xN)
    F - Fundamental matrix (3x3)
    '''
    epi_lines = F @ x1s
    
    errors = (np.abs(np.sum(epi_lines * x2s, axis=0))) / (np.sqrt(epi_lines[0]**2 + epi_lines[1]**2))
    
    return errors

def estimate_F_DLT(x1s, x2s):
    '''
    x1s and x2s contain matching points
    x1s - 2D image points in the first image in homogenous coordinates (3xN)
    x2s - 2D image points in the second image in homogenous coordinates (3xN)
    '''
    N = x1s.shape[1]
    
    M = np.zeros((N, 9))
    
    for i in range(N):
        x1 = x1s[:, i]
        x2 = x2s[:, i]
        
        M[i] = [x2[0]*x1[0], x2[0]*x1[1], x2[0]*x1[2], x2[1]*x1[0], x2[1]*x1[1], x2[1]*x1[2], x2[2]*x1[0], x2[2]*x1[1], x2[2]*x1[2]]
        
    U1, S1, Vt1 = np.linalg.svd(M)
    
    v = Vt1[-1]
    
    min_singular_value = S1[-1]
    Mv_norm = np.linalg.norm(M @ v)
    
    #print(f"Min singular value: {min_singular_value}")
    #print(f"Error vector: {Mv_norm}")
    
    F_approx = v.reshape((3, 3))
    
    return F_approx
    
def estimate_E_robust(x1, x2, eps, seed=None):
    """
    RANSAC estimate of essential matrix using normalized correspondences x1 and x2 and a normalized threshold.
    Note: Make sure to normalize things before using it in this function!
    -------------------------------------------
    x1: Normalized keypoints in image 1 - 3xN np.array or 2xN np.array, as you desire 
    x2: Normalized keypoints in image 2 - 3xN np.array or 2xN np.array, as you desire 
    eps: Normalized inlier threshold - float

    Returns:
    E: 3x3 essential matrix
    inliers: The inlier points
    errs: The epipolar errors
    iters: How many iterations it took
    """
    
    num_points = x1.shape[1]
    num_inliers = 0
    best_num_inliers = 0
    best_inlier_mask = None
    best_errs = None
    iters = 0
    max_iters = 2001
    
    while iters < max_iters:
        iters += 1
        randind = np.random.choice(num_points, size=8, replace=False)
        E = enforce_essential(estimate_F_DLT(x1[:, randind], x2[:, randind]))
        e1 = compute_epipolar_errors(E, x1, x2)**2 
        e2 = compute_epipolar_errors(E.T, x2, x1)**2
        
        errs = (1/2)*(e1+e2)
        inlier_mask = errs < eps**2
        num_inliers = sum(inlier_mask)

        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_errs = errs
            best_inlier_mask = inlier_mask
            ratio_inliers = num_inliers/num_points
            if ratio_inliers > 0.8:
                break
    
    x1_inliers = x1[:, best_inlier_mask]
    x2_inliers = x2[:, best_inlier_mask]           
    E = enforce_essential(estimate_F_DLT(x1_inliers, x2_inliers))
    inliers = np.vstack((x1_inliers,x2_inliers))
    errs = best_errs
    #print('inliers',sum(best_inlier_mask))
    
    return E, inliers, errs, iters

def count_points_in_front(P1, P2, x1_k, x2_k):
    n = x1_k.shape[1]
    count = 0
    for i in range(n):
        X = triangulate_3D_point_DLT(x1_k[:, i], x2_k[:, i], P1, P2)
        
        if (P1[2] @ X + P1[2,3] > 0) and (P2[2] @ X + P2[2,3] > 0):
            count += 1
            
    return count

def extract_P_from_E(E):
     '''
     A function that extract the four P2 solutions given above
     E - Essential matrix (3x3)
     P - Array containing all four P2 solutions (4x3x4) (i.e. P[i,:,:] is the ith solution) 
     '''
     U, S, Vt = np.linalg.svd(E)
     
     if np.linalg.det(U @ Vt) < 0:
          Vt = -Vt
          
     W = np.array([[0, -1, 0],
                    [1,  0, 0],
                    [0,  0, 1]])

     R1 = U @ W @ Vt
     R2 = U @ np.transpose(W) @ Vt
     t  = U[:, 2].reshape(3,1)

     P = np.zeros((4, 3, 4))
     P[0] = np.hstack((R1,  t))
     P[1] = np.hstack((R1, -t))
     P[2] = np.hstack((R2,  t))
     P[3] = np.hstack((R2, -t))

     return P