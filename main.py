import matplotlib.pyplot as plt
#plt.ion()

from project_helpers import *
import cv2

# Extracting data

K, img_names, init_pair, pixel_threshold = get_dataset_info(1)
img1 = (plt.imread('./data/1/kronan1.jpg') * 255).astype('uint8')
img2 = (plt.imread('./data/1/kronan2.jpg') * 255).astype('uint8')

# Compute SIFT-matches using OpenCV

from cv2 import SIFT_create, cvtColor, COLOR_RGB2GRAY, FlannBasedMatcher, drawMatchesKnn
rgb2gray = lambda img: cvtColor(img, COLOR_RGB2GRAY)
sift = SIFT_create(contrastThreshold=0.02, edgeThreshold=10, nOctaveLayers=3)

# We detect keypoints and simultaneously describe them using SIFT
kp1, des1 = sift.detectAndCompute(rgb2gray(img1),None)
kp2, des2 = sift.detectAndCompute(rgb2gray(img2),None)

# We use a k-NN-like system to find the most similar descriptions
all_matches = FlannBasedMatcher().knnMatch(des1, des2, k=2)
# Apply ratio test
# Here we filter out matches that are too similar to other matches (because then they are likely wrong)
# This is standard in OpenCV, see https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
matches = []
for m,n in all_matches:
    if m.distance < 0.75*n.distance:
        matches.append([m])

idx = np.random.choice(len(matches), size=10, replace=False)
matches_array = np.array(matches)
selected_matches = matches_array[idx]

# Run this code for a simple plot of the filtered matches

# Just making sure no other figures are impacting this one
plt.figure()
# Here is some supplied code from https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
# Feel free to play around with it
img3 = drawMatchesKnn(img1,kp1,img2,kp2,selected_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3,),plt.show()



# Count SIFT features per image
feat_count1 = len(kp1)
feat_count2 = len(kp2)
print(feat_count1)
print(feat_count2)

# Count number of total matches
total_matches_count = len(all_matches)
print(total_matches_count)

# Count the number of good matches after the ratio-test
good_matches_count = len(matches)
print(good_matches_count)

# Supplied code for extracting numpy arrays from matching keypoints
# Note, x1 and x2 are in homogenous coordinates after this
x1 = np.array([[kp1[match[0].queryIdx].pt[0], kp1[match[0].queryIdx].pt[1]] for match in matches])
x2 = np.array([[kp2[match[0].trainIdx].pt[0], kp2[match[0].trainIdx].pt[1]] for match in matches])
x1 = np.vstack((x1.T, np.ones(x1.shape[0])))
x2 = np.vstack((x2.T, np.ones(x2.shape[0])))
x = np.array([x1, x2])

x1_norm = np.linalg.inv(K) @ x[0]
x2_norm = np.linalg.inv(K) @ x[1]
threshold_px = 2
eps = threshold_px / K[0,0]
E_robust, inliers, errs, iters = estimate_E_robust(x1_norm, x2_norm, eps)
x1_inliers = inliers[:2]
x2_inliers = inliers[2:]

# Compute the essential matrix based on the keypoint matches we just computed between the two images

# Print the number of inliers

P1_k = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0]])

P_extract = extract_P_from_E(E_robust)

n = x1.shape[1] 
points_3D = np.zeros((4,4,n))

for i in range(len(P_extract)):
  for j in range(n):
    points_3D[i][:,j] = triangulate_3D_point_DLT(x1_k[:,j],x2_k[:,j],P1_k,P_extract[i])
  
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(points_3D[i][0], points_3D[i][1], points_3D[i][2], s=2, c='b')
  ax.set_aspect('equal')
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  plot_camera(P1_k, 1, ax)
  plot_camera(P_extract[i], 1, ax)
  
  plt.show()



# points_3D[i][:,j] = triangulate_3D_point_DLT(x1_k[:,j],x2_k[:,j],P1_k,P_extract[i])

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(points_3D[i][0], points_3D[i][1], points_3D[i][2], s=2, c='b')
# ax.set_aspect('equal')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plot_camera(P1_k, 1, ax)
# plot_camera(P_extract[i], 1, ax)

# plt.show()