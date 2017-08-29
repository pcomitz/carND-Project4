# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 19:07:19 2017

@author: pcomitz
phc 8/29/2017
"""
import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
# reads rgb
import matplotlib.image as mpimg
# for jupyter
# %matplotlib qt

"""
PERSPECTIVE TRANSFORM (UNUSED)
"""
#
# Perspective transform function
# example from lecture 
# THIS ONE UNUSED AS OF 8/28
#
def corners_unwarp(img, nx, ny, mtx, dist):
    # Use the OpenCV undistort() function to remove distortion
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    # Search for corners in the grayscaled image
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret == True:
        # If we found corners, draw them - visualization only
        # Choose offset from image corners to plot detected corners
        # This should be chosen to present the result at the proper aspect ratio
        # My choice of 100 pixels is not exact, but close enough for our purpose here
        offset = 100 # offset for dst points
        # Grab the image shape
        img_size = (gray.shape[1], gray.shape[0])

        # For source points I'm grabbing the outer four detected corners
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result 
        # again, not exact, but close enough for our purposes
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                     [img_size[0]-offset, img_size[1]-offset], 
                                     [offset, img_size[1]-offset]])
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M, img_size)

    # Return the resulting image and matrix
    return warped, M

"""
SOBEL TRANSFORM FUNCTION
"""
# Define a function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobelx = np.absolute(sobelx)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        
    # 6) Return this mask as your binary_output image
    # binary_output = np.copy(img) # Remove this line
    binary_output = sxbinary
    return binary_output


"""
PERSPECTIVE TRANSFROM FUNCTION
"""
# Perspective Transfrom
# From project examples
# pass in the source and destination
# coordinates and
# return warped image  
#
def warper(img, src, dst):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    return warped


###BEGIN EXECUTION###

#############################
#
# CAMERA CALIBRATION
#
#############################

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
#images = glob.glob('../camera_cal/calibration*.jpg')
images = glob.glob('camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for fname in images:
    #img = cv2.imread(fname)
    img = mpimg.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
        cv2.imshow('img',img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# Test undistortion on a checkerboard image
#img = cv2.imread('camera_cal/calibration1.jpg')
img = mpimg.imread('camera_cal/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
dst = cv2.undistort(img, mtx, dist, None, mtx)
# write the undistored image 
# this is just a scratch 
# real output image written by notebook  
cv2.imwrite('scratch/test_image.jpg',dst)

# Save the camera calibration result for later use 
# (we won't worry about rvecs / tvecs)
# SAVING IS PROBABLY NOT NEEDED FOR 
# PROJECT 4

dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "wide_dist_pickle.p", "wb" ) )

# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Checkerboard Image', fontsize=15)
ax2.imshow(dst)
ax2.set_title('Undistorted Checkerboard Image', fontsize=15)

##############################
# Pipleline single images
# Has the distortion correction been 
# correctly applied to each image? 
# How to tell if these are different? 
#############################
img = mpimg.imread('test_images/test2.jpg')
img_size = (img.shape[1], img.shape[0])
undist = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('scratch/undist_image.jpg',dst)

# visualize undistortion on test image
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Test Image', fontsize=15)
ax2.imshow(undist)
ax2.set_title('Undistorted Test Image', fontsize=15)

########################################
# Perspective transform
# choose the source and destination coordinates
# use test2.jpg , use the deer crossing 
# sign as a reference
########################################
p1x = (img_size[0]/2 -55)
p1y = img_size[1]/2 + 100
p2x = (img_size[0]/6) -10
p2y = img_size[1]
p3x = (img_size[0] * 5 /6) + 60
p3y = img_size[1]
p4x = (img_size[0]/2 + 55)
p4y = img_size[1]/2 + 100

d1x = (img_size[0]/4)
d1y = 0
d2x = (img_size[0]/4)
d2y = img_size[1]
d3x = (img_size[0]*3 /4)
d3y = img_size[1]
d4x = (img_size[0]*3 /4)
d4y = 0

src = np.float32(
[
 [p1x,p1y],
 [p2x,p2y],
 [p3x, p3y],
 [p4x,p4y]
]        
) 

dst = np.float32(
[
 [d1x,d1y],
 [d2x,d2y],
 [d3x,d3y],
 [d4x,d4y]
]
)
warpedPerspective = warper(undist,src,dst)

# Visualize original undistorted test
# image and warped perspective#
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
f.tight_layout()
#plot dots
ax1.imshow(undist)
ax1.plot(p1x,p1y,'.') # top right
ax1.plot(p2x,p2y,'.') # bottom right
ax1.plot(p3x,p3y,'.') # bottom left
ax1.plot(p4x,p4y,'.') # top left
# show outline of 4 src points 
ax1.plot([p1x,p2x],[p1y,p2y],'red',lw=3)
ax1.plot([p3x,p4x], [p3y,p4y],'red', lw=3)
ax1.plot([p2x,p3x],[p2y,p3y], 'red', lw=3)
ax1.plot([p4x,p1x],[p4y,p1y], 'red', lw=3)
ax1.set_title('Original Undistorted Image', fontsize=15)
ax2.imshow(warpedPerspective)
ax2.plot(d1x,d1y,'.') # top right
ax2.plot(d2x,d2y,'.') # bottom right
ax2.plot(d3x,d3y,'.') # bottom left
ax2.plot(d4x,d4y,'.') # top left
# show outline of 4 dst points 
ax2.plot([d1x,d2x],[d1y,d2y],'green',lw=3)
ax2.plot([d3x,d4x], [d3y,d4y],'green', lw=3)
ax2.plot([d2x,d3x],[d2y,d3y], 'green', lw=3)
ax2.plot([d4x,d1x],[d4y,d1y], 'green', lw=3)
ax2.set_title('Undistorted Test Image Perspective', fontsize=15)




