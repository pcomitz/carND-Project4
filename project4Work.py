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
FROM UDACITY
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
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

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

# Test undistortion on a chessboardimage
#img = cv2.imread('camera_cal/calibration1.jpg')
img = mpimg.imread('camera_cal/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
dst = cv2.undistort(img, mtx, dist, None, mtx)
# write the undistored image 
# this is just a scratch 
# real output image written by notebook  
mpimg.imsave('output_images/test_image_undist.jpg',dst)

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
ax1.set_title('Original Chessboard Image', fontsize=15)
ax2.imshow(dst)
ax2.set_title('Undistorted Chessboard Image', fontsize=15)

##############################
# Pipleline single images
# Has the distortion correction been 
# correctly applied to each image? 
# How to tell if these are different? 
#############################
img = mpimg.imread('test_images/test6.jpg')
img_size = (img.shape[1], img.shape[0])
undist = cv2.undistort(img, mtx, dist, None, mtx)
mpimg.imsave('output_images/undist_image2.jpg',undist)

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
mpimg.imsave('output_images/warperPerspective2.jpg',warpedPerspective)


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
ax1.set_title('Original Undistorted Image src points', fontsize=10)
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
ax2.set_title('Undistorted Test Image Perspective, dest points', fontsize=10)


"""
9/1 
Thresholding, Edge Detections 
""" 
###
#SOBEL
###
# Define a function that applies Sobel x or y, 
# then takes an absolute value and applies a threshold.
# Note: calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
def abs_sobel_thresh_phc(img, orient='x', thresh_min=0, thresh_max=255, sobel_kernel=3):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 2) Take the absolute value of the derivative in x or y
    #    given orient = 'x' or 'y'
    if(orient == 'x'):
        abs_sobel= np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    else:
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel))
   
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        
    # 6) Return this mask as your binary_output image
    # binary_output = np.copy(img) # Remove this line
    binary_output = sxbinary
    return binary_output

# Plot the result
image = mpimg.imread('output_images/warperPerspective2.jpg')
grad_binary = abs_sobel_thresh_phc(image, orient='x', thresh_min=20, thresh_max=100)
mpimg.imsave('output_images/grad_binary.jpg',grad_binary, cmap = 'gray')
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=15)
ax2.imshow(grad_binary, cmap='gray')
ax2.set_title('Thresholded Gradient x ', fontsize=15)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

# now y 
grad_binary = abs_sobel_thresh_phc(image, orient='y', thresh_min=20, thresh_max=100)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=15)
ax2.imshow(grad_binary, cmap='gray')
ax2.set_title('Thresholded Gradient y ', fontsize=15)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


####
#### Magnitude of the Gradient
#### 
# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

# Exercise magnitude of the gradient
image = mpimg.imread('output_images/warperPerspective2.jpg')
kernel = 19
mag_binary = mag_thresh(image, sobel_kernel=kernel, mag_thresh=(30, 110))
mpimg.imsave('output_images/mag_binary.jpg', mag_binary, cmap = 'gray')
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=15)
ax2.imshow(mag_binary, cmap='gray')
label = 'Magnitude of Gradient kernel =' + str(kernel)
ax2.set_title(label, fontsize=15)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

####
#### Direction of the Gradient
####  
# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

image = mpimg.imread('output_images/warperPerspective2.jpg')
kernel = 19
dir_binary = dir_threshold(image, sobel_kernel=kernel, thresh=(0.7, 1.3))
mpimg.imsave('output_images/dir_binary.jpg', dir_binary, cmap = 'gray')
# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=15)
ax2.imshow(dir_binary, cmap='gray')
label = 'Direction of Gradient kernel =' + str(kernel)
ax2.set_title(label, fontsize=15)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

####
#### Combining Thresholds 
####
# Choose a Sobel kernel size
kernel = 19 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
image = mpimg.imread('output_images/warperPerspective2.jpg')
#image = mpimg.imread('test_images/signs_vehicles_xygrad.jpg')
gradx = abs_sobel_thresh_phc(image, orient='x', thresh_min=15, thresh_max=90, sobel_kernel=kernel)
grady = abs_sobel_thresh_phc(image, orient='y',thresh_min=15, thresh_max=90,  sobel_kernel=kernel)
mag_binary = mag_thresh(image, sobel_kernel=kernel, mag_thresh=(15, 90))
dir_binary = dir_threshold(image, sobel_kernel=kernel, thresh=(0.8, 1.4))

combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

# Plot the result of the combined image
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=15)
ax2.imshow(combined, cmap='gray')
label = 'Combined Thresholds kernel =' + str(kernel)
mpimg.imsave('output_images/combined_k19.jpg',combined, cmap = 'gray')
ax2.set_title(label, fontsize=15)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

####
#### 9/2 Combine color channels and gradient 
####

####
#### img is the undistorted image
#### returns an array of tow images
#### first image is uint combined x gradient and thresholded S (saturation)
#### 
####  second image is color binary float 64
####  green channel is gradient
####  blue channel is thresholded s channel 
####
def pipeline(img, s_thresh_min=170, s_thresh_max = 255, 
             thresh_min=20, 
             thresh_max=100, 
             sobel_kernel=3):
    # what is this for? 
    img = np.copy(img)
    
   # 1) covert to HLS and separate S channel
    hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    #2) Get the grayscale image
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
    # 3) sobel x - take x derivative of grayscale
    # lane lines are near vertical
    # absolute x value is to accentuate lines away from horizontal
    abs_sobelx= np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    
    # 4) scale and normalize
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    #5) Apply threshold to x gradient
    #   color = 1 if grayscale between thresh min and thresh max
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    #6) apply threshold to s color channel
    # shape of s_channel is (720,1280)
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    
    #7)Combined binary Gradient and S Channel 
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    #8) Stack the images
    # pass in zeros, x gradient, s
    # red all zeros at the moment, green gradient, b is s channel
    # what else to put in first chanel (all zeros) 
    # color_binary = np.dstack((h_channel, sxbinary, s_binary))
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
    thresholdedImages = (combined_binary, color_binary)
    
    return thresholdedImages

# use the undistorted image
# image = mpimg.imread('output_images/undist_image.jpg')

#changed back to left curving perspective to see if shadow was an issue
# and it is 
image = mpimg.imread('output_images/warperPerspective.jpg')
#hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
#gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

result = pipeline(image)

# display pipeline (combined) thresholded images 
"""
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 3))
f.tight_layout()

ax1.imshow(image)
ax1.set_title('Original Undistorted Image', fontsize=10)

ax2.imshow(result[0], cmap = 'gray')
ax2.set_title('Pipeline Result Combined', fontsize=10)
mpimg.imsave('output_images/pipeline_combined.jpg',result[0], cmap = 'gray')

ax3.imshow(result[1] )
ax3.set_title('Pipeline Result Color Binary', fontsize=10)
mpimg.imsave('output_images/color_binary.jpg',result[1])
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
"""

####
#### begin sliding window
#### and fit ploynomial 
####

#
# function to assist with too many indices problem
#
def get_lane_pixels(img):
    img_shape = img.shape
    leftx = []
    lefty = []
    rightx = []
    righty = []
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            # print(img[y,x])
            if img[y,x] > 0:
                if (x <= img_shape[1]/2):
                    leftx.append(x)
                    lefty.append(y)
                else:
                    rightx.append(x)
                    righty.append(y)
    return leftx, lefty, rightx, righty

binary_warped = result[0]
histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
#plt.plot(histogram)

# create an output image to draw on and visualize result
# why stack same warped binary 3 times
out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

# find the peak of the left and right halves of histogram
# these are the starting points for the left and right lanes
midpoint = np.int(histogram.shape[0]/2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

# set number of sliding windows
nwindows = 9

# set height of windows
window_height = np.int(binary_warped.shape[0]/nwindows)

# identify x and y positions of all nonzero (= white for binary image)
# pixels in image
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])

# current positions to be updated for each window
leftx_current = leftx_base
rightx_current = rightx_base

# set width of windows +/- margin
margin = 100

# Set minimum number of pixels found to recenter window
minpix = 50

# Create empty lists to receive left and right lane pixel indices
left_lane_inds = []
right_lane_inds = []

# step through the windows one by one
for window in range(nwindows): 
    win_y_low = binary_warped.shape[0] - (window+1)*window_height
    win_y_high = binary_warped.shape[0] - window*window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
    print("Win: %d, %d, %d, %d, %d, %d" %(win_y_low, win_y_high, win_xleft_low,win_xleft_high, win_xright_low, win_xright_high))
    
    # windows coordinates calculated, draw windows on image, left then right
    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
    
    # find the nonzero pixels in x and y in the window
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

    # append the good left and right indices to the lists
    #left_lane_inds.append(good_left_inds)
    #right_lane_inds.append(good_right_inds)
    #left_lane_inds.append(good_left_inds)
    #right_lane_inds.append(good_right_inds)
    # see https://stackoverflow.com/questions/9775297/append-a-numpy-array-to-a-numpy-array
    left_lane_inds= np.append(left_lane_inds,good_left_inds)
    right_lane_inds= np.append(right_lane_inds,good_right_inds)
    
    #IndexError: arrays used as indices must be of integer (or boolean) type
    left_lane_inds = left_lane_inds.astype(int)
    right_lane_inds = right_lane_inds.astype(int)

    # if > minpix found, recenter next window on mean position
    if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:        
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
 # Concatenate the arrays of indices
 #zero-dimensional arrays cannot be concatenated
 # is this even needed? 
 #left_lane_inds = np.concatenate(left_lane_inds)
 #right_lane_inds = np.concatenate(right_lane_inds)   
    
# Extract left and right line pixel positions
"""
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds] 
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds] 
"""
    
 
leftx,lefty,rightx,righty = get_lane_pixels(binary_warped) 
    
# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
    
# visualize
# Generate x and y values for plotting

ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]


# display sliding window
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
f.tight_layout()
ax1.imshow(binary_warped, cmap='gray')
ax1.set_title("Original Binary Warped")

ax2.imshow(out_img)
ax2.set_title("Sliding Window")
ax2.plot(left_fitx, ploty, color='yellow')
ax2.plot(right_fitx, ploty, color='yellow')


#
# ax2.xlim(0, 1280)
# AttributeError: 'AxesSubplot' object has no attribute 'xlim'
# 
#ax2.xlim(0, 1280)
#ax2.ylim(720, 0)

#this give a slightly different result than display code above? 
# not drawn at all when above is present? 

'''
plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
'''
# when below is added, most of the horizontal lines are not drawn?  
# plt.xlim(0, 1280)
# plt.ylim(720, 0)

#cv2.imshow('Sliding Window with cv2', out_img)






    

    


    
    


    








