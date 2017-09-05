# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 15:27:50 2017

@author: pcomitz
impement sliding windows and 
fit a polynomial
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
from moviepy.editor import VideoFileClip
# for jupyter
from IPython.display import HTML



#### Perspective Transfrom
##### From project examples
#### pass in the source and destination
#### coordinates and
#### return warped or unwarped image 
#
def warper(img,inv = False):  
    # Perspective transform
    # choose the source and destination coordinates
    # used test2.jpg , use the deer crossing 
    # sign as a reference
    img_size = (img.shape[1], img.shape[0])
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
    [[p1x,p1y],
     [p2x,p2y],
     [p3x, p3y],
     [p4x,p4y]]) 
    
    dst = np.float32(
    [[d1x,d1y],
     [d2x,d2y],
     [d3x,d3y],
     [d4x,d4y]])
    # Compute and apply perpective transform
    if (inv == False):
        M = cv2.getPerspectiveTransform(src, dst)
    elif (inv == True):   
        M= cv2.getPerspectiveTransform(dst,src)
    
    # keep same size as input image
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  
    return warped


####
#### img is the undistorted image
#### returns an array of tow images
#### first image is uint combined x gradient and thresholded S (saturation)
#### 
####  second image is color binary float 64
####  green channel is gradient
####  blue channel is thresholded s channel 
####  
#### PHC - tweaked version, only returns sobelx and s combined 
####
def pipeline_sobel_hls (img, s_thresh_min=170, s_thresh_max = 255, 
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
    # color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
    # thresholdedImages = (combined_binary, color_binary)
    return combined_binary


#
# function to assist with too many indices problem
# extracts left and right line pixel positions
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

#
# function for sliding window
#
def slide_window(binary_warped,nwindows=9): 
    
    # create an output image to draw the windows on and visualize result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    # take histogram of bottom half of image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    # find the peak of the left and right halves of histogram
    # these are the starting points for the left and right lanes
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
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
                           
    return out_img,histogram,left_lane_inds,right_lane_inds



#
# Lane lines found, search a margin around 
# previous line positions
#
def search_region(binary_warped,left_lane_inds, right_lane_inds,left_fit,right_fit):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    
    # left lane
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    
    # right lane
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    
    # as before extract left amd right line pixel positions 
    leftx,lefty,rightx,righty = get_lane_pixels(binary_warped) 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return out_img, left_lane_inds, right_lane_inds,left_fitx,right_fitx


#
# Measuring Curvature
# from forum
# 

def measure_curvature(binary_warped,original_image,right_fitx,left_fitx):
    line_separation = np.mean(((right_fitx) + (left_fitx))/2)
    ym_per_pix = 30/700 
    xm_per_pix = 3.7/720 
    
    # Find offset of vehicle
    center_offset = line_separation - (binary_warped.shape[-1]//2)
    car_offset = center_offset * xm_per_pix
    
    #y_eval is the point where the radius of curvature is measured 
    # = 719 in these images
    y_eval = np.max(ploty)
    
    # fit polynomials
    # Fit new polynomials to x,y 
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    
    # do radius of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    ave_curverad = (left_curverad + right_curverad)/2
    
    car_offset = 'Car Offset: ' + '{0:.2f}'.format(car_offset) + 'm'
    ave_curverad = 'Radius of Curvature:' + '{0:.2f}'.format(ave_curverad) + 'm'
    
    # 
    # unwarp the image and plot on predicted lane lines
    #
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the blank image , green channel
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = warper(color_warp, inv = True)

    # Combine the result with the original image
    ######
    # quick and dirty test
    img = mpimg.imread('test_images/test6.jpg')
    original_image_test = warper(img,inv= False)
    print("np.shape(img)", np.shape(img))
    print("np.shape(original_image)", np.shape(original_image_test))
    print("np.shape(newwarp)", np.shape(newwarp))
    
    ######
    
    #quick hack test - works 
    #result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    
    result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)
    cv2.putText(result, car_offset , (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), thickness=2)
    cv2.putText(result, ave_curverad , (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), thickness=2)

    plt.imshow(result)
    mpimg.imsave('output_images/finalResultImage.jpg',result)
    
    return result

def process_image(img):
    
    #DEBUG: what does image look like
    original_image = img
    print(" original np.shape(img):", np.shape(original_image))
    
    # retreive the camera calibration data 
    file_name = 'wide_dist_pickle.p'
    parameters = pickle.load(open(file_name, 'rb'))
    mtx = parameters["mtx"]
    dist = parameters["dist"]
    print('mtx =', mtx)
    print('dist =', dist)
    
    #undistort the image
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    print("undist np.shape(img):", np.shape(undist))
    
    # do perspective transform 
    interim_result_persp = warper(undist,inv = False) 
    
    #DEBUG: whats is the shape of the ineterim result
    print("perspective np.shape(interim_result_persp):", np.shape(interim_result_persp))
    
    # do edge dection
    edge_hls_img = pipeline_sobel_hls(interim_result_persp)
    #DEBUG: whats is the shape of the edge_hls_img result
    print(" edge_hls_img np.shape(edge_hls_img):", np.shape(edge_hls_img))
    
     # do the initial sliding window search
    binary_warped = edge_hls_img
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    out_img,histogram,left_lane_inds,right_lane_inds = slide_window(binary_warped)    
        
    # Extract left and right line pixel positions
        
    leftx,lefty,rightx,righty = get_lane_pixels(binary_warped) 
 
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    # result at this stage (after below) out_img is binary image with sliding windows 
    # left lean is red , right lane is blue
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    #
    # once pix found, search region
    # TODO: get a new warped binary image from
    # next frame 
    #
    
    # not a new image 5:14
    # blows up if out_img
    # binary_warped is the sobel edge_hls_img
    #
    out_img = search_region(binary_warped,left_lane_inds, right_lane_inds,left_fit,right_fit)

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)

    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]


    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()

    #
    # TODO define margin  in one spot
    #
    margin = 100

    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    # display is  teh binary image, red left, blue right
    # green polygons showing region
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # original or undist? 
    #final_result =  measure_curvature(edge_hls_img, original_image,right_fitx,left_fitx)
    final_result =  measure_curvature(edge_hls_img, undist,right_fitx,left_fitx)
    
    return final_result

#############
# B E G I N #
#############

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
# save the undistored checkerboard image 
mpimg.imsave('output_images/test_image_undist.jpg',dst)

# Visualize undistortion
# plt.imshow(img)
# plt.imshow(dst)
"""
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Chessboard Image', fontsize=15)
ax2.imshow(dst)
ax2.set_title('Undistorted Chessboard Image', fontsize=15)
"""



############PROCESS IMAGE HERE ######
#TODO - get image from video
# note - i am starting here with the unistorted warped perspective

proj4_output = 'proj4.mp4'
clip1 = VideoFileClip('project_video.mp4')
proj4_clip = clip1.fl_image(process_image) 
proj4_clip.write_videofile(proj4_output, audio=False)
# %time for jupyter 


#use test3.jpg for display results
#test_image = mpimg.imread('test_images/test3.jpg')
#proj4_clip = process_image(test_image)

# not yet
#original_undist_perspective_image = proj4_clip
 
original_undist_perspective_image = mpimg.imread('output_images/warperPerspective.jpg')



# do edge dection
result = pipeline_sobel_hls(original_undist_perspective_image)
#does this need to be a binary image? - YES !!!
binary_warped = result
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])

# do the initial sliding window search
out_img,histogram,left_lane_inds,right_lane_inds = slide_window(binary_warped)    

### here 4:54 pm ####

#
# Extract left and right line pixel positions
#    
leftx,lefty,rightx,righty = get_lane_pixels(binary_warped) 
 
# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
    
# Generate x and y values for plotting
ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

### here 5:02

#visualize
plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
#the horizontal line aren't drawn if this code is included
#plt.xlim(0, 1280)
#plt.ylim(720, 0)

# order of plotting seems to matter
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
ax1.plot(histogram)
ax2.imshow(binary_warped, cmap = 'gray')

#
# once pix found, search region
# TODO: get a new warped binary image from
# next frame 
#
# return from search region is 
# return out_img, left_lane_inds, right_lane_inds,left_fitx,right_fitx
out_img = search_region(binary_warped,left_lane_inds, right_lane_inds,left_fit,right_fit)

#here 5:05 

#
#Visualize
#
# Create an image to draw on and an image to show the selection window
out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
window_img = np.zeros_like(out_img)

# Color in left and right line pixels
out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

# here 5:07

# Generate a polygon to illustrate the search window area
# And recast the x and y points into usable format for cv2.fillPoly()

#
# TODO define margin  in one spot
#
margin = 100

left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
left_line_pts = np.hstack((left_line_window1, left_line_window2))
right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
right_line_pts = np.hstack((right_line_window1, right_line_window2))

# Draw the lane onto the warped blank image
cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

# here 5:22

# order of plotting seems to matter
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
ax1.imshow(result)
ax1.plot(left_fitx, ploty, color='yellow')
ax1.plot(right_fitx, ploty, color='yellow')

ax2.imshow(binary_warped, cmap = 'gray')

#
# do curvature
# 5:26
original_undist_image = mpimg.imread('test_images/test6.jpg')
result =  measure_curvature(binary_warped, original_undist_image,right_fitx,left_fitx)
print("result")
plt.imshow(result)



