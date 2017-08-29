# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 18:52:00 2017

@author: pcomitz
Desrcibed in 
https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/example_writeup.pdf
"""


import numpy as np
import cv2
import matplotlib.pyplot as plt
# reads rgb
import matplotlib.image as mpimg

def warper(img, src, dst):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    return warped

# read a test image
img = mpimg.imread('test_images/straight_lines1.jpg')
img_size = (img.shape[1], img.shape[0])
print("image size is", img_size[0], ",",img_size[1])

# calculate src and dst using the oddball method shown in the example
src = np.float32(
        [
        [img_size[0] / 2 - 55, img_size[1]/2 + 100], 
        [img_size[0]/6  -10, img_size[1]], 
        [img_size[0] * 5 /6 + 60, img_size[1]], 
        [img_size[0]/2 + 55, img_size[1]/2 + 100]
        ]
        )

#above is very odd, verify 
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

dst2 = np.float32(
[
 [d1x,d1y],
 [d2x,d2y],
 [d3x,d3y],
 [d4x,d4y]
]
)