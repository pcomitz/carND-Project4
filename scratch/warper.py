# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 18:52:00 2017

@author: pcomitz
Desrcibed in 
https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/example_writeup.pdf
"""


import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
# reads rgb
import matplotlib.image as mpimg

def warper(img, src, dst):



    # Compute and apply perpective transform

    img_size = (img.shape[1], img.shape[0])

    M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image



    return warped