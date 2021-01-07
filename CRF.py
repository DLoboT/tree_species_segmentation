#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 13:49:50 2019

@author: daliana
"""

import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, unary_from_softmax
import sys

# Fully connected CRF post processing function

def do_crf(im, probs):
    
    H, W, C = probs.shape
       
    d = dcrf.DenseCRF2D(W, H, C)  # width, height, nlabels
    probs = probs.transpose(2, 0, 1)
    
    U = unary_from_softmax(probs)
    d.setUnaryEnergy(U)
    
    #This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3,3), compat=3)
    
    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    # im is an image-array, e.g. im.dtype == np.uint8 and im.shape == (640,480,3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=im.astype('uint8'), compat=10)
    
    #Run Inference for 5 steps 
    Q = d.inference(5) 

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0).reshape((H, W))

    return MAP

