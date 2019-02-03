#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 18:48:17 2018

@author: ubuntu
"""


from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np


def Pre_pro(current_image):
    #print(observe)
    processed = np.uint8(resize(rgb2gray(current_image), (84, 84), mode='constant') * 255)
    #print(processed_observe)
    return processed
