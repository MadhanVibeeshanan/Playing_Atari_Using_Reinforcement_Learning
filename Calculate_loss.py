#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 18:57:06 2018

@author: ubuntu
"""

from keras import backend as K

def cal_loss(y, q_val):
    err = K.abs(y - q_val)
    quad = K.clip(err, 0.0, 1.0)
    linear = err - quad
    loss = K.mean(0.5 * K.square(quad) + linear)
    return loss
