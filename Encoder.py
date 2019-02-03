#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 19:12:18 2018

@author: ubuntu
"""

import numpy as np

def one_hot_encoder(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]

