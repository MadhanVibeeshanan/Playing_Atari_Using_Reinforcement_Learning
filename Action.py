#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 18:50:35 2018

@author: ubuntu
"""


import numpy as  np
from Config import legal_actions, observe_step_num
import random



def pred_action(current_step, epsilon, step, model):
    if np.random.rand() <= epsilon or step <= observe_step_num:
        return random.randrange(legal_actions)
    else:
        q_value = model.predict([current_step, np.ones(legal_actions).reshape(1, legal_actions)])
        return np.argmax(q_value[0])