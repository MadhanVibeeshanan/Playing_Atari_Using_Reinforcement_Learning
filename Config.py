#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 18:33:23 2018

@author: ubuntu
"""



train_dir = '/home/ubuntu/Documents/dl_project_log'
restore_file_path = '/home/ubuntu/save_model/breakout_model_20180610205843_36h_12193ep_sec_version.h5'
num_episode = 10000
observe_step_num = 50000
epsilon_step_num = 1000000
refresh_target_model_num = 10000
replay_memory = 400000
no_op_steps = 30
regularizer_scale = 0.01
batch_size= 32
learning_rate = 0.00025
init_epsilon = 1.0
final_epsilon = 0.1
gamma = 0.99
resume = False
render = True
game_shape = (84, 84, 4)
legal_actions = 3
