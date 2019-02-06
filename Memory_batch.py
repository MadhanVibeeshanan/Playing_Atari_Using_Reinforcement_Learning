#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 18:53:56 2018

@author: ubuntu
"""


import random
from Config import legal_actions, game_shape, gamma, batch_size
from Encoder import one_hot_encoder
import numpy as np
from keras.callbacks import TensorBoard


def train_memory_batch(memory, model, log_dir):
    mini_batch_initializing = random.sample(memory, batch_size)
    history = np.zeros((batch_size, game_shape[0],game_shape[1], game_shape[2]))
    next_history = np.zeros((batch_size, game_shape[0], game_shape[1], game_shape[2]))
    target = np.zeros((batch_size,))
    actions, rewards, dead = [], [], []

    for index, value in enumerate(mini_batch_initializing):
        history[index] = value[0]
        next_history[index] = value[3]
        actions.append(value[1])
        rewards.append(value[2])
        dead.append(value[4])

    actions_mask = np.ones((batch_size, legal_actions))
    next_Q_values = model.predict([next_history, actions_mask])

    # like Q Learning, get maximum Q value at s'
    # But from target model
    for i in range(batch_size):
        if dead[i]:
            target[i] = -1
            # target[i] = reward[i]
        else:
            target[i] = rewards[i] + gamma * np.amax(next_Q_values[i])

    one_hot_encoder_action = one_hot_encoder(actions, legal_actions)
    one_hot_encoder_target = one_hot_encoder_action * target[:, None]

    tb_callback = TensorBoard(log_dir=log_dir, histogram_freq=0,write_graph=True, write_images=True)

    h = model.fit([history, one_hot_encoder_action], one_hot_encoder_target, epochs=1, batch_size=batch_size, verbose=2, callbacks=[tb_callback])

    #batch_size=batch_size #, verbose=0, callbacks=[tb_callback]

    #if h.history['loss'][0] > 10.0:
    #    print('too large')

    return h.history['loss'][0]

