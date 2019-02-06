#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 18:32:44 2018

@author: ubuntu
"""

from Config import train_dir, render, final_epsilon, init_epsilon, no_op_steps, replay_memory, refresh_target_model_num, epsilon_step_num, observe_step_num, num_episode
from Model_Atari import atari_model
from datetime import datetime
import tensorflow as tf
from keras.models import clone_model
import gym
import random
from Data_Pre_Processing import Pre_pro
import numpy as np
import os.path
import time
from Action import pred_action
from Memory import store_memory
from Memory_batch import train_memory_batch
from collections import deque
#import matplotlib.pyplot as plt

def train():
    episode_no = 0
    epsilon = init_epsilon

    memory = deque(maxlen=replay_memory)
    decay = (init_epsilon - final_epsilon) / epsilon_step_num
    g_step = 0
    model = atari_model()

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    log_dir = "{}/run-{}-log".format("/home/ubuntu", now)
    file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())

    model_target = clone_model(model)
    model_target.set_weights(model.get_weights())
    
    #Atari Emulator
    env = gym.make('Pong-v0') #Pong-v0 BreakoutDeterministic-v4
    
    #Until the episode number is less the number of episode mentioned do the following 
    while episode_no < num_episode:

        done = False
        dead = False
        inital_game = 1
        step, score, start_life = 0, 0, 5
        loss = 0.0
        observe = env.reset()

        #for the very first step select some random actions 
        for _ in range(random.randint(1, no_op_steps)):
            observe, _, _, _ = env.step(inital_game)

        #Preprocess the current step accroding to the alogrithm
        current_state = Pre_pro(observe)
        history = np.stack((current_state, current_state, current_state, current_state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        
        while not done:
            if render:
                env.render()
                time.sleep(0.01)
            
            #Call the get action to predict the action to be taken
            action = pred_action(history, epsilon, g_step, model_target)

            real_action = action + 1

            
            if epsilon > final_epsilon and g_step > observe_step_num:
                epsilon = epsilon - decay

            observe, reward, done, info = env.step(real_action)

            nxt_state = Pre_pro(observe)
            nxt_state = np.reshape([nxt_state], (1, 84, 84, 1))
            nxt_history = np.append(nxt_state, history[:, :, :, :3], axis=3)

            # An episode consist of n lives for each cames, untill all lives are gone the game will continue
            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            store_memory(memory, history, action, reward, nxt_history, dead)  #

            # condition to check if the storage memory is ready for training
            if g_step > observe_step_num:
                loss = loss + train_memory_batch(memory, model, log_dir)
                if g_step % refresh_target_model_num == 0:  # update the target model
                    model_target.set_weights(model.get_weights())

            score += reward

            if dead:
                dead = False
            else:
                history = nxt_history

            g_step += 1
            step += 1

            if done:
                if g_step <= observe_step_num:
                    state = "observe"
                elif observe_step_num < g_step <= observe_step_num + epsilon_step_num:
                    state = "explore"
                else:
                    state = "train"
                print('state: {}, episode: {}, score: {}, global_step: {}, avg loss: {}, step: {}, memory length: {}'
                      .format(state, episode_no, score, g_step, loss / float(step), step, len(memory)))
            

                if episode_no % 10 == 0 or (episode_no + 1) == num_episode:
                    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
                    file_name = "breakout_model_{}.h5".format(now)
                    model_path = os.path.join(train_dir, file_name)
                    model.save(model_path)


                loss_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="loss", simple_value=loss / float(step))])
                if loss_summary:
                    file_writer.add_summary(loss_summary, global_step = episode_no)

                score_summary = tf.Summary(
                    value=[tf.Summary.Value(tag="score", simple_value=score)])
                if score_summary:
                    file_writer.add_summary(score_summary, global_step = episode_no)
                
                episode_no += 1

    file_writer.close()
