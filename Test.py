#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 20:01:36 2018

@author: ubuntu
"""

from Config import observe_step_num, num_episode, restore_file_path
import gym
from Data_Pre_Processing import Pre_pro
import numpy as np
import time
from Action import pred_action
from Calculate_loss import cal_loss
from keras.models import load_model


def test():
    env = gym.make('BreakoutDeterministic-v4')

    episode_number = 0
    epsilon = 0.001
    global_step = observe_step_num+1
    model = load_model(restore_file_path, custom_objects={'loss': cal_loss})

    while episode_number < num_episode:

        done = False
        dead = False
        score, start_life = 0, 5
        observe = env.reset()

        observe, _, _, _ = env.step(1)

        state = Pre_pro(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        while not done:
            env.render()
            time.sleep(0.01)

            action = pred_action(history, epsilon, global_step, model)

            real_action = action + 1

            observe, reward, done, info = env.step(real_action)
            # pre-process the observation --> history
            next_state = Pre_pro(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            # if the agent missed ball, agent is dead --> episode is not over
            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            # TODO: may be we should give negative reward if miss ball (dead)
            reward = np.clip(reward, -1., 1.)

            score += reward

            # If agent is dead, set the flag back to false, but keep the history unchanged,
            # to avoid to see the ball up in the sky
            if dead:
                dead = False
            else:
                history = next_history

            # print("step: ", global_step)
            global_step += 1

            if done:
                episode_number += 1
                print('episode: {}, score: {}'.format(episode_number, score))