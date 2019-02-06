#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 18:52:40 2018

@author: ubuntu
"""


def store_memory(memory, current_step, action, reward, next_history, dead):
    memory.append((current_step, action, reward, next_history, dead))

