#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 19:15:08 2018

@author: ubuntu
"""
import sys
sys.path.append('/home/ubuntu/Documents/Atari_Project_FInal_Modification')

from Training import train
from Test import test
import tensorflow as tf

def main(argv=None):
    train()
    #test()

if __name__ == '__main__':
    tf.app.run()