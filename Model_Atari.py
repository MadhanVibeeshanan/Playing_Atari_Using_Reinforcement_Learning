#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 18:43:27 2018

@author: ubuntu
"""



from keras import layers
from keras.models import Model
from keras.optimizers import RMSprop
from Calculate_loss import cal_loss
from Config import legal_actions, game_shape, learning_rate

def atari_model():
    # With the functional API we need to define the inputs.
    frames = layers.Input(game_shape, name='frames')
    actions = layers.Input((legal_actions,), name='action')

    # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
    normalizing_input = layers.Lambda(lambda x: x / 255.0, name='normalizing_input')(frames)

    # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
    convolutional_layer_1 = layers.convolutional.Conv2D(16, (8, 8), strides=(4, 4), activation='relu')(normalizing_input)
    # "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
    convolutional_layer_2 = layers.convolutional.Conv2D(32, (4, 4), strides=(2, 2), activation='relu')(convolutional_layer_1)
    # Flattening the second convolutional layer.
    flattening = layers.core.Flatten()(convolutional_layer_2)
    # "The final hidden layer is fully-connected and consists of 256 rectifier units."
    hidden = layers.Dense(256, activation='relu')(flattening)
    # "The output layer is a fully-connected linear layer with a single output for each valid action."
    out = layers.Dense(legal_actions)(hidden)
    # Finally, we multiply the output by the mask!
    final_output = layers.Multiply(name='QValue')([out, actions])

    model = Model(inputs=[frames, actions], outputs=final_output)
    model.summary()
    optimizer = RMSprop(lr=learning_rate, rho=0.95, epsilon=0.01)
    # model.compile(optimizer, loss='mse')
    # to changed model weights more slowly, uses MSE for low values and MAE(Mean Absolute Error) for large values
    model.compile(optimizer, loss=cal_loss)
    return model