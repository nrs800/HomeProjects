#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 21:20:49 2024

@author: nathanaelseay
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch

# Define the autoencoder model
def build_autoencoder(hp):
    inputs = keras.Input(shape=(timesteps, input_dim))
    
    # Define the encoder architecture with dropout
    encoder = layers.LSTM(units=hp.Int('encoder_units', min_value=32, max_value=256, step=32),
                          return_sequences=True)(inputs)
    encoder = layers.Dropout(rate=hp.Float('encoder_dropout', min_value=0.0, max_value=0.5, step=0.1))(encoder)
    
    # Define the decoder architecture with dropout
    decoder = layers.LSTM(units=hp.Int('decoder_units', min_value=32, max_value=256, step=32),
                          return_sequences=True)(encoder)
    decoder = layers.Dropout(rate=hp.Float('decoder_dropout', min_value=0.0, max_value=0.5, step=0.1))(decoder)
    
    # Define the output layer
    outputs = layers.TimeDistributed(layers.Dense(input_dim))(decoder)
    
    # Build the autoencoder model
    autoencoder = keras.Model(inputs, outputs)
    autoencoder.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                        loss='mse')
    return autoencoder

# Define the data dimensions
timesteps = X_train.shape[1]
input_dim = 1

# Generate random data for demonstration purposes
import numpy as np
x_train = np.random.random((1000, timesteps, input_dim))

# Instantiate the Keras Tuner RandomSearch tuner
tuner = RandomSearch(
    build_autoencoder,
    objective='val_loss',
    max_trials=5,
    executions_per_trial=3,
    directory='my_dir',
    project_name='lstm_autoencoder_tuning'
)

# Perform the hyperparameter search
tuner.search(x=x_train, y=x_train, epochs=10, validation_split=0.2)

# Get the best hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the best hyperparameters and train it
best_model = tuner.hypermodel.build(best_hps)
best_model.fit(x_train, x_train, epochs=10, validation_split=0.2)
