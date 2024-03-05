#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 19:46:26 2024

@author: nathanaelseay
"""

import os 
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from deephyper.search.hps import RandomSearch
from deephyper.search.nas.model.space.node import Choice, Dense
from deephyper.search.nas.model.space.op.op1d import Dense, Identity
from deephyper.search import util

def file_call():
    def import_files_in_directory(directory):
        imported_data = []

        for root, dirs, files in os.walk(directory):
            for file_name in files:
                if file_name.endswith('.39'):  # Process only text files
                    file_path = os.path.join(root, file_name)
                    imported_data.append(file_path)  # Store file path and data

        return imported_data
    
    current_directory = os.getcwd()  # Get the current working directory
    global imported_files
    imported_files = import_files_in_directory(current_directory)
    return imported_files 

filenames =file_call()

waves_df = pd.DataFrame()

for filename in filenames:
     df=pd.read_csv(filename, sep='\t', header= None)
     df= df[0]
     waves_df=pd.concat([waves_df,df], axis= 1)
 
waveform_data = waves_df.values

def make_generator_model(units):
    model = tf.keras.Sequential()
    model.add(layers.Dense(units, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(20480, activation='tanh')) # Output shape is (44100,)

    return model

def make_discriminator_model(units):
    model = tf.keras.Sequential()
    model.add(layers.Dense(units, input_shape=(20480,)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

# Define the search space for hyperparameters
space = [
    Choice('units', [64, 128, 256]),
    Choice('batch_size', [16, 32, 64]),
    Choice('epochs', [5000, 10000, 15000])
]

@util.inject_hyperparameters(space)
def run(units, batch_size, epochs):
    # Create the generator and discriminator models
    generator = make_generator_model(units)
    discriminator = make_discriminator_model(units)

    # Compile the discriminator
    discriminator.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                          loss='binary_crossentropy')

    # The generator takes noise as input and generates waveforms
    z = layers.Input(shape=(100,))
    waveform = generator(z)

    # For the combined model we will only train the generator
    discriminator.trainable = False

    # The discriminator takes generated images as input and determines validity
    validity = discriminator(waveform)

    # The combined model  (stacked generator and discriminator)
    combined = Model(z, validity)
    combined.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                     loss='binary_crossentropy')

    # Train the GAN
    for epoch in range(epochs):
        # Sample random noise for the generator
        noise = np.random.normal(0, 1, (batch_size, 100))

        # Generate a batch of new waveforms
        generated_waveforms = generator.predict(noise)

        # Introduce anomalies into a subset of generated waveforms
        num_anomalies = batch_size // 4
        anomaly_indices = np.random.choice(batch_size, num_anomalies, replace=False)

        for idx in anomaly_indices:
            generated_waveforms[idx] += np.random.normal(0, 0.5, 20480)  # Add Gaussian noise as anomaly

        # Get a random batch of waveforms from the training set
        idx = np.random.randint(0, waveform_data.shape[1], batch_size)
        real_waveforms = waveform_data[:,idx]
        real_waveforms = real_waveforms.transpose()

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(real_waveforms, np.ones((batch_size, 1)))
        d_loss_fake = discriminator.train_on_batch(generated_waveforms, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train the generator
        g_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))

        # Print progress
        print(f"Epoch {epoch+1}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")

        # If at save interval, save generated waveforms
        if (epoch + 1) % 10 == 0:
            # Generate some example waveforms
            examples = generator.predict(np.random.normal(0, 1, (10, 100)))

            # # Plot examples
            # plt.figure(figsize=(10, 5))
            # for i in range(examples.shape[0]):
            #     plt.subplot(2, 5, i + 1)
            #     plt.plot(examples[i])
            #     plt.axis('off')
            # plt.tight_layout()
            # plt.savefig(f"generated_waveforms_epoch_{epoch+1}.png")
            # plt.close()

# Create and run the hyperparameter search
search = RandomSearch(run, space, num_trials=10)
search.search()
