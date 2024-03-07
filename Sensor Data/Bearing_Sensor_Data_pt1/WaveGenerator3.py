#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 18:29:44 2024

@author: nathanaelseay
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
from deephyper.nas.metrics import r2, mse
from deephyper.problem import HpProblem
from deephyper.evaluator import Evaluator
from deephyper.search.hps import CBO
 
def file_call():
 
    def import_files_in_directory(directory):
        imported_data = []
 
        for root, dirs, files in os.walk(directory):
            for file_name in files:
                if file_name.endswith('.39'):  # Process only text files
                    file_path = os.path.join(root, file_name)
                    # with open(file_path, 'r') as file:
                    #     data = file.read()
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
     #df=df.transpose()
     #df= df["1"]
    
     waves_df=pd.concat([waves_df,df], axis= 1)
    
     
waveform_data = waves_df.values
 
 
 
 
# Hyperparameters
# epochs = 100000
noise_dim = 100
# batch_size = 32
 
def build_and_train_model(config):
    print(config)
    def make_generator_model(config):
        model = tf.keras.Sequential()
        model.add(layers.Dense(config["units"], use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
 
        model.add(layers.Dense(20480, activation=config["generator_activation"])) # Output shape is (44100,)
 
        return model
 
    # Define the Discriminator
    def make_discriminator_model(config):
        model = tf.keras.Sequential()
        model.add(layers.Dense(config["units"], input_shape=(20480,)))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(config["dropout_rate"]))
 
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation=config["generator_activation"]))
 
        return model
    print("buildT_Top")
    # Create the generator and discriminator models
    generator = make_generator_model(config)
    discriminator = make_discriminator_model(config)
 
    # Compile the discriminator
    discriminator.compile(optimizer=tf.keras.optimizers.RMSprop(config["learning_rate"]),
                      loss='binary_crossentropy')
 
    # The generator takes noise as input and generates waveforms
    z = layers.Input(shape=(noise_dim,))
    waveform = generator(z)
 
    # For the combined model we will only train the generator
    discriminator.trainable = False
 
    # The discriminator takes generated images as input and determines validity
    validity = discriminator(waveform)
   
    # The combined model  (stacked generator and discriminator)
    combined = Model(z, validity)
    combined.compile(optimizer=tf.keras.optimizers.RMSprop(config["learning_rate"]),
                 loss='binary_crossentropy')
 
 
 
    # Train the GAN
 
   
    g_df = pd.DataFrame()
    d_df = pd.DataFrame()
    g_loss = pd.DataFrame()
    d_loss = pd.DataFrame()
   
    for epoch in range(config["epochs"]):
        # Sample random noise for the generator
        noise = np.random.normal(0, 1, (config["batch_size"], noise_dim))
 
        # Generate a batch of new waveforms
        generated_waveforms = generator.predict(noise)
 
        # Introduce anomalies into a subset of generated waveforms
        num_anomalies = config["batch_size"] // 4
        anomaly_indices = np.random.choice(config["batch_size"], num_anomalies, replace=False)
 
        for idx in anomaly_indices:
            generated_waveforms[idx] += np.random.normal(0, 0.5, 20480)  # Add Gaussian noise as anomaly
 
        # Get a random batch of waveforms from the training set
        idx = np.random.randint(0, waveform_data.shape[1], config["batch_size"])
        real_waveforms = waveform_data[:,idx]
        real_waveforms =real_waveforms.transpose()
        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(real_waveforms, np.ones((config["batch_size"], 1)))
        d_loss_fake = discriminator.train_on_batch(generated_waveforms, np.zeros((config["batch_size"], 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
 
        # Train the generator
        g_loss = combined.train_on_batch(noise, np.ones((config["batch_size"], 1)))
 
        # Print progress
        print(f"Epoch {epoch+1}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")
  
    
   
        g_loss1= pd.DataFrame([g_loss])
        g_df=pd.concat([g_df, g_loss1], axis= 1)
        d_loss1= pd.DataFrame([d_loss])
        d_df=pd.concat([d_df, d_loss1], axis= 1)
   
        # If at save interval, save generated waveforms
        if (epoch + 1) % 10 == 0:
            # Generate some example waveforms
            examples = generator.predict(np.random.normal(0, 1, (10, noise_dim)))
 
            # # Plot examples
            # plt.figure(figsize=(10, 5))
            # for i in range(examples.shape[0]):
                #     plt.subplot(2, 5, i + 1)
                #     plt.plot(examples[i])
                #     plt.axis('off')
                # plt.tight_layout()
                # plt.savefig(f"generated_waveforms_epoch_{epoch+1}.png")
                # plt.close()
               
        g_np = g_df.values
        d_np = d_df.values
        g_np=g_np.transpose()
        d_np=d_np.transpose()      
        #figsize= (14, 6)       
        plt.plot(d_np, linestyle='-')
        plt.plot(g_np, linestyle='--')
        plt.show()
       
        return g_df, d_df
   
def run(config):
   
    tf.keras.backend.clear_session()
   
    (generation_loss, discrimination_loss) = build_and_train_model(config)
    print("run")
    return -discrimination_loss[-1]
 
 
 
if __name__ == "__main__": 
    evaluator = Evaluator.create(
         run,
         method="serial",
         method_kwargs={
             "num_workers": 1,
             }
         )  
    problem = HpProblem()
    problem.add_hyperparameter((10,128), "units", default_value=128)
    problem.add_hyperparameter(["sigmoid", "tanh", "relu"], "generator_activation", default_value = "tanh")
    problem.add_hyperparameter(["sigmoid", "tanh", "relu"], "discriminator_activation", default_value = "tanh")
    problem.add_hyperparameter([1e-4, 1e-2, "log-uniform"], "learning_rate", default_value = 1e-2)
    problem.add_hyperparameter((2,32), "batch_size", default_value = 32)
    problem.add_hyperparameter((0.0,0.5), "dropout_rate", default_value = 0.0)
    problem.add_hyperparameter((1,3), "num_layers", default_value= 1)
    problem.add_hyperparameter((100,1000), "epochs", default_value= 100)
    problem
       
 
 
    search = CBO(problem , evaluator)#, log_dir="cbo_results", random_state = 42)
 
    results = search.search(max_evals =1)
 