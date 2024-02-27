#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 18:28:16 2024

@author: nathanaelseay
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
tf.keras.utils.set_random_seed(10)
from tensorflow.keras import layers
from keras_tuner.tuners import RandomSearch

def file_call():

    def import_files_in_directory(directory):
        imported_data = []

        for root, dirs, files in os.walk(directory):
            for file_name in files:
                if file_name.endswith('.txt'):  # Process only text files
                    file_path = os.path.join(root, file_name)
                    # with open(file_path, 'r') as file:
                    #     data = file.read()
                    imported_data.append(file_path)  # Store file path and data

        return imported_data
    
    current_directory = os.getcwd()  # Get the current working directory
    imported_files = import_files_in_directory(current_directory)
    
    return imported_files

def Shaper(bearing):
    
    filenames = import_files
    
    dfs1 = pd.DataFrame()
    dfs2 = pd.DataFrame()
    dfs3 = pd.DataFrame()
    dfs4 = pd.DataFrame()
    
    for filename in filenames:
        df=pd.read_csv(filename, sep='\t', header= None)
    
        #df1 = df.iloc[0]
        df1 = df.iloc[:, lambda df: [0]]
        dfs1= pd.concat([dfs1, df1], axis = 1)
        
        df2 = df.iloc[:, lambda df: [1]]
        dfs2= pd.concat([dfs2, df2], axis = 1)
        
        df3 = df.iloc[:, lambda df: [2]]
        dfs3= pd.concat([dfs3, df3], axis = 1)
        
        df4 = df.iloc[:, lambda df: [3]]
        dfs4= pd.concat([dfs4, df4], axis = 1)

    if (bearing==1):
        melt = pd.melt(dfs1)
        
    if (bearing==2):
        melt = pd.melt(dfs2)
        
    if (bearing==3):
        melt = pd.melt(dfs1)
            
    if (bearing==4):
        melt = pd.melt(dfs2)
        
    melt = melt["value"]
        
    return melt