#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 13:03:40 2024

@author: nathanaelseay
"""
# import libraries
import pandas as pd
from Anomaly_Shaper import Shaper
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
# %matplotlib inline

from numpy.random import seed
import tensorflow as tf
# from tensorflow import set_random_seed

tf.keras.utils.set_random_seed(10)

# random.seed(seed)
# np.random.seed(seed)
# tf.random.set_seed(seed)

# tf.logging.set_verbosity(tf.logging.ERROR)

from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers

bearing = 4


wave_data = Shaper(bearing)
plt.plot(wave_data, label='bearing', linewidth=1)
plt.legend()
plt.show()

train_amount = int(len(wave_data) * 0.8)
test_amount = len(wave_data)

train = wave_data.iloc[:train_amount]
test = wave_data.iloc[train_amount:]

print("Training dataset shape:", train.shape)
print("Test dataset shape:", test.shape)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(train.values.reshape(-1, 1)) 
X_test = scaler.transform(test.values.reshape(-1, 1))  
scaler_filename = "scaler_data"
joblib.dump(scaler, scaler_filename)






