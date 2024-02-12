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
#from sklearn.externals import joblib
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
#%matplotlib inline

from numpy.random import seed
import tensorflow as tf
#from tensorflow import set_random_seed



tf.keras.utils.set_random_seed(10)

# random.seed(seed)
# np.random.seed(seed)
# tf.random.set_seed(seed)

#tf.logging.set_verbosity(tf.logging.ERROR)


from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers

bearing = 4

wave_data = Shaper(bearing)

fig, ax = plt.subplot()
ax.plot(wave_data, label = 'bearing', linewidth = 1)

plot = wave_data.plot()



train_amount= len(wave_data)*.8

test_amount=len(wave_data)

train_amount = int(train_amount)

test_amount = int(test_amount)

train = wave_data.iloc[1:train_amount]

test = wave_data.iloc[train_amount:test_amount]


print("Training dataset shape:", train.shape)
print("Test dataset shape:", test.shape)