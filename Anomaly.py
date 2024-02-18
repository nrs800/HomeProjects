#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 13:03:40 2024

@author: nathanaelseay
"""
# import libraries
import pandas as pd
# from Anomaly_Shaper import Shaper
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


def Shaper(bearing):

    import glob
    #import pandas as pd

    # Get data file names
    path1 = '/Users/nathanaelseay/Desktop/HomeProjects/LSTM-Autoencoder-for-Anomaly-Detection/Sensor Data/Bearing_Sensor_Data_pt1'
    path2 = '/Users/nathanaelseay/Desktop/HomeProjects/LSTM-Autoencoder-for-Anomaly-Detection/Sensor Data/Bearing_Sensor_Data_pt2'


    filenames1 = glob.glob(path1 + "/*.39")
    filenames2 = glob.glob(path2 + "/*.39")
    filenames= filenames1+filenames2

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
    
    # melt1 = pd.melt(dfs1)
    # melt2 = pd.melt(dfs2)
    # melt3 = pd.melt(dfs3)
    # melt4 = pd.melt(dfs4)
    
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

# define the autoencoder network model
def autoencoder_model(X):
    inputs = Input(shape=(X.shape[1], 1))  
    L1 = LSTM(16, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(1))(L5)  
    model = Model(inputs=inputs, outputs=output)
    return model


model = autoencoder_model(X_train)
model.compile(optimizer='adam', loss='mae')
model.summary()

nb_epochs = 15
batch_size = 10
history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size,
                    validation_split=0.05).history

fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(history['loss'], 'b', label='Train', linewidth=2)
ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
ax.set_title('Model loss', fontsize=16)
ax.set_ylabel('Loss (mae)')
ax.set_xlabel('Epoch')
ax.legend(loc='upper right')
plt.show()


X_pred = model.predict(X_train)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred)
#X_pred.index = X_train.index


scored = pd.DataFrame()
#Xtrain = X_train.reshape(X_train.shape[0], X_train.shape[2])
scored['Loss_mae'] = np.mean(np.abs(X_pred-X_train), axis = 1)

min_value = scored['Loss_mae'].min()
max_value = scored['Loss_mae'].max()

range_ = max_value - min_value


range_of=  range_*.98

threshold = min_value+ range_of



X_pred = model.predict(X_test)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred)
X_pred.index = test.index

scored = pd.DataFrame(index=test.index)
Xtest = X_test.reshape(X_test.shape[0], X_test.shape[1])
scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtest), axis = 1)
scored['Threshold'] = threshold
scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
scored.head()


def traceback(scored):
    
    #index files
    #take in scored array
    # add a column indicating which file it came from 
    #if anomaly then return file
    #delete repeats 
    import glob
    import pandas as pd
    import numpy as np

    # Get data file names
    path1 = '/Users/nathanaelseay/Desktop/HomeProjects/LSTM-Autoencoder-for-Anomaly-Detection/Sensor Data/Bearing_Sensor_Data_pt1'
    path2 = '/Users/nathanaelseay/Desktop/HomeProjects/LSTM-Autoencoder-for-Anomaly-Detection/Sensor Data/Bearing_Sensor_Data_pt2'


    filenames1 = glob.glob(path1 + "/*.39")
    filenames2 = glob.glob(path2 + "/*.39")
    filenames= filenames1+filenames2
    
    sample_rate= 20480
    entry_number= len(scored['Loss_mae'])
    increment= entry_number/sample_rate
    #every increment add 1 to file_number
    fileID = []
    j = 1
    for i in range(0, entry_number):
        if i % increment == 0:  # Check if the index is a multiple of the increment
            j += 1
        fileID.append(j)
    scored['File_ID'] = fileID
    ID_scored = scored.loc[scored['Anomaly'] == True]
    
    
    return ID_scored

ID_scored=traceback(scored)
 
 



















