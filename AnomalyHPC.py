#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 13:03:40 2024

@author: nathanaelseay
"""
# import libraries
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
import glob

# Get data file names
path1 = '/Users/nathanaelseay/Desktop/HomeProjects/LSTM-Autoencoder-for-Anomaly-Detection/Sensor Data/Bearing_Sensor_Data_pt1'
path2 = '/Users/nathanaelseay/Desktop/HomeProjects/LSTM-Autoencoder-for-Anomaly-Detection/Sensor Data/Bearing_Sensor_Data_pt2'

filenames1 = glob.glob(path1 + "/*.39")
filenames2 = glob.glob(path2 + "/*.39")
filenames= filenames1+filenames2

def Shaper(bearing, filenames):

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

bearing = 4

wave_data = Shaper(bearing, filenames)
plt.plot(wave_data, label='bearing', linewidth=1)
plt.legend()
plt.show()
plt.savefig('Waveform.jpg')
plt.savefig('Waveform.png')

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

def build_autoencoder(hp):
    inputs = keras.Input(shape=(timesteps, input_dim))

    encoder = layers.LSTM(units=hp.Int('encoder_units', min_value=32, max_value=256, step=32),
                          return_sequences=True)(inputs)
    encoder = layers.Dropout(rate=hp.Float('encoder_dropout', min_value=0.0, max_value=0.5, step=0.1))(encoder)

    for i in range(hp.Int('encoder_layers', 1, 3)):
        encoder = layers.LSTM(units=hp.Int(f'encoder_units_layer_{i}', min_value=32, max_value=256, step=32),
                              return_sequences=True)(encoder)
        encoder = layers.Dropout(rate=hp.Float(f'encoder_dropout_layer_{i}', min_value=0.0, max_value=0.5, step=0.1))(encoder)

    decoder = layers.LSTM(units=hp.Int('decoder_units', min_value=32, max_value=256, step=32),
                          return_sequences=True)(encoder)
    decoder = layers.Dropout(rate=hp.Float('decoder_dropout', min_value=0.0, max_value=0.5, step=0.1))(decoder)

    for i in range(hp.Int('decoder_layers', 1, 3)):
        decoder = layers.LSTM(units=hp.Int(f'decoder_units_layer_{i}', min_value=32, max_value=256, step=32),
                              return_sequences=True)(decoder)
        decoder = layers.Dropout(rate=hp.Float(f'decoder_dropout_layer_{i}', min_value=0.0, max_value=0.5, step=0.1))(decoder)

    outputs = layers.TimeDistributed(layers.Dense(input_dim))(decoder)

    autoencoder = keras.Model(inputs, outputs)
    autoencoder.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),loss='mse')
    return autoencoder

# Define the data dimensions
timesteps = X_train.shape[0] #maybe [1]
input_dim = 1

tuner = RandomSearch(
    build_autoencoder,
    objective='val_loss',
    max_trials=5,
    executions_per_trial=3,
    directory='my_dir',
    project_name='lstm_autoencoder_tuning'
)

# Perform the hyperparameter search
tuner.search(x=X_train, y=X_train, epochs=10, validation_split=0.2, batch_size=256)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the best hyperparameters and train it
best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(X_train, X_train, epochs=10, validation_split=0.2, batch_size=256).history

best_model.save('Anomaly_detector.h5')
print('Model Saved!')

fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(history['loss'], 'b', label='Train', linewidth=2)
ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
ax.set_title('Model loss', fontsize=16)
ax.set_ylabel('Loss (mae)')
ax.set_xlabel('Epoch')
ax.legend(loc='upper right')
plt.show()

plt.savefig('Model_loss.jpg')
plt.savefig('Model_loss.png')

X_pred = best_model.predict(X_train)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred)

scored = pd.DataFrame()
scored['Loss_mae'] = np.mean(np.abs(X_pred-X_train), axis = 1)

min_value = scored['Loss_mae'].min()
max_value = scored['Loss_mae'].max()

range_ = max_value - min_value
range_of=  range_*.98
threshold = min_value+ range_of

X_pred = best_model.predict(X_test)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred)
X_pred.index = test.index

scored = pd.DataFrame(index=test.index)
Xtest = X_test.reshape(X_test.shape[0], X_test.shape[1])
scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtest), axis = 1)
scored['Threshold'] = threshold
scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
scored.head()

def traceback(scored, X_train):

    sample_rate= 20480
    entry_number= len(scored['Loss_mae'])
    print(entry_number)
    increment= entry_number/sample_rate
   
    fileID = []
    j = 1
    for i in range(0, entry_number):
        j=j+1
        fileID.append(j)
    scored['File_ID'] = fileID
    #print(fileID)
    ID_scored = scored.loc[scored['Anomaly'] == True]
    ID_scored = ID_scored.drop_duplicates(subset=['File_ID'])
    ID_scored['file_pointer']=(ID_scored['File_ID']+len(X_train))/sample_rate
    return ID_scored

ID_scored=traceback(scored, X_train)
file_pointer=(ID_scored['file_pointer']).astype(int)
file_pointer=file_pointer.drop_duplicates()

f = open("Anomaly_log.txt", "a")

for index in file_pointer:
    if 0 <= index < len(filenames):
        print(f"File at index {index}: {filenames[index]}")
        f.write(f"File at index {index}: {filenames[index]}")
    else:
        print(f"Index {index} is out of range.")

f.close()

















