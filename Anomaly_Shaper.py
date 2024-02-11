# -*- coding: utf-8 -*-



def Shaper():

    import glob
    import pandas as pd

    # Get data file names
    path1 = '/Users/nathanaelseay/Desktop/HomeProjects/LSTM-Autoencoder-for-Anomaly-Detection/Sensor Data/Bearing_Sensor_Data_pt1'
    path2 = '/Users/nathanaelseay/Desktop/HomeProjects/LSTM-Autoencoder-for-Anomaly-Detection/Sensor Data/Bearing_Sensor_Data_pt2'


    filenames1 = glob.glob(path1 + "/*.39")
    filenames2 = glob.glob(path2 + "/*.39")
    filenames= filenames1+filenames2

    dfs = pd.DataFrame()
    for filename in filenames:
        df=pd.read_csv(filename, sep='\t', header= None)
    
        df1 = df.iloc[0]
    
        dfs= pd.concat([dfs, df1], axis = 1)
    
    melt = pd.melt(dfs)