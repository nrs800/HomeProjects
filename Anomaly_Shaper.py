# -*- coding: utf-8 -*-



def Shaper(bearing):

    import glob
    import pandas as pd

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
        
        df4 = df.iloc[:, lambda df: [4]]
    
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
        
    return melt


Data = Shaper(1)    
