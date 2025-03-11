# import libraries
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
from numpy.random import seed
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras.models import load_model
from keras import regularizers,optimizers, Sequential


class ChatterAutoEncoder:


    def __init__(self):
        #import saved model
        model_load = load_model('AutoEncoderModel_RolandTL_Data_AverageExtraLayers.h5')
        self.model = Sequential()
        for layer in model_load.layers: #load model in and play with layers
            self.model.add(layer)

    def load_data(self, the_data, smooth_by=7):
        #read in sensor reads
        data_accel = pd.DataFrame(the_data)
        data_accel_mean_abs = np.array(data_accel.abs().rolling(window=smooth_by).mean())
        data_accel_mean_abs = pd.DataFrame(data_accel_mean_abs.reshape(-1,3))
        data_accel_mean_abs = data_accel_mean_abs.fillna(0)
        data_accel_mean_abs.columns = ['x_accel', 'y_accel', 'z_accel']
        self.data = data_accel_mean_abs
        scaler=MinMaxScaler()
        self.X_data = scaler.fit_transform(self.data)
        self.X_data = self.X_data.reshape(self.X_data.shape[0], 1, self.X_data.shape[1])
        
    def predict_chatter(self, data_size, threshold=0.15):
        # calculate the loss on the data set
        X_pred = self.model.predict(self.X_data)
        X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
        X_pred = pd.DataFrame(X_pred, columns=self.data.columns)
        X_pred.index = self.data.index
        self.scored = pd.DataFrame(index=self.data.index)
        Xdata = self.X_data.reshape(self.X_data.shape[0], self.X_data.shape[2])
        self.scored['Loss_mae'] = np.mean(np.abs(X_pred-Xdata), axis = 1)
        self.scored['Threshold'] = threshold
        self.scored['Anomaly'] = self.scored['Loss_mae'] > self.scored['Threshold']
        total = 0
        CUT_FACTOR = 7 # due to the rolling window and smoothing the first few loss readings do not change, hence we are ignoring them
        SIZE_OF_INPUT = 63
        for i in range(CUT_FACTOR, SIZE_OF_INPUT): #loop through all of the valid acceleration readings
            if self.scored.loc[i][2]: #if model determines there is chatter
                total += 1 #up the count of chatter readings
        if total >= ((SIZE_OF_INPUT-CUT_FACTOR)*0.70): #if more than 70% readings are chatter then this set is chatter
            return True #return chatter
        return False #return not chatter


    def plot_chatter(self):
        self.scored.plot(logy=True,  figsize=(16,9), color=['blue','green'])
        plt.xlabel('Time (ms)',fontsize = 20)
        plt.ylabel('Reconstruction Loss',fontsize = 20)
        plt.legend(fontsize = 20)
        plt.show()