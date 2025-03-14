import pandas as pd
from ChatterAutoEncoder import ChatterAutoEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import sys
import time


class ChatterAutoEncoderRefactored(ChatterAutoEncoder):

    def load_data(self, csv, smooth_by=7):
        """
        Loads sensor data from a CSV file or DataFrame.
        Expects CSV to have columns: T, X, Y, Z.
        Only the X, Y, Z columns are used.
        """
        # Read in the CSV file.
        if isinstance(csv, str):
            data = pd.read_csv(csv)
        else:
            data = pd.DataFrame(csv)

        # Select only the sensor columns: X, Y, and Z.
        data_accel = data[['X', 'Y', 'Z']]

        # Apply rolling window smoothing on the absolute values.
        data_accel_mean_abs = data_accel.abs().rolling(window=smooth_by).mean().fillna(0)
        data_accel_mean_abs.columns = ['X', 'Y', 'Z']

        # Save processed data.
        self.data = data_accel_mean_abs

        # Scale the data using MinMaxScaler.
        scaler = MinMaxScaler()
        self.X_data = scaler.fit_transform(self.data)

        # Reshape for the model (samples, time steps, features).
        self.X_data = self.X_data.reshape(self.X_data.shape[0], 1, self.X_data.shape[1])

    def predict_chatter(self, threshold=0.15, k=20.0):
        """
        Computes a continuous anomaly score for each sample.

        The reconstruction loss is computed, and then mapped to an anomaly
        score using a sigmoid function:

            score = 1 / (1 + exp(-k*(loss - threshold)))

        This results in a continuous value between 0 and 1, where a score of
        0 indicates 0% chance of chatter and a score of 1 indicates 100% chance.
        """
        # Get predictions from the autoencoder.
        X_pred = self.model.predict(self.X_data)
        X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
        X_pred = pd.DataFrame(X_pred, columns=self.data.columns)
        X_pred.index = self.data.index

        self.scored = pd.DataFrame(index=self.data.index)
        # Reshape original data back to 2D.
        Xdata = self.X_data.reshape(self.X_data.shape[0], self.X_data.shape[2])
        self.scored['Loss_mae'] = np.mean(np.abs(X_pred - Xdata), axis=1)
        self.scored['Threshold'] = threshold

        # Map the loss to a continuous anomaly score using a sigmoid.
        self.scored['Anomaly_Score'] = 1.0 / (1.0 + np.exp(-k * (self.scored['Loss_mae'] - threshold)))
        return self.scored['Anomaly_Score']

    # def plot_chatter(self):
    #     """
    #     Plots the continuous anomaly score (from 0 to 1) over time.
    #     """
    #     plt.figure(figsize=(16, 9))
    #     plt.plot(self.scored.index, self.scored['Anomaly_Score'], color='purple', label='Anomaly Score')
    #     plt.xlabel('Time (ms)', fontsize=20)
    #     plt.ylabel('Anomaly Score (0 to 1)', fontsize=20)
    #     plt.title('Continuous Chatter Prediction Score', fontsize=22)
    #     plt.legend(fontsize=20)
    #     plt.ylim(0, 1)
    #     plt.show()

    def plot_chatter(self):
        """
        Plots a square wave for chatter detection using a direct threshold.
        If the reconstruction error exceeds the threshold, the output is 1;
        otherwise, it is 0.
        """
        # Create a binary signal: 1 when Loss_mae > Threshold, else 0.
        binary_signal = (self.scored['Loss_mae'] > self.scored['Threshold']).astype(int)
        
        plt.figure(figsize=(16, 9))
        # Use a step plot to produce the square wave appearance.
        plt.step(self.scored.index, binary_signal, where='post', color='purple', label='Chatter (Binary)')
        plt.xlabel('Time (ms)', fontsize=20)
        plt.ylabel('Chatter (0 or 1)', fontsize=20)
        plt.title('Square Wave Chatter Prediction (Threshold Based)', fontsize=22)
        plt.legend(fontsize=20)
        plt.ylim(-0.1, 1.1)
        plt.show()



if __name__ == "__main__":
    # Example usage: load a CSV file with columns T, X, Y, Z.
    csv_path = "C:/LJM-Data-Collection/csv/8500_1.csv"

    # Create an instance of the refactored chatter autoencoder.
    chatter_detector = ChatterAutoEncoderRefactored()

    # Load the data from the CSV.
    chatter_detector.load_data(csv_path, smooth_by=7)

    # Run prediction using a threshold of 0.15.
    anomaly_scores = chatter_detector.predict_chatter(threshold=0.30, k=20.0)

    # Print a few anomaly scores.
    print("First few anomaly scores:")
    print(anomaly_scores.head())

    # Display the continuous anomaly score plot.
    chatter_detector.plot_chatter()