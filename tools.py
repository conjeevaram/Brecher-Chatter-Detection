import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy import signal
import pandas as pd

class ChatterDetector:
    def __init__(self, spindle_speed):
        self.spindle_speed = spindle_speed
        self.revolution_time = 60 / spindle_speed
        self.chatter_threshold = 0.8  # Empirical threshold for wood
        self.f_sample = None
        self.time = None
        self.dispX = None
        self.dispY = None
        self.bisection_indices = []

    def load_data(self, filepath, column_order="TXYZ"):
        data = pd.read_csv(filepath)
        cols = list(data.columns)
        
        self.time = data[cols[column_order.find("T")]].values.astype(float)
        accelX = data[cols[column_order.find("X")]].values.astype(float)
        accelY = data[cols[column_order.find("Y")]].values.astype(float)
        
        # Remove time offset
        self.time -= self.time[0]
        self.f_sample = int(len(self.time) / (self.time[-1] - self.time[0]))
        
        # Process signals
        self.accelX, self.accelY = self._filter_signals(accelX, accelY)
        self.dispX, self.dispY = self._integrate_signals()
        self.bisection_indices = self._find_bisection_points()

    def _filter_signals(self, accelX, accelY):
        """Bandpass filter 50-330Hz for wood chatter"""
        nyq = 0.5 * self.f_sample
        low = 50 / nyq
        high = 330 / nyq
        
        sos = signal.butter(4, [low, high], btype='band', output='sos')
        return (
            signal.sosfiltfilt(sos, accelX),
            signal.sosfiltfilt(sos, accelY)
        )

    def _integrate_signals(self):
        """Double integration with drift removal"""
        velX = cumtrapz(self.accelX, self.time, initial=0)
        velY = cumtrapz(self.accelY, self.time, initial=0)
        
        # linear detrending
        velX = signal.detrend(velX, type='linear')
        velY = signal.detrend(velY, type='linear')
        
        dispX = cumtrapz(velX, self.time, initial=0)
        dispY = cumtrapz(velY, self.time, initial=0)
        return (
            signal.detrend(dispX, type='linear'),
            signal.detrend(dispY, type='linear')
        )

    def _find_bisection_points(self):
        """Find indices for each revolution"""
        indices = []
        current_time = 0
        while current_time <= self.time[-1]:
            idx = np.abs(self.time - current_time).argmin()
            indices.append(idx)
            current_time += self.revolution_time
        return indices

    def calculate_chatter_indicators(self, window_size=0.3, step_size=0.1):
        """Calculate CI using sliding window"""
        window_samples = int(window_size * self.f_sample)
        step_samples = int(step_size * self.f_sample)
        
        ci_times = []
        ci_values = []
        
        for w_start in range(0, len(self.time)-window_samples, step_samples):
            w_end = w_start + window_samples
            
            # Get bisection points in window
            bis_points = [i for i in self.bisection_indices 
                        if w_start <= i < w_end]
            
            if len(bis_points) < 2:
                ci_values.append(0)
                ci_times.append(self.time[w_start + window_samples//2])
                continue
                
            # Calculate standard deviations
            bis_x = self.dispX[bis_points]
            bis_y = self.dispY[bis_points]
            traj_x = self.dispX[w_start:w_end]
            traj_y = self.dispY[w_start:w_end]
            
            ci = (np.std(bis_x) * np.std(bis_y)) / \
                (np.std(traj_x) * np.std(traj_y) + 1e-9)
            
            ci_values.append(ci)
            ci_times.append(self.time[w_start + window_samples//2])
        
        return ci_times, ci_values