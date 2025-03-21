import numpy as np
import mmt
from .data_reader import Data_Reader
from .filters import *


class ElectrogramDetector:
    def __init__(self):
        self.buffer = np.zeros(2000)
        self.T = 1.2  # Sample Period
        self.fs = 1000.0  # sample rate, Hz
        self.cutoff = 20  # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
        self.nyq = 0.5 * self.fs  # Nyquist Frequency
        self.order = 3  # sin wave can be approx represented as quadratic
        self.n = int(self.T * self.fs)
        self.detected_qrs = []

    def load_new_data(self, electrogram: Data_Reader):
        self.buffer[0:1800] = self.buffer[200:2000]
        self.buffer[1800:2000] = electrogram.get_next_data(amount=200)

    def detect_new_data(self):
        buffer = self.buffer
        out1 = filters.butter_lowpass_filter(data=buffer, cutoff=self.cutoff, fs=self.fs, order=self.order)
        out2 = filters.butter_highpass_filter(data=out1, cutoff=self.cutoff, fs=self.fs, order=self.order)
        out_mean = np.mean(out2)
        out = np.abs(out2 - out_mean)
        out = np.convolve(out, np.ones(111, dtype=np.int), 'same')
        raw = buffer[100:]
        return out, raw