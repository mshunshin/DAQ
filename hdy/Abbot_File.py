import os
import logging
import zipfile

import scipy.signal
from scipy.interpolate import interp1d

from collections import defaultdict

import numpy as np
import pandas as pd


class Abbot_File():
    def __init__(self, file_path):

        # ["ecg", "boxa", "boxb", "BP", "bpao", "plethg", "plethi", "plethr", "plethh", "qfin"]

        self.file_path = file_path

        #self.sources = ["ecg", "boxa", "boxb", "BP", "bpao", "plethg", "plethi", "plethr", "plethh", "qfin"]
        self.sources = ["bpao", "BP", "ecg", "boxa"]

        data_table = pd.read_table(file_path)
        self.sampling_rate = int(1/np.median(np.diff(data_table['Time'])))
        self.bpao = np.array(data_table['Pd'])
        self.BP = np.array(data_table['Pa'])
        self.ecg = np.array(data_table['ECG'])
        self.boxa = np.array(data_table['Heart rate'])
        self.blank = np.array(np.zeros_like(self.bpao))

        self.zip_fl = file_path


if __name__ == "__main__":
    test_file = "c:\\BVP 80.txt"
    daq_file = Abbot_File(test_file)
    print(daq_file.bpao)
