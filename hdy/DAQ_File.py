import os
import zipfile

import scipy.signal
from scipy.interpolate import interp1d

from collections import defaultdict

import numpy as np
import pandas as pd


class DAQ_File():
    def __init__(self, zip_dir, zip_fn, channels = None, search_for_files = False):

        if channels is None:
            channels = ["ecg", "boxa", "boxb", "BP", "bpao", "plethg", "plethi", "plethr", "plethh", "qfin",
                     "Resp", "pot", "flow", "spO2", "pot", "bp_prox", "bp_dist", "bp", "ld"]

        sources = []

        if search_for_files:

            file_index = defaultdict(str)

            for root, dirs, files in os.walk(zip_dir, topdown=False):
                for name in files:
                    fn = name
                    fl = os.path.join(root, name)
                    file_index[fn] = fl

            self.zip_fl = file_index[str(zip_fn)]

        else:

            self.zip_fl = os.path.join(zip_dir, zip_fn)


        print("Opening DAQ File: ", self.zip_fl)

        with zipfile.ZipFile(self.zip_fl, "r") as zip_f:

            sampling_f = zip_f.open("rate.txt", 'r')

            self.sampling_rate = int(np.loadtxt(sampling_f))

            for file in channels:
                try:
                    file_f = zip_f.open(file + ".txt", 'r')
                    file_d = self.fast_load_txt(file_f)
                    print(f'file:{file}, shape{file_d.shape}')

                    if file == 'ld':
                        ld_inter = interp1d(np.arange(file_d.shape[0]), file_d)
                        out_x = np.linspace(0, file_d.shape[0]-1, self.ecg.shape[0])

                        file_d = ld_inter(out_x)

                    setattr(self, file, file_d)
                except Exception as e:
                    print("Exception", e)
                else:
                    sources.append(file)

        self.sources = sources
        self.blank = np.zeros_like(getattr(self, channels[0]))

    @staticmethod
    def fast_load_txt(file_f):
        return(np.array(pd.read_csv(file_f, delimiter=' ', dtype=np.float, header=None).iloc[:, 0]))

