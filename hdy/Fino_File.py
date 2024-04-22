import os
import zipfile

import scipy.signal
from scipy.interpolate import interp1d

from collections import defaultdict

import numpy as np
import pandas as pd

# rebap = 200

def first_substring(strings, substring):
    return next(i for i, string in enumerate(strings) if substring in string)


class Fino_File():
    def __init__(self, zip_dir, zip_fn, channels = None, search_for_files = False):

        if channels is None:
            channels = {"ECG II.csv": {"name":"ecg", "up": 10, "down":3},
                        "reBAP.csv": {"name":"BP", "up": 5, "down":1}}

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

            self.sampling_rate = 1000

            zip_file_list = [x.filename for x in zip_f.filelist]

            for file, file_hints in channels.items():
                try:
                    file_index = first_substring(zip_file_list, file)
                    file_f = zip_f.open(zip_file_list[file_index], 'r')
                    file_d = self.fast_load_txt(file_f, sep=";")
                    file_d = scipy.signal.resample_poly(file_d, up=file_hints['up'], down=file_hints['down'])
                    print(f'file:{file}, shape{file_d.shape}')

                    if file == 'ld':
                        ld_inter = interp1d(np.arange(file_d.shape[0]), file_d)
                        out_x = np.linspace(0, file_d.shape[0]-1, self.ecg.shape[0])

                        file_d = ld_inter(out_x)

                    setattr(self, file_hints["name"], file_d)
                except Exception as e:
                    print("Exception", e)
                else:
                    sources.append(file_hints["name"])

        self.sources = sources
        min_len = min([len(getattr(self, x)) for x in self.sources])

        for source in sources:
            temp = getattr(self, source)[:min_len]
            setattr(self, source, temp)

        self.blank = np.zeros_like(getattr(self, "ecg"))

    @staticmethod
    def fast_load_txt(file_f, sep=" "):
        return(np.array(pd.read_csv(file_f, sep=sep, dtype=np.float32, header=None, skiprows=8, usecols=[0,1]).iloc[:, 1]))

if __name__ == "__main__":
    f = Fino_File(zip_dir="/Volumes/Matt-Temp/fino-test", zip_fn="2023-01-24_15.40.55_HUTT (csv) Raw.zip")
    print(f)
