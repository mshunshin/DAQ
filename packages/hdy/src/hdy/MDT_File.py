import os
from pathlib import Path

from collections import defaultdict

import numpy as np
import pandas as pd

# rebap = 200

def first_substring(strings, substring):
    return next(i for i, string in enumerate(strings) if substring in string)


class MDT_File():
    def __init__(self, zip_dir, zip_fn, channels = None, search_for_files = False):

        channels = {"Time (s)": {"name":"time"},
                    "DOP_FLOW (Volts)":{"name": "flow"},
                    "Z (Ohms)": {"name": "imp"},
                    "AO (mmHg)": {'name': "bp"},
                    "ITP (mmHg)": {},
                    "SCAT (Volts)": {"name": "laser1"},
                    "AC 400 (Volts)": {},
                    "I (Volts)": {"name": "ecg"},
                    "Phase (Degrees)": {},
                    "II (Volts)": {},
                    "AM1 (Volts)": {"name": "ppg_amber"},
                    "GR1 (Volts)": {"name": "ppg_green"},
                    "IR1 (Volts)": {"name": "ppg_infrared"},
                    "III (Volts)": {},
                    "BioZ (cnt)": {"name": "imp2"},
                    "egm1 (cnt)": {},
                    "HeartSound (cnt)": {},
                    }

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

        self.sampling_rate = 1000
        self.sources = []

        d_pd = pd.read_csv(self.zip_fl)

        for channel, channel_data in channels.items():
            try:
                channel_name = channel_data.get("name", None)
                channel_data = d_pd[channel]
            except Exception:
                continue

            if channel_name is not None:
                self.sources.append(channel_name)

                setattr(self, channel_name, channel_data)

        self.blank = np.zeros_like(getattr(self, "ecg"))

    @staticmethod
    def fast_load_txt(file_f, sep=" "):
        return(np.array(pd.read_csv(file_f, sep=sep, dtype=np.float32, header=None, skiprows=8, usecols=[0,1]).iloc[:, 1]))


if __name__ == "__main__":
    file_path = Path("/Volumes/Matt-Temp/Study Data (25-Oct)/Animal 351736 Data0007 (Run 1)_3.9.csv")
    #d_pd = pd.read_csv(file_path)
    #print("helo")
    d = MDT_File(file_path.parent, file_path.name)
    print("hello")
