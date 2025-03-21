import sys
import os

import logging
logging.basicConfig(level=logging.INFO)

import csv
import datetime

from pathlib import Path

import matplotlib
if sys.platform == "darwin":
    logging.info("On MacOSX, using macos backend")
    matplotlib.use("MacOSX")
else:
    logging.info("On windows or linux, using QtAgg backend")
    matplotlib.use("QtAgg")

import matplotlib.pyplot as plt

import numpy as np


class Impedance:
    def __init__(self, data_folder):
        self.path = Path(data_folder)
        logging.info(f'reading {self.path}')
        files = sorted(os.listdir(self.path))
        files = [os.path.join(self.path, x) for x in files if x.endswith(".spec")]

        timestamps = []
        all_real_ohm = []
        all_imag_ohm = []
        for file in files:
            logging.info(file)
            with open(file, "r") as csv_f:
                unknown = csv_f.readline().strip("\n")
                fn = csv_f.readline().strip("\n")
                offset = csv_f.readline().strip("\n")
                channels = csv_f.readline().strip("\n")
                if channels == "Overvoltage detected":
                    channels = csv_f.readline().strip("\n")
                    logging.warning("Overvoltage detected")
                    #continue
                timestamp = csv_f.readline().strip("\n")
                timestamp = datetime.datetime.strptime(timestamp, '%d-%b-%Y %I:%M:%S:%f %p')
                timestamps.append(timestamp)

                freq = []
                real_ohm = []
                imag_ohm = []

                reader = csv.DictReader(csv_f)
                for row in reader:
                    freq.append(float(row['frequency[Hz]']))
                    real_ohm.append(float(row['Re[Ohm]']))
                    imag_ohm.append(float(row['Im[Ohm]']))

            all_real_ohm.append(real_ohm)
            all_imag_ohm.append(imag_ohm)

        self.freq = np.array(freq)
        first_timestamp = min(timestamps)
        time = [(x-first_timestamp).total_seconds() for x in timestamps]

        sort_idx = sorted(range(len(time)), key=lambda k: time[k])

        self.timestamps = np.array(time)[sort_idx]
        self.real_ohm_np = np.array(all_real_ohm)[sort_idx]
        self.imag_ohm_np = np.array(all_imag_ohm)[sort_idx]
        self.abs_ohm_np = np.sqrt(self.real_ohm_np ** 2 + self.imag_ohm_np ** 2)


def main():
    DATA_FOLDER = input("Please enter the root folder: ")

    GROUP_1 = input("Please enter GROUP_1 or z for 'EXT2 1 2 5 6': ")

    if GROUP_1 == "z":
        GROUP_1 = 'EXT2 1 2 5 6'

    GROUP_2 = input("Please enter GROUP_2 or z for 'EXT2 9 10 13 14': ")

    if GROUP_2 == "z":
        GROUP_2 = 'EXT2 9 10 13 14'

    data_folder_1 = Path(DATA_FOLDER) / GROUP_1
    data_folder_2 = Path(DATA_FOLDER) / GROUP_2

    imp1 = Impedance(data_folder_1)
    imp2 = Impedance(data_folder_2)

    while True:
        fig, axs = plt.subplots(2, sharex='all')

        axs[0].imshow(np.log(imp1.abs_ohm_np.T),
                      extent=[min(imp1.timestamps),
                              max(imp1.timestamps),
                              min(imp1.freq),
                              max(imp1.freq)],
                      aspect='auto')

        axs[1].imshow(np.log(imp2.abs_ohm_np.T),
                      extent=[min(imp2.timestamps),
                              max(imp2.timestamps),
                              min(imp2.freq),
                              max(imp2.freq)],
                      aspect='auto')

        plt.show()

        input_freq = input("Pick a frequency (Hz) [or x to exit]: ")
        if input_freq == "x":
            break

        input_freq = float(input_freq)
        imp1_freq_idx = np.argmin(np.abs(imp1.freq - input_freq))
        imp2_freq_idx = np.argmin(np.abs(imp2.freq - input_freq))

        print(f"Nearest imp1 freq is: {imp1.freq[imp1_freq_idx]}")
        print(f"Nearest imp2 freq is: {imp2.freq[imp2_freq_idx]}")

        while True:
            fig, axs = plt.subplots(2, sharex='all')

            axs[0].plot(imp1.abs_ohm_np[:, imp1_freq_idx])
            axs[1].plot(imp2.abs_ohm_np[:, imp2_freq_idx])

            plt.show()

            start_time = input("Start time (s) [or x to exit]: ")
            if start_time == "x":
                break

            start_time = float(start_time)

            end_time = input("End time(s) [or x to exit]: ")
            if end_time == "x":
                break
            end_time = float(end_time)

            imp1_start_time_idx = np.argmin(np.abs(imp1.timestamps - start_time))
            imp1_end_time_idx = np.argmin(np.abs(imp1.timestamps - end_time))

            imp2_start_time_idx = np.argmin(np.abs(imp2.timestamps - start_time))
            imp2_end_time_idx = np.argmin(np.abs(imp2.timestamps - end_time))

            imp1_real_mean = np.mean(imp1.real_ohm_np[imp1_start_time_idx: imp1_end_time_idx, imp1_freq_idx])
            imp1_imag_mean = np.mean(imp1.imag_ohm_np[imp1_start_time_idx: imp1_end_time_idx, imp1_freq_idx])
            imp1_abs_mean = np.mean(imp1.abs_ohm_np[imp1_start_time_idx: imp1_end_time_idx, imp1_freq_idx])

            imp2_real_mean = np.mean(imp2.real_ohm_np[imp2_start_time_idx: imp2_end_time_idx, imp2_freq_idx])
            imp2_imag_mean = np.mean(imp2.imag_ohm_np[imp2_start_time_idx: imp2_end_time_idx, imp2_freq_idx])
            imp2_abs_mean = np.mean(imp2.abs_ohm_np[imp2_start_time_idx: imp2_end_time_idx, imp2_freq_idx])

            print(f"Imp1 Real mean: {imp1_real_mean}")
            print(f"Imp1 Imag mean: {imp1_imag_mean}")
            print(f"Imp1 Abs mean: {imp1_abs_mean}")

            print(f"Imp2 Real mean: {imp2_real_mean}")
            print(f"Imp2 Imag mean: {imp2_imag_mean}")
            print(f"Imp2 Abs mean: {imp2_abs_mean}")

            input("Hit enter when ready for next measurement <enter>")


if __name__ == "__main__":
    main()
