__author__ = 'matthew'

import os
import sys
import numpy as np
import pandas
import scipy
import scipy.signal
import scipy.fftpack
import matplotlib.pyplot as plt



sys.path.append("/Volumes/Matt-Data/Projects/Matt-Python-Lib/")
import mmt

class LPM(object):
    def __init__(self, ecg_f, fino_f, laser1_f, laser2_f, tcd1_f, tcd2_f, Hz=1000):
        self.ecg = np.array(pandas.read_csv(ecg_f, delimiter=' ', dtype=np.float, header=False).iloc[:,0])
        self.laser1 = np.array(pandas.read_csv(laser1_f, delimiter=' ', dtype=np.float, header=False).iloc[:,0]) * 100
        self.laser2 = np.array(pandas.read_csv(laser2_f, delimiter=' ', dtype=np.float, header=False).iloc[:,0]) * 100
        self.fino = np.array(pandas.read_csv(fino_f, delimiter=' ', dtype=np.float, header=False).iloc[:,0]) * 100
        self.tcd1 = np.array(pandas.read_csv(tcd1_f, delimiter=' ', dtype=np.float, header=False).iloc[:,0])
        self.tcd2 = np.array(pandas.read_csv(tcd2_f, delimiter=' ', dtype=np.float, header=False).iloc[:,0])

        self.laser1[self.laser1<=0] = np.min(self.laser1[self.laser1>0])/2
        self.laser2[self.laser2<=0] = np.min(self.laser2[self.laser2>0])/2
        self.fino[self.fino<=0] = np.min(self.fino[self.fino>0])/2

        self.Hz = Hz

    def set_selection(self, start, end, freq_hint, ecg_hint):
        self.begin_selection = int(start)
        self.end_selection = int(end)
        self.freq_hint = freq_hint
        self.ecg_hint = ecg_hint

        selection = range(self.begin_selection, self.end_selection)
        self.selection = selection

        self.ecg_signal = self.ecg[selection]
        self.ecg_signal = scipy.signal.detrend(self.ecg_signal)

        self.fino_signal = self.fino[selection]
        self.fino_mean = np.mean(self.fino_signal)
        self.fino_signal = np.log(self.fino_signal)
        self.fino_signal = scipy.signal.detrend(self.fino_signal)

        self.laser1_signal = self.laser1[selection]
        self.laser1_mean = np.mean(self.laser1_signal)
        self.laser1_signal = np.log(self.laser1_signal)
        self.laser1_signal = scipy.signal.detrend(self.laser1_signal)

        self.laser2_signal = self.laser2[selection]
        self.laser2_mean = np.mean(self.laser2_signal)
        self.laser2_signal = np.log(self.laser2_signal)
        self.laser2_signal = scipy.signal.detrend(self.laser2_signal)

    def plot_traces(self):
        plt.ioff()
        fig = plt.figure()

        ax_ecg = fig.add_subplot(611)
        ax_ecg.plot(self.ecg[:], "k")
        ax_ecg.set_ylabel("ECG")

        ax_fino = fig.add_subplot(612, sharex=ax_ecg)
        ax_fino.plot(self.fino[:], "r")
        ax_fino.set_ylabel("Fino")

        ax_laser1 = fig.add_subplot(613, sharex=ax_ecg)
        ax_laser1.plot(self.laser1[:], "g")
        ax_laser1.set_ylabel("Laser 1")

        ax_laser2 = fig.add_subplot(614, sharex=ax_ecg)
        ax_laser2.plot(self.laser2[:], "b")
        ax_laser2.set_ylabel("Laser 2")

        ax_tcd1 = fig.add_subplot(615, sharex=ax_ecg)
        ax_tcd1.plot(scipy.signal.savgol_filter(self.tcd1[:], 71,3), "k")
        ax_tcd1.set_ylabel("Cranial 1")

        ax_tcd2 = fig.add_subplot(616, sharex=ax_ecg)
        ax_tcd2.plot(scipy.signal.savgol_filter(self.tcd2[:], 71,3), "k")
        ax_tcd2.set_ylabel("Cranial 2")

        plt.show()

    def calc_fft(self):

        selection = self.selection

        N = len(selection)
        Nq = self.Hz/2
        self.xss = np.fft.rfftfreq(N, 1/self.Hz)[0:N//2]

        self.ecg_fft = np.abs(np.fft.rfft(self.ecg_signal))
        self.fino_fft = np.abs(np.fft.rfft(self.fino_signal))
        self.laser1_fft = np.abs(np.fft.rfft(self.laser1_signal))
        self.laser2_fft = np.abs(np.fft.rfft(self.laser2_signal))

        self.ecg_fft_plt = 2/N * self.ecg_fft
        self.fino_fft_plt = 2/N * self.fino_fft
        self.laser1_fft_plt = 2/N * self.laser1_fft
        self.laser2_fft_plt = 2/N * self.laser2_fft

        ###With biological signals you oftern end up with lower order harmonics
        ###This shifts it up to the expected frequency.
        self.ecg_fft_plt = 2/N * (np.repeat(self.ecg_fft[0:N//4],2) +self.ecg_fft[0:N//2])
        self.fino_fft_plt = 2/N * (np.repeat(self.fino_fft[0:N//4],2) + self.fino_fft[0:N//2])
        self.laser1_fft_plt = 2/N * (np.repeat(self.laser1_fft[0:N//4],2) + self.laser1_fft[0:N//2])
        self.laser2_fft_plt = 2/N * (np.repeat(self.laser2_fft[0:N//4],2) + self.laser2_fft[0:N//2])

        ecg_peaks = scipy.signal.find_peaks_cwt(self.ecg_fft_plt, widths = np.array([1,2,3,4,5,6,7,8,9,10]), noise_perc=10)
        ecg_peaks = scipy.signal.argrelmax(self.ecg_fft_plt, order=5)[0]

        ecg_peak_freq = mmt.find_nearest_value(self.xss[ecg_peaks], self.freq_hint)
        self.ecg_peak_freq = ecg_peak_freq
        ecg_peak_idx = mmt.find_nearest_idx(self.xss, self.ecg_peak_freq)

        self.ecg_peak_power = self.ecg_fft_plt[ecg_peak_idx]
        self.fino_peak_power = self.fino_fft_plt[ecg_peak_idx]
        self.laser1_peak_power = self.laser1_fft_plt[ecg_peak_idx]
        self.laser2_peak_power = self.laser2_fft_plt[ecg_peak_idx]

        self.ecg_power = np.sum(self.ecg_fft_plt[ecg_peak_idx-1:ecg_peak_idx+2])
        self.fino_power = np.sum(self.ecg_fft_plt[ecg_peak_idx-1:ecg_peak_idx+2])
        self.laser1_power = np.sum(self.laser1_fft_plt[ecg_peak_idx-1:ecg_peak_idx+2])
        self.laser2_power = np.sum(self.laser2_fft_plt[ecg_peak_idx-1:ecg_peak_idx+2])

        self.mean_arterial = np.mean(self.fino[selection])

        print("Peak index; peak frequency")
        print(ecg_peak_idx, self.xss[ecg_peak_idx])
        print("At Peak")
        print(self.ecg_peak_power, self.fino_peak_power, self.laser1_peak_power, self.laser2_peak_power)
        print("Peak+1 either side")
        print(self.ecg_power, self.fino_power, self.laser1_power, self.laser2_power)

    def plot_fft(self):
        plt.ioff()
        plt.plot(self.xss, self.ecg_fft_plt, "k")
        plt.plot(self.xss, self.fino_fft_plt, "r")
        plt.plot(self.xss, self.laser1_fft_plt, "b")
        plt.plot(self.xss, self.laser2_fft_plt, "g")
        plt.xlim(0, 8)
        plt.ylim(0, 2)
        plt.show()

    def plot_selection(self):
        plt.ioff()
        fig = plt.figure()

        ax_ecg = fig.add_subplot(411)
        ax_ecg.plot(self.ecg_signal, "k")
        ax_ecg.plot(self.ecg_peaks, self.ecg_signal[self.ecg_peaks], "ro")
        ax_ecg.set_ylabel("ECG")

        ax_fino = fig.add_subplot(412, sharex=ax_ecg)
        ax_fino.plot(self.fino_signal, "r")
        ax_fino.set_ylabel("Fino")

        ax_laser1_1 = fig.add_subplot(413, sharex=ax_ecg)
        ax_laser1_1.plot(self.laser1_signal, "g")
        ax_laser1_1.set_ylabel("Laser 1")

        ax_laser2_1 = fig.add_subplot(414, sharex=ax_ecg)
        ax_laser2_1.plot(self.laser2_signal, "b")
        ax_laser2_1.set_ylabel("Laser 2")
        plt.show()

    def qrs_sample_detect(self):
        if self.ecg_hint.lower() != "vf":
            ecg_detect = np.sum(rolling_window(np.diff(self.ecg_signal)**2, 150),1)
            ecg_detect[ecg_detect < np.max(ecg_detect)//10] = 0
            ecg_detect_peaks = np.array(scipy.signal.find_peaks_cwt(ecg_detect, widths=np.linspace(100,200,10), noise_perc=30))
            ecg_detect_peaks = ecg_detect_peaks+50

            relmax_peaks = scipy.signal.argrelmax(np.abs(self.ecg_signal), order=10)[0]
            final_peaks = []
            for peak in ecg_detect_peaks:
                final_peaks.append(mmt.find_nearest_value(relmax_peaks, peak))

        if self.ecg_hint.lower() == "vf":
            ecg_detect = np.sum(rolling_window(np.diff(self.ecg_signal)**2, 10),1)
            ecg_detect_peaks = np.array(scipy.signal.find_peaks_cwt(ecg_detect, widths=np.linspace(10,200,10), noise_perc=30))
            ecg_detect_peaks = ecg_detect_peaks+5

            relmax_peaks = scipy.signal.argrelmax(np.abs(self.ecg_signal), order=10)[0]

            final_peaks = []
            for peak in ecg_detect_peaks:
                final_peaks.append(mmt.find_nearest_value(relmax_peaks, peak))

        self.ecg_peaks = final_peaks

    def calc_magic(self):
        mean_RR = int(2*(np.mean(np.diff(self.ecg_peaks))//2))
        half_mean_RR = mean_RR // 2
        laser1_sum = np.zeros(mean_RR)
        laser2_sum = np.zeros(mean_RR)
        inc_peaks = 0

        for peak in self.ecg_peaks:
            if peak>half_mean_RR and peak<len(self.laser1_signal)-half_mean_RR:
                temp1 = self.laser1_signal[peak-half_mean_RR:peak+half_mean_RR]
                temp2 = self.laser2_signal[peak-half_mean_RR:peak+half_mean_RR]
                laser1_sum = laser1_sum + temp1 - np.mean(temp1)
                laser2_sum = laser2_sum + temp2 - np.mean(temp2)
                inc_peaks = inc_peaks + 1

        self.laser1_sum = laser1_sum
        self.laser2_sum = laser2_sum

        self.laser1_magic = scipy.signal.savgol_filter(laser1_sum / inc_peaks, 101, 3)
        self.laser2_magic = scipy.signal.savgol_filter(laser2_sum / inc_peaks, 101, 3)

        laser1_max_results = scipy.signal.argrelmax(self.laser1_magic, order=mean_RR//4)[0]
        laser1_min_results = scipy.signal.argrelmin(self.laser1_magic, order=mean_RR//4)[0]
        if (len(laser1_max_results) < 1) or (len(laser1_min_results)<1):
            self.laser1_magic_value = 0
        else:
            self.laser1_max_idx = laser1_max_results[0]
            self.laser1_min_idx = laser1_min_results[0]
            self.laser1_ptp = self.laser1_magic[self.laser1_max_idx] - self.laser1_magic[self.laser1_min_idx]
            self.laser1_magic_value = 100*(np.exp((self.laser1_ptp)/2)-1)
            print("Laser1 Upper", self.laser1_max_idx, "Laser1 Lower", self.laser1_min_idx)
            print("Magic laser1", self.laser1_magic_value)

        laser2_max_results = scipy.signal.argrelmax(self.laser2_magic, order=mean_RR//4)[0]
        laser2_min_results = scipy.signal.argrelmin(self.laser2_magic, order=mean_RR//4)[0]
        if (len(laser2_max_results) < 1) or (len(laser2_min_results)<1):
            self.laser2_magic_value = 0
        else:
            self.laser2_max_idx = laser2_max_results[0]
            self.laser2_min_idx = laser2_min_results[0]
            self.laser2_ptp = self.laser2_magic[self.laser2_max_idx] - self.laser2_magic[self.laser2_min_idx]
            self.laser2_magic_value = 100*(np.exp((self.laser2_ptp)/2)-1)
            print("Laser2 Upper", self.laser2_max_idx, "Laser2 Lower", self.laser2_min_idx)
            print("Magic laser2", self.laser2_magic_value)


    def plot_magic(self):
        plt.ioff()
        fig = plt.figure()

        ax_laser1 = fig.add_subplot(211)
        ax_laser1.plot(self.laser1_magic)
        try:
            ax_laser1.plot(self.laser1_max_idx, self.laser1_magic[self.laser1_max_idx], "ro")
            ax_laser1.plot(self.laser1_min_idx, self.laser1_magic[self.laser1_min_idx], "go")
        except AttributeError:
            pass

        ax_laser2 = fig.add_subplot(212)
        ax_laser2.plot(self.laser2_magic)
        try:
            ax_laser2.plot(self.laser2_max_idx, self.laser2_magic[self.laser2_max_idx], "ro")
            ax_laser2.plot(self.laser2_min_idx, self.laser2_magic[self.laser2_min_idx], "go")
        except AttributeError:
            pass
        plt.show()


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)