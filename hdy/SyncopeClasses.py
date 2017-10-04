import os
import zipfile
import collections

import numpy as np
import pandas as pd

import scipy
import scipy.signal
import scipy.cluster.vq
import scipy.ndimage
import scipy.ndimage.morphology
import scipy.stats
import scipy.interpolate

import sklearn
import sklearn.cluster

import mmt


from . import *


class SyncopeRun():

    def __init__(self, source: str, source_hints):

        if source.lower() == 'database':
            self.database_fl = source_hints['database_fl']
            self.daq_dir = source_hints['daq_dir']
            self.patient = source_hints['patient']
            self.exp = source_hints['exp']

            db = pd.read_csv(self.database_fl)
            row = db[(db.Patient == self.patient) & (db.Experiment == self.exp)].iloc[0]

            self.hints = row

            ecg_source = self.hints['ECG']
            pressure_source = self.hints['BP']

            self.daq_raw = daq_raw = DAQ_File(zip_dir=self.daq_dir,
                                              zip_fn=self.hints.File,
                                              channels=[ecg_source, pressure_source],
                                              search_for_files=True)


        else:
            raise Exception

        self.ecg = DAQContainerECG(data = getattr(daq_raw, ecg_source), sampling_rate=daq_raw.sampling_rate)
        self.pressure = DAQContainerBP(data = getattr(daq_raw, pressure_source), sampling_rate=daq_raw.sampling_rate)

        start = self.hints['Period_Start']
        end = self.hints['Period_End']

        if not pd.isnull(start) and not pd.isnull(end):
            self.ecg.data = self.ecg.data[start:end]
            self.pressure.data = self.pressure.data[start:end]

    def process(self):
        self.ecg.calc_ecg_peaks()
        self.ecg.calc_peaks_correlation()
        self.pressure.calc_peaks()
        self.calc_ecg_sd()

    def plot(self, fig_fl):
        plt.ioff()
        fig = plt.figure(figsize=(10,8))

        ax_ecg = fig.add_subplot(211)
        ax_ecg.plot(self.ecg.peaks_sample, self.ecg_HR, "k.")
        ax_ecg.plot(self.ecg.peaks_sample, self.ecg_HR_RM, "r-")
        ax_ecg.set_ylabel("ECG HR")

        ax_fino = fig.add_subplot(212, sharex=ax_ecg)
        ax_fino.plot(self.pressure.data, "r")
        ax_fino.set_ylabel("Fino Pressure")

        if fig_fl:
            fig_dir, _ = os.path.split(fig_fl)
            os.makedirs(fig_dir, exist_ok=True)
            fig.savefig(fig_fl, dpi=300)

        #plt.show()
        plt.close()

    def plot_raw(self, fig_fl=False):
        plt.ioff()
        fig = plt.figure(figsize=(10,8))

        ax_ecg = fig.add_subplot(211)
        ax_ecg.plot(self.ecg.data, "k")
        ax_ecg.plot(self.ecg.peaks_sample[self.ecg.peaks_similar], self.ecg.peaks_value[self.ecg.peaks_similar], "ko")
        ax_ecg.plot(self.ecg.peaks_sample[~self.ecg.peaks_similar], self.ecg.peaks_value[~self.ecg.peaks_similar], "ro")
        ax_ecg.set_ylabel("ECG")

        ax_fino = fig.add_subplot(212, sharex=ax_ecg)
        ax_fino.plot(self.pressure.data, "r")
        ax_fino.plot(self.pressure.peaks_sample, self.pressure.peaks_value, "ro")
        ax_fino.set_ylabel("Pressure")

        if fig_fl:
            fig_dir, _ = os.path.split(fig_fl)
            os.makedirs(fig_dir, exist_ok=True)
            fig.savefig(fig_fl, dpi=300)

        plt.show()
        #plt.close()

    def calc_ecg_sd(self):
        ecg_peaks_sample = self.ecg.peaks_sample.astype(np.float32)
        ecg_peaks_similar = self.ecg.peaks_similar
        ecg_peaks_sample[~ecg_peaks_similar] = np.NaN
        ecg_RR = np.diff(ecg_peaks_sample)
        ecg_RR = np.concatenate((np.array([ecg_RR[0]]), ecg_RR))
        ecg_HR = 60.0*self.ecg.sampling_rate/ecg_RR
        ecg_HR_RM = scipy.ndimage.generic_filter(ecg_HR, np.nanmedian, 31)
        #ecg_HR_RM = scipy.signal.savgol_filter(ecg_HR_RM, 101, 5)
        ecg_HR_error = np.abs(ecg_HR - ecg_HR_RM)
        ecg_HR_error_RSD = scipy.ndimage.generic_filter(ecg_HR_error, np.nanmedian, 61)
        #ecg_HR_error_RSD = scipy.signal.savgol_filter(ecg_HR_error_RSD, 101, 3)

        self.ecg_HR = ecg_HR
        self.ecg_HR_RM = ecg_HR_RM
        self.ecg_HR_error_RSD =ecg_HR_error_RSD

    def plot_ecg_adv(self, fig_fl=False):
        plt.ioff()
        fig = plt.figure(figsize=(10,8))

        ax_ecg = fig.add_subplot(211)
        ax_ecg.plot(self.ecg.peaks_sample, self.ecg_HR, "k.")
        ax_ecg.plot(self.ecg.peaks_sample, self.ecg_HR_RM, "r-")
        ax_ecg.set_ylabel("ECG HR (60/RR)")

        ax_fino = fig.add_subplot(212, sharex=ax_ecg)
        ax_fino.plot(self.ecg.peaks_sample, self.ecg_HR_error_RSD, "r.")
        ax_fino.set_ylabel("ECG HR local SD")
        ax_fino.set_ylim((0,25))

        if fig_fl:
            fig_dir, _ = os.path.split(fig_fl)
            os.makedirs(fig_dir, exist_ok=True)
            fig.savefig(fig_fl, dpi=300)
            plt.close()
        else:
            plt.show()


def plot_NN_var(pt):
    pt_diff = np.diff(pt.ecg_peaks_sample)
    pt_HR = 60 / (pt_diff / 1000.0)
    plt.plot(pt.ecg_peaks_sample[1:], pt_HR, "b.")
    plt.ylim((40, 180))

