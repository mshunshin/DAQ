import os
import sys
import collections

import numpy as np
import pandas as pd

import scipy
import scipy.signal
import scipy.fftpack
import scipy.interpolate

import mmt

from . import *

PR_DELAY = 120

class LaserAnalysis(object):

    def __init__(self, database_dir, database_fn, patient, exp, mode="normal"):

        self.database_dir = database_dir
        self.database_fn = database_fn
        self.patient = patient
        self.exp = exp
        self.mode = mode

        self.database_fl = os.path.join(database_dir, database_fn)

        db = pd.read_csv(self.database_fl)
        row = db[(db.Patient == patient) & (db.Experiment == exp)].iloc[0]

        self.hints = row

        self.zip_fl = os.path.join(database_dir, patient, "Haem", row.File)
        self.daq_raw = daq_raw = DAQ_File(*os.path.split(self.zip_fl))

        self.results = collections.OrderedDict()
        self.freq_hint = float(getattr(self.hints, 'Freq_Hint', 1))
        self.ecg_hint = getattr(self.hints, 'ECG_Hint', "Sinus")
        self.period = getattr(self.hints, 'Period', "Undefined")

        print("Loading: {0}".format(self.zip_fl))
        print("Experiment: {0}".format(self.exp))
        print("Patient: {0}".format(self.patient))


        try:
            self.begin = self.hints['Begin']
        except:
            self.begin = None

        try:
            self.end = self.hints['End']
        except:
            self.end = None

        ecg_source = str(getattr(self.hints, 'ECG', 'blank'))
        pressure_source = str(getattr(self.hints, 'BP', 'blank'))
        laser1_source = str(getattr(self.hints, 'Laser1', 'blank'))
        laser2_source = str(getattr(self.hints, 'Laser2', 'blank'))
        cranial1_source = str(getattr(self.hints, 'TCD1', 'blank'))
        cranial2_source = str(getattr(self.hints, 'TCD2', 'blank'))
        resp_source = str(getattr(self.hints, 'Resp', 'blank'))
        boxa_source = str(getattr(self.hints, 'BoxA', 'blank'))

        self.ecg = DAQContainerECG(data = getattr(daq_raw, ecg_source, daq_raw.blank), sampling_rate=daq_raw.sampling_rate)
        self.pressure = DAQContainerBP(data = getattr(daq_raw, pressure_source, daq_raw.blank), sampling_rate=daq_raw.sampling_rate)
        self.laser1 = DAQContainerLaser(data = getattr(daq_raw, laser1_source, daq_raw.blank), sampling_rate=daq_raw.sampling_rate)
        self.laser2 = DAQContainerLaser(data = getattr(daq_raw, laser2_source, daq_raw.blank), sampling_rate=daq_raw.sampling_rate)
        self.cranial1 = DAQContainerLaser(data = getattr(daq_raw, cranial1_source, daq_raw.blank), sampling_rate=daq_raw.sampling_rate)
        self.cranial2 = DAQContainerLaser(data = getattr(daq_raw, cranial2_source, daq_raw.blank), sampling_rate=daq_raw.sampling_rate)
        self.resp = DAQSignal(data = getattr(daq_raw, resp_source, daq_raw.blank), sampling_rate=daq_raw.sampling_rate)
        self.boxa = DAQContainerBoxA(data = getattr(daq_raw, boxa_source, daq_raw.blank), sampling_rate=daq_raw.sampling_rate)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def process(self):

        begin = int(self.begin)
        end = int(self.end)

        self.init_results()
        self.ecg.calc_ecg_peaks(begin=begin, end=end, ecg_hint=self.hints['Period'])

        try:
            self.pressure.calc_peaks(begin=begin, end=end)
        except Exception as e:
            print(e)

        self.calc_fft_results(begin=begin, end=end)
        self.calc_magic_results(begin=begin, end=end, laser="laser1")
        self.calc_magic_results(begin=begin, end=end, laser="laser2")

        envelope1_data = mmt.butter_bandpass_filter(self.laser1.data[begin:end], 0.5, 5, 1000, order=2)
        envelope1_data = np.abs(scipy.signal.hilbert(envelope1_data))
        envelope1_data = mmt.butter_lowpass_filter(envelope1_data, 1, 1000, 2)

        envelope2_data = mmt.butter_bandpass_filter(self.laser2.data[begin:end], 0.5, 5, 1000, order=2)
        envelope2_data = np.abs(scipy.signal.hilbert(envelope2_data))
        envelope2_data = mmt.butter_lowpass_filter(envelope2_data, 1, 1000, 2)

        self.results['Laser1_Mean'] = np.mean(self.laser1.data[begin:end])
        self.results['Laser2_Mean'] = np.mean(self.laser2.data[begin:end])
        self.results['MAP_Mean'] = np.mean(self.pressure.data[begin:end])
        self.results['SBP_Mean'] = np.mean(self.pressure.peaks_value)
        self.results['Laser1_SJM'] = np.mean(envelope1_data)
        self.results['Laser2_SJM'] = np.mean(envelope2_data)
        self.results['Median_RR'] = int(np.median(np.diff(self.ecg.peaks_sample)))

    def init_results(self):
        results = self.results
        results['Patient'] = self.patient
        results['Experiment'] = self.exp
        results['File'] = self.hints["File"]
        results['Period'] = self.period


    def calc_fft_results(self, begin=None, end=None):
        results = self.results

        self.ecg.calc_fft(begin=begin, end=end, log=False, detrend=True)
        self.pressure.calc_fft(begin=begin, end=end, log=True, detrend=True)
        self.laser1.calc_fft(begin=begin, end=end, log=True, detrend=True)
        self.laser2.calc_fft(begin=begin, end=end, log=True, detrend=True)

        ecg_fft = self.ecg.FFT
        pressure_fft = self.pressure.FFT
        laser1_fft = self.laser1.FFT
        laser2_fft = self.laser2.FFT

        ecg_fft_freqs = ecg_fft.freqs
        ecg_fft_power = ecg_fft.power_rpt

        ecg_fft_peaks = scipy.signal.find_peaks_cwt(ecg_fft_power, widths = np.array([1,2,3,4,5,6,7,8,9,10]), noise_perc=10)
        ecg_fft_peaks = scipy.signal.argrelmax(ecg_fft_power, order=5)[0]

        ecg_fft_peak_freq = mmt.find_nearest_value(ecg_fft_freqs[ecg_fft_peaks], self.freq_hint)
        ecg_fft_peak_freq_idx = mmt.find_nearest_idx(ecg_fft_freqs, ecg_fft_peak_freq) #Change to np.where

        self.ecg_fft_peak_freq = ecg_fft_peak_freq
        self.ecg_fft_peak_freq_idx = ecg_fft_peak_freq_idx

        results['ECG_Peak_Power'] = ecg_fft.power_rpt[ecg_fft_peak_freq_idx]
        results['Fino_Peak_Power'] = pressure_fft.power_rpt[ecg_fft_peak_freq_idx]
        results['Laser1_Peak_Power'] = laser1_fft.power_rpt[ecg_fft_peak_freq_idx]
        results['Laser2_Peak_Power'] = laser2_fft.power_rpt[ecg_fft_peak_freq_idx]

        results['ECG_Power'] = np.sum(ecg_fft.power_rpt[ecg_fft_peak_freq_idx-1:ecg_fft_peak_freq_idx+2])
        results['Fino_Power'] = np.sum(pressure_fft.power_rpt[ecg_fft_peak_freq_idx-1:ecg_fft_peak_freq_idx+2])
        results['Laser1_Power'] = np.sum(laser1_fft.power_rpt[ecg_fft_peak_freq_idx-1:ecg_fft_peak_freq_idx+2])
        results['Laser2_Power'] = np.sum(laser2_fft.power_rpt[ecg_fft_peak_freq_idx-1:ecg_fft_peak_freq_idx+2])

    def calc_magic_results(self, begin=None, end=None, laser="laser1"):

        if begin == None:
            ecg_peaks_sample = self.ecg.peaks_sample
        else:
            ecg_peaks_sample = self.ecg.peaks_sample - begin


        if 'atrial' in self.mode:
            ecg_peaks_sample = ecg_peaks_sample - PR_DELAY
            ecg_peaks_sample = ecg_peaks_sample[ecg_peaks_sample > 0]
            print("Atrial fix ecg" + str(ecg_peaks_sample))

        if self.mode == 'double_count' or self.mode == 'double_count_fix':
            ecg_peaks_sample_2 = ecg_peaks_sample - np.median(np.diff(ecg_peaks_sample))/3
            out = list(ecg_peaks_sample)
            out.extend(list(ecg_peaks_sample_2))
            ecg_peaks_sample = sorted(out)
            ecg_peaks_sample = np.array(ecg_peaks_sample)
            ecg_peaks_sample = ecg_peaks_sample[ecg_peaks_sample>0]
            ecg_peaks_sample = ecg_peaks_sample.astype(np.int)
            print(ecg_peaks_sample)

        if self.mode == 'broken_lead':
            ecg_peaks_sample = np.array(sorted(set(np.random.uniform(0,(end-begin),(end-begin)//250).astype(np.int))))

        if 'double_count_fix' in self.mode:
            LOOP_NUMBER = 2 #Loop through all, then take the best magic result. But to save time pick 2.
            test_intervals = [300, 400, 800, 1000, 1200, 1500]
            test_latitude = 150
            test_interval = test_intervals[LOOP_NUMBER]

            out = []
            ecg_peaks_sample = list(ecg_peaks_sample)
            ecg_peaks_sample.pop(0)
            out.append(ecg_peaks_sample[0])
            last_missed = False
            for sample in ecg_peaks_sample:
                if last_missed:
                    out.append(sample)
                    last_missed = False
                    continue
                else:
                    if sample - out[-1] < test_interval:
                        last_missed = True
                        continue
                    else:
                        out.append(sample)

            ecg_peaks_sample = np.array(out)


        laser_data = self[laser].data[begin:end]+10
        laser_data = np.log(laser_data) + laser_data/100
        laser_data = mmt.butter_bandpass_filter(laser_data, 0.5, 25.0, 1000, order=2)

        print("ecg_peaks ", ecg_peaks_sample)


        mean_RR = int(4*(np.mean(np.diff(ecg_peaks_sample))//4))
        median_RR = int(np.median(np.diff(ecg_peaks_sample)))
        print("Mean {0} RR".format(mean_RR))
        print("Median {0} RR".format(median_RR))


        laser_peaks = mmt.find_peaks.find_peaks_cwt_refined(-laser_data, np.array([20,50,100,150,200,300,500]), decimate=True, decimate_factor=10)

        outer_delay = np.subtract.outer(ecg_peaks_sample, laser_peaks)
        outer_delay[outer_delay>=0] = -100000

        outer_delay_max = np.max(outer_delay, axis=1)

        shift = np.int(-np.median(outer_delay_max))

        if shift <0:
            shift = 0

        if shift > 1000:
            shift = 0

        print("shift: ", shift)

        pre_shift = shift
        post_shift = shift

        laser_sum = np.zeros(1000)
        inc_peaks = 0
        laser_list = []

        peaks_num = ecg_peaks_sample.shape[0]

        for i in np.arange(peaks_num-1):
            print(i)
            beat_begin = ecg_peaks_sample[i] + pre_shift
            beat_end = ecg_peaks_sample[i+1] + post_shift

            if beat_end > len(laser_data):
                continue

            if self.hints['Period'].lower() != "vf":
                if abs((beat_end-beat_begin)-median_RR) > median_RR*0.3:
                    print("ectopic ignored")
                    continue


            laser_temp = laser_data[beat_begin:beat_end]

            xs = np.linspace(0, 1000, num=laser_temp.shape[0])
            laser_f = scipy.interpolate.interp1d(xs, laser_temp)
            laser_temp_thousand = laser_f(np.linspace(0, 1000, num=1000))
            laser_temp_thousand = scipy.signal.detrend(laser_temp_thousand, type='constant')
            laser_sum = laser_sum + laser_temp_thousand
            laser_list.append(laser_temp_thousand)
            inc_peaks = inc_peaks+1


        temp = np.array(laser_list)
        print(temp.shape)

        laser_magic = laser_sum/ inc_peaks
        laser_magic = scipy.signal.savgol_filter(laser_magic, 101, 3)

        error = np.mean((temp - laser_magic[None,:])**2)


        laser_max_results = scipy.signal.argrelmax(laser_magic, order=mean_RR//2)[0]
        #laser_min_results = scipy.signal.argrelmin(laser_magic, order=mean_RR//2)[0]
        laser_min_results = np.array([0])

        if (len(laser_max_results) < 1) or (len(laser_min_results)<1):
            laser_ptp = 0
        else:
            laser_max_idx = laser_max_results[0]
            laser_min_idx = laser_min_results[0]
            laser_ptp = laser_magic[laser_max_idx] - laser_magic[laser_min_idx]
            self[laser + '_max_idx'] = laser_max_idx
            self[laser + '_min_idx'] = laser_min_idx

        laser_magic_value = 100*(np.exp((laser_ptp)/2)-1)

        conf = error/(laser_ptp+0.0001)**2
        conf_pct = 100*(1-conf)
        if conf_pct < 0:
            conf_pct = 0

        print(error, laser_ptp, conf_pct)

        self[laser + '_magic_data_all'] = temp
        self[laser + '_magic_data'] = laser_magic
        self[laser + '_magic_value'] = self.results[laser.title() + '_Magic'] = laser_magic_value
        self[laser + '_conf_value'] = self.results[laser.title() + '_Conf'] = conf_pct

