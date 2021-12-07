import os
import sys
import collections

import numpy as np
import pandas as pd

import scipy
import scipy.signal
import scipy.fftpack
import scipy.interpolate


import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, RANSACRegressor, TheilSenRegressor, HuberRegressor
from sklearn.metrics import r2_score


import mmt

from . import *

PR_DELAY = 120

class BSplineFeatures(sklearn.base.TransformerMixin):
    def __init__(self, knots, degree=3, periodic=False):
        self.bsplines = self.get_bspline_basis(knots, degree, periodic=periodic)
        self.nsplines = len(self.bsplines)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        nsamples, nfeatures = X.shape
        features = np.zeros((nsamples, nfeatures * self.nsplines))
        for ispline, spline in enumerate(self.bsplines):
            istart = ispline * nfeatures
            iend = (ispline + 1) * nfeatures
            features[:, istart:iend] = scipy.interpolate.splev(X, spline)
        return features

    def get_bspline_basis(self, knots, degree=3, periodic=False):
        """Get spline coefficients for each basis spline."""
        nknots = len(knots)
        y_dummy = np.zeros(nknots)

        knots, coeffs, degree = scipy.interpolate.splrep(knots, y_dummy, k=degree,
                                          per=periodic)
        ncoeffs = len(coeffs)
        bsplines = []
        for ispline in range(nknots):
            coeffs = [1.0 if ispl == ispline else 0.0 for ispl in range(ncoeffs)]
            bsplines.append((knots, coeffs, degree))
        return bsplines

class LaserAnalysis(object):

    def __init__(self, database_dir=None, database_fn=None, patient=None, exp=None, zip_fl=None, mode="normal"):

        self.results = collections.OrderedDict()
        self.results_beatbybeat = collections.OrderedDict()

        if zip_fl is None:
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

            self.freq_hint = float(getattr(self.hints, 'Freq_Hint', 1))
            self.ecg_hint = getattr(self.hints, 'ECG_Hint', "Sinus")
            self.period = getattr(self.hints, 'Period', "Undefined")
            self.notes = getattr(self.hints, "Notes", "NA")

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

        ecg_hint = self.hints['Period']
        print(ecg_hint)
        self.ecg.calc_ecg_peaks(begin=begin, end=end, ecg_hint=ecg_hint)

        try:
            self.pressure.calc_peaks(begin=begin, end=end)
        except Exception as e:
            print(e)

        if True:
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

        for i, sbp in enumerate(self.pressure.peaks_value):
            self.results_beatbybeat['Patient'].append(self.patient)
            self.results_beatbybeat['Experiment'].append(self.exp)
            self.results_beatbybeat['File'].append(self.hints["File"])
            self.results_beatbybeat['Period'].append(self.period)
            self.results_beatbybeat['Notes'].append(self.notes)
            self.results_beatbybeat["Beat"].append(i)
            self.results_beatbybeat["SBP"].append(sbp)

        pass

    def init_results(self):
        self.results['Patient'] = self.patient
        self.results['Experiment'] = self.exp
        self.results['File'] = self.hints["File"]
        self.results['Period'] = self.period
        self.results['Notes'] = self.notes

        self.results_beatbybeat['Patient'] = []
        self.results_beatbybeat['Experiment'] = []
        self.results_beatbybeat['File'] = []
        self.results_beatbybeat['Period'] = []
        self.results_beatbybeat['Notes'] = []
        self.results_beatbybeat['Beat'] = []
        self.results_beatbybeat["SBP"] = []


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

        ecg_peaks_sample_raw = ecg_peaks_sample.copy()

        laser_data = self[laser].data[begin:end+2000]


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

            test_intervals = np.array([300, 400, 800, 1000, 1200, 1500, 300, 400, 800, 1000, 1200, 1500])
            test_ignore_first_list = np.array([False,False,False,False,False,False, True, True, True, True, True, True])
            test_ecg_peaks_sample = []
            results_magic = np.zeros_like(test_intervals)

            for i, (test_interval, test_ignore_first) in enumerate(zip(test_intervals, test_ignore_first_list)):
                pass

                out = []
                ecg_peaks_sample = list(ecg_peaks_sample_raw)
                if test_ignore_first:
                    ecg_peaks_sample.pop(0)
                first = ecg_peaks_sample.pop(0)
                out.append(first)
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

                test_ecg = np.array(out)
                test_ecg_peaks_sample.append(test_ecg)
                result = LaserAnalysis._calc_laser_magic(test_ecg, laser_data)
                if result.conf_value < 25:
                    results_magic[i] = -100
                else:
                    results_magic[i] = result.magic_value


            max_idx = np.nanargmax(results_magic)
            ecg_peaks_sample = test_ecg_peaks_sample[max_idx]

        result = LaserAnalysis._calc_laser_magic(ecg_peaks_sample, laser_data)

        self[laser + '_min_idx'] = result.min_idx
        self[laser + '_min_idx'] = result.max_idx
        self[laser + '_magic_data_all'] = result.magic_data_all
        self[laser + '_magic_data'] = result.magic_data
        self[laser + '_magic_value'] = self.results[laser.title() + '_Magic'] = result.magic_value
        self[laser + '_conf_value'] = self.results[laser.title() + '_Conf'] = result.conf_value

    @staticmethod
    def _calc_laser_magic(ecg_peaks_sample, laser_data):

        laser_data = laser_data + 10
        laser_data = np.log(laser_data) + laser_data/200
        laser_data = mmt.butter_bandpass_filter(laser_data, 0.5, 25.0, fs=1000, order=2)
        laser_data = scipy.signal.savgol_filter(laser_data,11,3)

        print(f"ecg_peaks: {ecg_peaks_sample}")


        mean_RR = int(4*(np.mean(np.diff(ecg_peaks_sample))//4))
        median_RR = int(np.median(np.diff(ecg_peaks_sample)))
        print(f"Mean {mean_RR} RR")
        print(f"Median {median_RR} RR")

        if True:
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

        else:
            pre_shift = 0
            post_shift = 0

        laser_sum = np.zeros(1000)
        inc_peaks = 0
        laser_list = []

        peaks_num = ecg_peaks_sample.shape[0]

        for i in np.arange(peaks_num-1):
            print(i)
            beat_begin = ecg_peaks_sample[i] + pre_shift
            beat_end = ecg_peaks_sample[i+1] + post_shift

            if beat_end > len(laser_data):
                print("Not enough laser data after shifting")
                continue

            if beat_end - beat_begin > median_RR * 2:
                print("Hello")
                continue

            if beat_end - beat_begin < median_RR * 0.5:
                print("Hello")
                continue

            laser_temp = laser_data[beat_begin:beat_end]

            xs = np.linspace(0, 1000, num=laser_temp.shape[0])
            laser_f = scipy.interpolate.interp1d(xs, laser_temp)
            laser_temp_thousand = laser_f(np.linspace(0, 1000, num=1000))
            laser_temp_thousand = scipy.signal.detrend(laser_temp_thousand, type='constant')
            laser_sum = laser_sum + laser_temp_thousand
            laser_list.append(laser_temp_thousand)
            inc_peaks = inc_peaks+1


        laser_ar = np.array(laser_list)

        knots = np.linspace(0, 1000, 11)
        bspline_features = BSplineFeatures(knots, degree=3, periodic=False)

        x_fit = np.arange(1000).repeat(laser_ar.shape[0]).ravel()
        y_fit = laser_ar.T.ravel()

        model = make_pipeline(bspline_features, HuberRegressor())
        model.fit(x_fit[:,None], y_fit)

        x_predict = np.arange(0,1000)
        y_predict = model.predict(x_predict[:,None])

        laser_magic = y_predict

        y_predict_all = model.predict(x_fit[:, None])
        conf_pct = r2_score(y_fit, y_predict_all) * 100

        if laser_ar.shape[0] <3:
            conf_pct = -1

        laser_max_idx = np.argmax(laser_magic)
        laser_min_idx = np.argmin(laser_magic)

        laser_ptp = laser_magic[laser_max_idx] - laser_magic[laser_min_idx]

        laser_magic_value = 100 * (np.exp((laser_ptp) / 2) - 1)

        from collections import namedtuple

        MagicResults = namedtuple('MagicResults', ['max_idx', 'min_idx', 'magic_data_all', 'magic_data', 'magic_value', 'conf_value'])

        print(f"Laser Value{laser_ptp}, Laser Conf{conf_pct}")

        out = MagicResults(max_idx = laser_min_idx,
                           min_idx = laser_min_idx,
                           magic_data_all = laser_ar,
                           magic_data = laser_magic,
                           magic_value = laser_magic_value,
                           conf_value = conf_pct)

        return out



