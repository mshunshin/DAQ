import os
import zipfile

from collections import OrderedDict, namedtuple, Counter

import numpy as np
import bottleneck as bn

import scipy
import scipy.signal
import scipy.cluster.vq
import scipy.ndimage.morphology
import scipy.stats
import scipy.interpolate

import sklearn
import sklearn.cluster

import mmt


class DAQSignal:
    def __init__(self, data, sampling_rate, scale=1):
        self.sampling_rate = sampling_rate
        self.data = data * scale

    def calc_fft(self, begin=None, end=None, log=False, detrend=False):

        data = self.data[begin:end]
        sampling_rate = self.sampling_rate

        if log:
            data_min = np.min(data[data>0])
            data[data<=0] = data_min/2
            data = np.log(data)

        if detrend:
            data = scipy.signal.detrend(data)

        N = data.shape[0]
        Nq = sampling_rate / 2

        fft_freqs = np.fft.rfftfreq(N, 1.0 / sampling_rate)[0:N // 2]

        fft_power_raw = np.abs(np.fft.rfft(data))

        fft_power = (2.0 / N) * fft_power_raw

        N4 = (N//4)*4

        fft_power_rpt = (2.0 / N) * (np.repeat(fft_power_raw[0:N4 // 4], 2) + fft_power_raw[0:N4 // 2])/2

        FFT_Data = namedtuple('FFT_Data', ['power', 'power_rpt', 'freqs'])
        self.FFT =  FFT_Data(fft_power, fft_power_rpt, fft_freqs)


class DAQContainerECG(DAQSignal):
    def __init__(self, ecg_hint='sinus', **kwds):
        super().__init__(**kwds)

        self.data = mmt.butter_bandpass_filter(self.data, 0.5, 40, self.sampling_rate, order=5)
        self.ecg_hint = ecg_hint

    def calc_ecg_peaks(self, begin=None, end=None, ecg_hint='sinus'):
        print("Calculating ECG Peaks")

        req_end = end
        req_begin = begin

        if end is not None and begin is not None:
            if end - begin < 40000:
                begin = max(0, begin-20000)
                if end - begin < 40000:
                    end = min(self.data.shape[0], 40000+begin)

        data = self.data[begin:end]
        #data_dec = scipy.signal.decimate(data, 10, zero_phase=True)

        if ecg_hint.lower() == "vf":
            print(f'ECG_Hint was vf')
            relmax_peaks = scipy.signal.argrelmax(data, order=50)[0]
            peaks_sample = np.array(relmax_peaks)

        elif ecg_hint.lower() == "old-vf":
            print(f'ECG_Hint was old-vf')
            ecg_detect = bn.move_sum(np.diff(data_dec) ** 2, 4, 1)
            ecg_detect[ecg_detect < np.max(ecg_detect) // 10] = 0
            ecg_detect_peaks = np.array(
                scipy.signal.find_peaks_cwt(ecg_detect, widths=np.linspace(5, 10, 1), noise_perc=10)) * 10
            ecg_detect_peaks = ecg_detect_peaks + 75

            relmax_peaks = scipy.signal.argrelmax(np.diff(data)**2, order=50)[0]
            final_peaks = []
            for peak in ecg_detect_peaks:
                final_peaks.append(mmt.find_nearest_value(relmax_peaks, peak))
            peaks_sample = np.array(relmax_peaks)

        else:
            print(f'ECG_Hint was not vf')

            ecgdetector = mmt.ecg.ECGDetectors(1000)
            ecg_detect_peaks = ecgdetector.matt_detector(data)

            peaks_sample = ecg_detect_peaks

        print("Finished calculating ECG Peaks")

        peaks_sample = np.sort(list(set(peaks_sample)))

        if req_begin is None:
            peaks_sample = peaks_sample
        else:
            peaks_sample = peaks_sample + begin

        if req_begin is not None:
            peaks_sample = peaks_sample[peaks_sample > req_begin]

        if req_end is not None:
            peaks_sample = peaks_sample[peaks_sample < req_end]

        self.peaks_sample = peaks_sample

    @property
    def peaks_value(self):
        return self.data[self.peaks_sample]

    @property
    def peaks_num(self):
        return len(self.peaks_sample)


    def calc_peaks_correlation(self, begin=None, end=None):
        ecg_peaks_num = self.peaks_num
        peaks_sample = self.peaks_sample

        data = self.data

        pre_peak = 150
        post_peak = 250

        beat_matrix = np.zeros((ecg_peaks_num, pre_peak+post_peak))
        for i, ecg_peak in enumerate(peaks_sample):
            beat_matrix[i,:] = data.take(range(ecg_peak - pre_peak,ecg_peak + post_peak), mode='clip')

        corr_array = mmt.stats.corr2_coeff(beat_matrix, beat_matrix)

        self.peaks_correlation = corr_array

        peaks_similar = (np.sum(corr_array > 0.8, axis=1) * 1.0 / len(self.peaks_sample)) > 0.3

        self.peaks_similar = peaks_similar

    @property
    def good_peaks_sample(self):
        return self.peaks_sample[self.peaks_similar]

    def calc_peaks_cluster(self, target_clusters=2):
        ecg_peaks_num = self.peaks_num

        corr_array = self.peaks_correlation
        good_peaks = self.peaks_similar
        good_peaks_sample = self.good_peaks_sample
        good_corr_array = corr_array[good_peaks, :][:, good_peaks]

        if target_clusters > 1:
            cluster_fit = sklearn.cluster.SpectralCoclustering(n_clusters=target_clusters, svd_method='randomized')
            cluster_fit.fit(good_corr_array)
            ecg_good_peaks_code_raw = cluster_fit.row_labels_
        else:
            ecg_good_peaks_code_raw = np.ones_like(good_peaks)

        if False:
            dist_array = np.sqrt(2 * (1 - corr_array))

            # Yes - I know i send in a distance array as a feature array, but it seems to work quite nicely
            # Change, if to disntance array, set eps to be after the first group on the histogram of the distances.
            cluster_fit = sklearn.cluster.DBSCAN().fit(dist_array)

            ecg_good_peaks_code_raw = cluster_fit.labels_

        self.good_peaks_code_raw = ecg_good_peaks_code_raw

        ecg_peaks_code = ecg_good_peaks_code_raw
        ecg_peaks_code_cn = Counter(ecg_peaks_code)
        ecg_peaks_code_freq = [x for x, _ in ecg_peaks_code_cn.most_common(target_clusters)]
        ecg_peaks_code[~np.in1d(ecg_peaks_code, ecg_peaks_code_freq)] = -1
        nans = ecg_peaks_code < 0
        nz = lambda x: x.nonzero()[0]
        ecg_peaks_code_ip = scipy.interpolate.interp1d(nz(~nans), ecg_peaks_code[~nans], kind="nearest",
                                                       bounds_error=False)
        ecg_peaks_code[nz(nans)] = ecg_peaks_code_ip(nz(nans))
        ecg_peaks_code[nz(nans)] = np.interp(nz(nans), nz(~nans), ecg_peaks_code[~nans])

        ecg_peaks_code = scipy.ndimage.generic_filter(ecg_peaks_code, mmt.stats.matt_mode, 5)

        self.good_peaks_code = ecg_peaks_code

    @property
    def good_peaks_value(self):
        return self.data[self.good_peaks_sample.astype(dtype=np.int)]

    def calc_transitions(self):
        self.transitions_beat = np.argwhere(np.diff(self.good_peaks_code) != 0)[:, 0] + 1
        self.transitions_sample = self.good_peaks_sample[self.transitions_beat]

    @property
    def LombScargle(self):

        fmin = 0.0
        fmax = 0.4
        N = 1000
        df = (fmax - fmin) / N
        freqs = df + (df * np.arange(N))

        peaks_sample = self.peaks_sample.astype(np.float)
        peaks_sample[~self.peaks_similar] = np.nan

        ys = np.diff(peaks_sample)
        peaks_mask = ~np.isnan(ys)
        xs = self.peaks_sample[1:]
        xs = xs[peaks_mask] / self.sampling_rate
        ys = ys[peaks_mask]

        ang_freqs = freqs*2.0*np.pi
        power = scipy.signal.lombscargle(xs, ys - ys.mean(), ang_freqs)
        len_xs = len(xs)
        power = power * 2 / (len_xs * ys.std() ** 2)

        LS_Data = namedtuple('LS_Data', ['power', 'freqs', 'totpwr', 'ulf', 'vlf', 'lf', 'hf'])

        return LS_Data(power, freqs, np.sum(power),
                       np.sum(power[(freqs <0.003)]),
                       np.sum(power[(freqs>=0.003) & (freqs<0.04)]),
                       np.sum(power[(freqs>=0.04) & (freqs<0.15)]),
                       np.sum(power[(freqs>=0.15) & (freqs<=0.4)]))



    @property
    def SDNN(self):
        peaks_sample = self.peaks_sample.astype(np.float)
        peaks_sample[~self.peaks_similar] = np.nan
        ys = np.diff(peaks_sample)
        peaks_mask = ~np.isnan(ys)
        ys = ys[peaks_mask]

        sdnn = np.std(ys)
        return sdnn


class DAQContainerBPAO(DAQSignal):
    def __init__(self, data, sampling_rate=1000, scale=100, **kwds):
        super().__init__(data=data, sampling_rate=sampling_rate, scale=scale, **kwds)

        self.data_type = 'clean'

        savgol_width = 2*(sampling_rate//20)+1
        self.data = scipy.signal.savgol_filter(self.data, savgol_width, 3)

        data_clean = self.data.copy()

        data_dpdt = np.zeros_like(data_clean)
        data_dpdt[1:] = np.diff(data_clean)
        data_dpdt[0] = data_dpdt[1]
        data_dpdt = data_dpdt * self.sampling_rate
        data_dpdt = scipy.signal.savgol_filter(data_dpdt, savgol_width, 3)
        self.data_dpdt = data_dpdt
        self.data_clean = data_clean

    def set_dpdt(self):
        self.data = self.data_dpdt
        self.data_type = 'dpdt'

    def set_clean(self):
        self.data = self.data_clean
        self.data_type = 'clean'

    def calc_peaks(self, begin=None, end=None, dpdt=False):
        data = self.data[begin:end]
        sampling_rate = self.sampling_rate

        print("Calculating BP Peaks")
        if dpdt:
            widths = np.arange(sampling_rate // 40, sampling_rate // 6, sampling_rate // 40)
        else:
            widths = np.arange(sampling_rate // 20, sampling_rate // 3, sampling_rate // 20)
        widths = np.arange(2*(sampling_rate // 12), 6*(sampling_rate // 12), sampling_rate // 12)

        peaks_sample = mmt.find_peaks.find_peaks_cwt_refined(data, widths, decimate=True, noise_perc=30)
        print("Finished calculating BP peaks")

        if begin == None:
            self.peaks_sample = peaks_sample
        else:
            self.peaks_sample = peaks_sample + begin

    @property
    def peaks_value(self):
        return self.data[self.peaks_sample]




class DAQContainerBP(DAQContainerBPAO):
    def __init__(self, data, sampling_rate, scale=100, **kwds):

        ##Get rid of extreem finopress data.
        data[data < -10] = 0
        data[data > 500] = 500

        super().__init__(data=data, sampling_rate=sampling_rate, scale=scale, **kwds)



class DAQContainerLaser(DAQSignal):
    def __init__(self, scale=100, *args, **kwds):
        super().__init__(scale=scale, *args, **kwds)
        data = self.data
        try:
            data[data <= 0] = np.min(data[data > 0]) / 2
        except ValueError:
            data[:] = 1

        #self.data = data
        self.data = scipy.signal.savgol_filter(data, 101, 3)


class DAQContainerBoxA(DAQSignal):
    def __init__(self, scale=1,  *args, **kwds):
        super().__init__(scale=scale, *args, **kwds)

        self.codebook_default = OrderedDict([
            (0.0, '0ms'),
            (0.2, '20ms'),
            (0.4, '40ms'),
            (0.6, '80ms'),
            (0.8, '120ms'),
            (1.0, '140ms'),
            (1.2, '160ms'),
            (1.4, '180ms'),
            (1.6, '200ms'),
            (1.8, '240ms'),
            (2.0, '280ms'),
            (2.2, '320ms'),
            (2.4, '360ms'),
            (2.6, '380ms'),
            (5.0, '250ms'),
            (6.0, '300ms'),
            (7.0, '350ms'),
            (8.0, '60ms'),
            (8.1, '100ms'),
            (8.2, '220ms'),
            (8.3, '260ms'),
            (8.4, '340ms'),
            (10, 'Unknown')])

        self.codebook_volts = list(self.codebook_default.keys())
        self.codebook_default_labels = list(self.codebook_default.values())

    def fix_codebook(self, ref='Max', override_test=None):
        self.ref = ref
        self.override_test = override_test

        self.codebook = self.codebook_default.copy()

        for key, value in self.codebook.items():
            if value == ref:
                self.codebook[key] = "ref"
            elif override_test is not None:
                self.codebook[key] = override_test

        self.codebook_labels = list(self.codebook.values())

    def calc_sample_code(self):
        print("Calculating BoxA codes")
        boxa_quant, _ = scipy.cluster.vq.vq(self.data, self.codebook_volts)
        self.sample_code = scipy.ndimage.generic_filter(boxa_quant, mmt.stats.matt_mode, 1001)
        self.sample_code_set = set(self.sample_code)
        print("Finished calculating BoxA codes")

    def calc_transitions(self):
        self.transitions_sample = np.argwhere(np.diff(self.sample_code) != 0)[:, 0] + 1


class DAQContainerCranial(DAQSignal):
    def __init__(self, scale=100, *args, **kwds):
        super().__init__(scale=scale, *args, **kwds)
        self.data = scipy.signal.savgol_filter(self.data, 101, 3)
