import numpy as np
import scipy.signal as signal

from scipy.signal import find_peaks_cwt
from math import *

def num_sim(n1, n2):
    sum = n1 * n1
    sum2 = n2 * n2
    return sum / sqrt(sum * sum2)


def inf_tab(lst, min, max):
    count = 0
    for i in range(len(lst)):
        if lst[i] < max and lst[i] > min:
            count += 1
    return count


def interval0(val1, val2, interval):
    max = val1 + interval / 2
    min = val1 - interval / 2
    if (val2 < min or val2 > max):
        return True
    else:
        return True


def interval(val1, val2, interval1, interval2):
    if (interval1 < interval2):
        inter = interval1
    else:
        inter = interval2
    max = val1 + inter / 1  # 1.35
    min = val1 - inter / 1  # 1.35
    if (val2 < min or val2 > max):
        return False
    else:
        return True


class Detectors_had:
    """ECG heartbeat detection algorithms
    General useage instructions:
    r_peaks = detectors.the_detector(ecg_in_samples)
    The argument ecg_in_samples is a single channel ECG in volt
    at the given sample rate.
    """

    def __init__(self, sampling_frequency):
        """
        The constructor takes the sampling rate in Hz of the ECG data.
        """

        ## Sampling rate
        self.fs = sampling_frequency

        ## This is set to a positive value for benchmarking
        self.engzee_fake_delay = 0

        ## 2D Array of the different detectors: [[description,detector]]
        self.detector_list = [
            ["Pan Tompkins", self.pan_tompkins_detector]
        ]

    def pan_tompkins_detector(self, unfiltered_ecg):
        """
        Jiapu Pan and Willis J. Tompkins.
        A Real-Time QRS Detection Algorithm.
        In: IEEE Transactions on Biomedical Engineering
        BME-32.3 (1985), pp. 230–236.
        """
        global squared

        maxQRSduration = 0.150  # sec
        f1 = 5 / self.fs
        f2 = 15 / self.fs

        b, a = signal.butter(1, [f1 * 2, f2 * 2], btype='bandpass')

        filtered_ecg = signal.lfilter(b, a, unfiltered_ecg)

        diff = np.diff(filtered_ecg)

        squared = diff * diff

        N = int(maxQRSduration * self.fs)
        mwa = MWA_cumulative(squared, N)
        mwa[:int(maxQRSduration * self.fs * 2)] = 0

        r_peaks = panPeakDetect(mwa, self.fs)

        return r_peaks


    def Hoai_Anne_ecg_detector(self, unfiltered_ecg):
        ecg_filt = self.pan_tompkins_detector(unfiltered_ecg)

        list_squared = list(squared)

        dist = []
        for i in range(1, len(ecg_filt)):
            dist.append(abs(ecg_filt[i] - ecg_filt[i - 1]))

        dist.sort(reverse=True)
        new_dist = []
        for i in range(floor(len(dist) / 1)):  # 3
            new_dist.append(dist[i])

        smallest_dist = np.mean(new_dist)
        print("The smallest distance is: ", smallest_dist)

        list_squared = list_squared[::8]
        r_peaks = find_peaks_cwt(list_squared, widths=np.arange(5,smallest_dist / 8))

        return r_peaks



# Fast implementation of moving window average with numpy's cumsum function
def MWA_cumulative(input_array, window_size):
    ret = np.cumsum(input_array, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]

    for i in range(1, window_size):
        ret[i - 1] = ret[i - 1] / i
    ret[window_size - 1:] = ret[window_size - 1:] / window_size

    return ret



def panPeakDetect(detection, fs):
    min_distance = int(0.25 * fs)

    signal_peaks = [0]
    noise_peaks = []

    SPKI = 0.0
    NPKI = 0.0

    threshold_I1 = 0.0
    threshold_I2 = 0.0

    RR_missed = 0
    index = 0
    indexes = []

    missed_peaks = []
    peaks = []

    for i in range(len(detection)):

        if i > 0 and i < len(detection) - 1:
            if detection[i - 1] < detection[i] and detection[i + 1] < detection[i]:
                peak = i
                peaks.append(i)

                if detection[peak] > threshold_I1 and (peak - signal_peaks[-1]) > 0.3 * fs:

                    signal_peaks.append(peak)
                    indexes.append(index)
                    SPKI = 0.125 * detection[signal_peaks[-1]] + 0.875 * SPKI
                    if RR_missed != 0:
                        if signal_peaks[-1] - signal_peaks[-2] > RR_missed:
                            missed_section_peaks = peaks[indexes[-2] + 1:indexes[-1]]
                            missed_section_peaks2 = []
                            for missed_peak in missed_section_peaks:
                                if missed_peak - signal_peaks[-2] > min_distance and signal_peaks[
                                    -1] - missed_peak > min_distance and detection[missed_peak] > threshold_I2:
                                    missed_section_peaks2.append(missed_peak)

                            if len(missed_section_peaks2) > 0:
                                missed_peak = missed_section_peaks2[np.argmax(detection[missed_section_peaks2])]
                                missed_peaks.append(missed_peak)
                                signal_peaks.append(signal_peaks[-1])
                                signal_peaks[-2] = missed_peak

                else:
                    noise_peaks.append(peak)
                    NPKI = 0.125 * detection[noise_peaks[-1]] + 0.875 * NPKI

                threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)
                threshold_I2 = 0.5 * threshold_I1

                if len(signal_peaks) > 8:
                    RR = np.diff(signal_peaks[-9:])
                    RR_ave = int(np.mean(RR))
                    RR_missed = int(1.66 * RR_ave)

                index = index + 1

    signal_peaks.pop(0)

    return signal_peaks

