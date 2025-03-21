__author__ = 'matthew'

import numpy as np
import scipy.signal

import mmt

def refine_peaks(data, peaks_est):

    peaks_est_interval = np.diff(peaks_est)
    peaks_est_mean = np.mean(peaks_est_interval)
    peaks_est_sd = np.std(peaks_est_interval)
    peaks_search_dist = int((peaks_est_mean - 3*peaks_est_sd)/2)
    if peaks_search_dist < 3:
        peaks_search_dist = int(np.max(np.min(peaks_est_interval/2), 3))

    peaks_local = scipy.signal.argrelmax(data, order=peaks_search_dist)[0]

    refined_peaks = np.zeros_like(peaks_est)

    for i, peak in enumerate(peaks_est):
        refined_peaks[i] = mmt.find_nearest_value(peaks_local,peak)

    return refined_peaks

