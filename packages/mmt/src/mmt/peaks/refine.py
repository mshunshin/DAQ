__author__ = 'matthew'

import numpy as np
import scipy.signal

import mmt


def refine_fitter(peaks_x, trace_smooth, trace_raw):

    peaks_fit_x = np.zeros_like(peaks_x)
    trace_fit = np.copy(trace_smooth)

    for i, peak_x in enumerate(peaks_x):

        y_clip = trace_smooth[peak_x]//2

        lhs_clip_dist = np.min([np.argmax(trace_smooth[:peak_x][::-1]<y_clip), 30])
        rhs_clip_dist = np.min([np.argmax(trace_smooth[peak_x:]<y_clip), 30])

        fit_range_x = np.arange(peak_x-lhs_clip_dist, peak_x+rhs_clip_dist)

        fit = np.polyfit(fit_range_x, trace_raw[fit_range_x],4)
        fitter = np.poly1d(fit)
        fitter_y = fitter(fit_range_x)

        trace_fit[fit_range_x] = fitter_y
        peaks_fit_x[i] = fit_range_x[np.where(fitter_y == np.max(fitter_y))]

    return peaks_fit_x, trace_fit

def refine_max(data, peaks_est):

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