import numpy as np
import scipy
import scipy.signal

import mmt

def find_peaks_cwt_refined(vector, widths, decimate=False, decimate_factor=10):

    if decimate:
        decimate_factor = int(decimate_factor)
        vector_decimate = scipy.signal.decimate(vector, decimate_factor, zero_phase=True)
        widths_decimate = np.array(list(set(np.array(widths)//decimate_factor)))
        peaks_cwt = np.array(scipy.signal.find_peaks_cwt(vector_decimate, widths_decimate)) * decimate_factor
    else:
        peaks_cwt = np.array(scipy.signal.find_peaks_cwt(vector, widths))

    peaks_cwt_interval = np.diff(peaks_cwt)
    peaks_search_dist = max(int(np.percentile(peaks_cwt_interval, 20)//2),3)
    peaks_local = scipy.signal.argrelmax(vector, order=peaks_search_dist)[0]

    refined_peaks = np.zeros_like(peaks_cwt)

    for i, peak in enumerate(peaks_cwt):
        refined_peaks[i] = mmt.find_nearest_value(peaks_local,peak)

    return np.sort(list(set(refined_peaks)))
