import os

import numpy as np

import scipy
import scipy.signal
import scipy.ndimage

import mmt
import hdy


class VolcanoAnalysis:
    def __init__(self, sdy_fl):
        self.sdy_fl = sdy_fl

        sdy_exp = hdy.SDY_File(sdy_fl)
        self.sdy_exp = sdy_exp

        self.sampling_rate = self.sdy_exp.sampling_rate

        self.pd = hdy.DAQContainerBP(data=hdy.remove_outliers(sdy_exp.pd_raw),
                                     sampling_rate=self.sampling_rate,
                                     scale=0.25)

        self.pa = hdy.DAQContainerBP(data=hdy.remove_outliers(sdy_exp.pa_raw),
                                     sampling_rate=self.sampling_rate,
                                     scale=0.25)

        self.ecg = hdy.DAQContainerECG(data=hdy.remove_outliers(sdy_exp.ecg_raw),
                                       sampling_rate=self.sampling_rate)

        self.flow = hdy.DAQContainerCranial(data=hdy.remove_outliers(sdy_exp.flow_raw),
                                            sampling_rate=self.sampling_rate)

        self.roi_start = 0
        self.roi_end = self.ecg.data.shape[0]
        self.plot_order = True

    def process(self, start, end):
        self.roi_start = start
        self.roi_end = end
        self.pa.find_peaks(self.roi_start, self.roi_end)
        self.pa.find_valleys(self.roi_start, self.roi_end)
        self.pd.find_peaks(self.roi_start, self.roi_end)
        self.pa.find_valleys(self.roi_start, self.roi_end)
        self.calc_ffr()

    def calc_ffr(self):
        start = self.roi_start
        end = self.roi_end
        sampling_rate = self.sampling_rate
        window = sampling_rate * 10  ##I.e. a 10 second smoother. Check what we should use here

        pa = self.pa.data[start:end]
        pd = self.pd.data[start:end]

        pa_mean = scipy.ndimage.generic_filter(pa, np.mean, size=window)  # This is slow - find a built in
        pd_mean = scipy.ndimage.generic_filter(pd, np.mean, size=window)  # This is slow - find a built in
        ffr_live = pd_mean / pa_mean
        ffr = np.min(ffr_live)  # Perhaps use lower 95% CI

        self.pa_mean = pa_mean
        self.pd_mean = pd_mean
        self.ffr_live = ffr_live
        self.ffr = ffr
