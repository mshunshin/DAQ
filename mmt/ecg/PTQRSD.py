# Author: Raja S. <rajajs@gmail.com>
# Copyleft Dec 2008
# License: GPL

# QRS detection in ECGs using a modified Pan Tomkins method.
# Modifications based on OSEA documentation - Patrick Hamilton

#-------------------------------------
# filtfilt based on http://www.scipy.org/Cookbook/FiltFilt

# modified Pan-Tompkins algo based on Pat Hamilton's implementation
# (manual available at http://www.eplimited.com/software.htm)

# Looked at matlab code at http://www.andreamaesani.com/home/node/17
# -----------------------------------------------

from __future__ import division

import scipy
import scipy.signal
import datetime
import numpy as np


# ------- Exception -----------
class QrsDetectionError(Exception):
    """Raise error related to qrs detection"""
    def __init__(self, value):
        self.parameter = value

    def __str__(self):
        return repr(self.parameter)

# -------- ECG class -----------
class PTQRSD():
    """The ECG data """
    # ------- private functions -------
    def __init__(self, ecgdata, infodict):
        """
        - ecgdata : The ecgdata as an array - points x leads
                      or a vector
        - infodict: A dictionary object holding info
                    - 'name' = Patients name
                    - 'age' = Age in years
                    - 'sex' = 'm', 'f' or 'u' for unknown
                    - 'samplingrate' = Hz
                    - all are optional
        """
        self.infodict = infodict
        self._readinfo()
        self.data = scipy.array(ecgdata)

        # convert vector to column array
        if len(self.data.shape) == 1:
            self.data = scipy.array([self.data]).transpose()

        self.points, self.leads = self.data.shape
        if len(self.data.shape) > 1 and self.leads > self.points:
            raise QrsDetectionError("Data has more columns than rows")

        self.plot_peaks = []
        self.plot_sig_thresh = []

    def _warning(self, msg):
        """Handle warning messages"""
        # TODO: verbosity to determine how the warning will be treated.
#        print msg
        pass

    def _readinfo(self):
         """Read the info and fill missing values with defaults"""
         self.name = self.infodict.get('name', '')
         self.age = self.infodict.get('age', 0)
         self.sex = self.infodict.get('sex', 'u')
         try:
             self.samplingrate = self.infodict.get('samplingrate')
         except KeyError:
             self.samplingrate = 250
             self._warning("Info does not samplingrate, assuming 250")

    def _zeropad(self, shortvec, l):
        """Pad the vector shortvec with terminal zeros to length l"""
        return scipy.hstack((shortvec, scipy.zeros((l - len(shortvec)),
                                                   dtype='int')))

    def _sample_to_time(self, sample):
        """convert from sample number to a string representing
        time in a format required for the annotation file.
        This is in the form (hh):mm:ss.sss"""
        time_ms = int(sample*1000 / self.samplingrate)
        hr, min, sec, ms = time_ms//3600000 % 24, time_ms//60000 % 60, \
                           time_ms//1000 % 60, time_ms % 1000
        timeobj = datetime.time(hr, min, sec, ms*1000) # last val is microsecs
        timestring = timeobj.isoformat()
        # if there is no ms value, add it
        if not '.' in timestring:
            timestring += '.000000'
        return timestring[:-3] # back to ms

    def _write_ann(self, annfile):
        """Write annotation file in a format that is usable with wrann"""
        fi = open(annfile, 'w')
        for qrspeak in self.QRSpeaks:
            fi.write('%s %s %s %s %s %s\n' %(
                self._sample_to_time(qrspeak), qrspeak, 'N', 0, 0, 0))
        fi.close()

    def _initializeBuffers(self, start = 0):
        """Initialize the buffers using values from the first 8 seconds"""
        srate = self.samplingrate
        self.signal_peak_buffer = [max(self.int_ecg[start+i*srate:start+i*srate+srate])
                                                  for i in range(8)]
        self.noise_peak_buffer = [0] * 8
        self.rr_buffer = [srate] * 8 #rr buffer initialized with one sec
        self._updateThreshold()

    def _updateThreshold(self):
        """Update thresholds from buffers"""
        noise = scipy.mean(self.noise_peak_buffer)
        signal = scipy.mean(self.signal_peak_buffer)
        self.threshold = noise + 0.3125 * (signal - noise)
        self.meanrr = scipy.mean(self.rr_buffer)

    def _process_ecg(self):
        "process the raw ecg signal"
        self.filtered_ecg = self._bpfilter(self.raw_ecg)
        self.diff_ecg  = scipy.diff(self.filtered_ecg)
        self.abs_ecg = abs(self.diff_ecg)
        self.int_ecg = self._mw_integrate(self.abs_ecg)

    def _peakdetect(self, ecg):
        """Detect all local maxima with no larger maxima within 200 ms"""
        # list all local maxima
        all_peaks = [i for i in range(1,len(ecg)-1)
                     if ecg[i-1] < ecg[i] > ecg[i+1]]
        peak_amplitudes = [ecg[peak] for peak in all_peaks]
        final_peaks = []
        minRR = self.samplingrate * 0.2

        # start with first peak
        peak_candidate_index = all_peaks[0]
        peak_candidate_amplitude = peak_amplitudes[0]
        # test successively against other peaks
        for peak_index, peak_amplitude in zip(all_peaks, peak_amplitudes):
            close_to_lastpeak = peak_index - peak_candidate_index <= minRR
            # if new peak is less than minimumRR away and is larger,
            # it becomes candidate
            if close_to_lastpeak and peak_amplitude > peak_candidate_amplitude:
                peak_candidate_index = peak_index
                peak_candidate_amplitude = peak_amplitude
            # if new peak is more than 200 ms away, candidate is added to
            # final peak and new peak becomes candidate
            elif not close_to_lastpeak:
                final_peaks.append(peak_candidate_index)
                peak_candidate_index = peak_index
                peak_candidate_amplitude = peak_amplitude
            else:
                pass

        #Remove peaks within 10ms of either end.
        ms10 = self.samplingrate * 10 / 1000
        temp = np.array(final_peaks)
        temp_fp = temp[(temp>ms10)&(temp<self.points-ms10)]
        return final_peaks

    def _acceptasQRS(self, peak, amplitude):
        # if we are in relook mode, a qrs detection stops that
        if self.RELOOK:
            self.RELOOK = False
        # add to peaks, update signal buffer
        self.QRSpeaks.append(peak)
        self.signal_peak_buffer.pop(0)
        self.signal_peak_buffer.append(amplitude)
        # update rr buffer
        if len(self.QRSpeaks) > 1:
            self.rr_buffer.pop(0)
            self.rr_buffer.append(self.QRSpeaks[-1] - self.QRSpeaks[-2])
        self._updateThreshold()

        self.plot_peaks.append(peak)
        self.plot_sig_thresh.append(self.threshold)

    def _acceptasNoise(self, peak, amplitude):
        self.noise_peak_buffer.pop(0)
        self.noise_peak_buffer.append(amplitude)
        self._updateThreshold()

        self.plot_peaks.append(peak)
        self.plot_sig_thresh.append(self.threshold)

    def _mw_integrate(self, ecg):
        """ Integrate the ECG signal over a defined time period"""
        # window of 80 ms - better than using a wider window
        window_length = int(80 * (self.samplingrate / 1000))
        # fancy integration using cumsum
        int_ecg = scipy.zeros_like(ecg)
        cs = ecg.cumsum()
        int_ecg[window_length:] = (cs[window_length:] - cs[:-window_length]
                                   ) / window_length
        int_ecg[:window_length] = cs[:window_length] / scipy.arange(
                                                   1, window_length + 1)
        return int_ecg

    def _filtfilt(self, b,a,x):
        """
        Filter with given parameters forward and in reverse to eliminate
        phase shifts. Initial state is calculated with lfilter_zi and
        mirror images of the sample are added at end and beginning to
        remove edge effects.
        """
        #For now only accepting 1d arrays
        ntaps=max(len(a),len(b))
        edge=ntaps*3
        if len(a) < ntaps:
            a=scipy.r_[a,scipy.zeros(len(b)-len(a))]
        if len(b) < ntaps:
            b=scipy.r_[b,scipy.zeros(len(a)-len(b))]
        zi=self._lfilter_zi(b,a)
        #Grow the signal with inverted replicas of the signal
        s=scipy.r_[2*x[0]-x[edge:1:-1],x,2*x[-1]-x[-1:-edge:-1]]

        (y,zf)=scipy.signal.lfilter(b,a,s,-1,zi*s[0])
        (y,zf)=scipy.signal.lfilter(b,a,scipy.flipud(y),-1,zi*y[-1])
        return scipy.flipud(y[edge-1:-edge+1])

    def _lfilter_zi(self, b,a):
        """compute the zi state from the filter parameters."""
        n=max(len(a),len(b))
        zin = (scipy.eye(n-1) - scipy.hstack( (-a[1:n,scipy.newaxis],
                                     scipy.vstack((scipy.eye(n-2),
                                                   scipy.zeros(n-2))))))
        zid=  b[1:n] - a[1:n]*b[0]
        zi_matrix=scipy.linalg.inv(zin)*(scipy.matrix(zid).transpose())
        zi_return=[]
        #convert the result into a regular array (not a matrix)
        for i in range(len(zi_matrix)):
          zi_return.append(float(zi_matrix[i][0]))
        return scipy.array(zi_return)

    def _bpfilter(self, ecg):
         """Bandpass filter the ECG from 5 to 15 Hz"""
         # TODO: Explore - different filter designs
         Nyq = self.samplingrate / 2
         wn = [5/ Nyq, 15 / Nyq]
         b,a = scipy.signal.butter(2, wn, btype = 'bandpass')
         return self. _filtfilt(b,a,ecg)


    def qrsDetect(self, qrslead=0):
         """Detect QRS onsets using modified PT algorithm"""
         # If ecg is a vector, it will be used for qrs detection.
         # If it is a matrix, use qrslead (default 0)
         if len(self.data.shape) == 1:
             self.raw_ecg = self.data
         else:
             self.raw_ecg = self.data[:,qrslead]

         self._process_ecg() #creates diff_ecg and int_ecg

         # Construct buffers with last 8 values
         self._initializeBuffers()

         peaks = self._peakdetect(self.int_ecg)
         self._checkPeaks(peaks)

         # compensate for delay during integration
         self.QRSpeaks = self.QRSpeaks - 40 * (self.samplingrate / 1000)

         return self.QRSpeaks


    def qrs_detect_multiple_leads(self, leads=[]):
        """Use multiple leads for qrs detection.
        Leads to use may be given as list of lead indices.
        Default is to use all leads"""
        # leads not specified, switch to all leads
        if leads == []:
            leads = range(self.leads)

        # qrs detection for each lead
        qrspeaks = []
        for lead in leads:
            qrspeaks.append(self.qrsDetect(lead))

        # zero pad detections to match lengths
        maxlength = max([len(qrspeak_lead) for qrspeak_lead in
                         qrspeaks])
        for lead in range(len(qrspeaks)):
            qrspeaks[lead] = self._zeropad(qrspeaks[lead], maxlength)

        qrspeaks_array = scipy.array(qrspeaks).transpose()
        self.QRSpeaks = self.multilead_peak_match(qrspeaks_array)
        return self.QRSpeaks


    def _checkPeaks(self, peaks):
        """Go through the peaks one by one and classify as qrs or noise
        according to the changing thresholds"""
        srate = self.samplingrate
        ms10 = srate * 10 / 1000
        amplitudes = [self.int_ecg[peak] for peak in peaks]
        self.QRSpeaks = [-360] #initial val which we will remove later
        self.RELOOK = False # are we on a 'relook' run?

        for index in range(len(peaks)):
            peak, amplitude = peaks[index], amplitudes[index]
            # booleans
            above_thresh = amplitude > self.threshold
            distant = (peak-self.QRSpeaks[-1])*(1000/srate) > 360
            classified_as_qrs = False
            distance = peak - self.QRSpeaks[-1] # distance from last peak

            # Need to go back with fresh thresh if no qrs for 8 secs
            if distance > srate * 8:
                # If this is a relook, abandon
                if self.RELOOK:
                    self.RELOOK = False
                else:
                    # reinitialize buffers
                    self.RELOOK = True
                    index = peaks.index(self.QRSpeaks
                                        [-1])
                    self._initializeBuffers(peaks[index])

            # If distance more than 1.5 rr, lower threshold and recheck
            elif distance > 1.5 * self.meanrr:
                i = abs(index - 1)
                lastpeak = self.QRSpeaks[-1]
                last_maxder = max(self.abs_ecg[lastpeak-ms10:lastpeak+ms10])
                while peaks[i] > lastpeak:
                    this_maxder = max(self.abs_ecg[ms10:peak+ms10])
                    above_halfthresh = amplitudes[i] > self.threshold*0.5
                    distance_inrange = (peaks[i] - lastpeak)*(1000/srate) > 360
                    slope_inrange = this_maxder > last_maxder * 0.6
                    if above_halfthresh and distance_inrange and slope_inrange:
                        self._acceptasQRS(peaks[i], amplitudes[i])
                        break
                    else:
                        i -= 1

            # Rule 1: > thresh and >360 ms from last det
            if above_thresh and distant:
                classified_as_qrs = True

            # Rule 2: > thresh, <360 ms from last det
            elif above_thresh and not distant:
                this_maxder = max(self.abs_ecg[peak-ms10:peak+ms10])
                lastpeak = self.QRSpeaks[-1]
                last_maxder = max(self.abs_ecg[lastpeak-ms10:lastpeak+ms10])
                if this_maxder >= last_maxder * 0.6: #modified to 0.6
                    classified_as_qrs = True

            if classified_as_qrs:
                self._acceptasQRS(peak, amplitude)
            else:
                self._acceptasNoise(peak, amplitude)

        self.QRSpeaks.pop(0) # remove that -360
        self.QRSpeaks = scipy.array(self.QRSpeaks)
        return

    def multilead_peak_match(self, peaks):
        """Reconcile QRS detections from multiple leads.
        peaks is a matrix of peak_times x leads.
        If the number of rows is different,
        pad shorter series with zeros at end"""
        # TODO: dont use with 2 lead
        # TODO: when 2 leads only there, experiment
        # with using a manufactured third lead

        ms90 = 90 * self.samplingrate / 1000
        Npeaks, Nleads = peaks.shape
        current_peak = 0
        final_peaks = []

        while current_peak < Npeaks:
            all_values = peaks[current_peak, :]
            outer = all_values.max()
            outerlead = all_values.argmax()
            inner = all_values.min()
            innerlead = all_values.argmin()

            # classify as leads within 90 ms of min or max
            near_inner = sum(all_values < inner + ms90)
            near_outer = sum(all_values > outer - ms90)

            #all are within 90 ms
            if near_inner == near_outer == Nleads:
                final_peaks.append(int(scipy.median(all_values)))
                current_peak += 1

            # max is wrong
            elif near_inner > near_outer:
                print("max is wrong")
                peaks[current_peak+1:Npeaks, outerlead] = peaks[current_peak:Npeaks-1, outerlead]
                peaks[current_peak, outerlead] = scipy.median(all_values)
                # do not change current peak now

            # min is wrong
            elif near_inner <= near_outer:
                if current_peak < Npeaks-1:
                    peaks[current_peak:Npeaks-1, innerlead] = peaks[current_peak+1:Npeaks, innerlead]
                    peaks[-1, innerlead] = 0
                else: # if this is the last peak
                    peaks[current_peak, innerlead] = scipy.median(all_values)

        return final_peaks

