import numpy as np
#import pywt
import pathlib
import scipy.signal as signal


class Detectors:
    
    
    def __init__(self, sampling_frequency):

        self.fs = sampling_frequency


    def hamilton_detector(self, unfiltered_ecg):
        
        f1 = 8/self.fs
        f2 = 16/self.fs

        b, a = signal.butter(1, [f1*2, f2*2], btype='bandpass')

        filtered_ecg = signal.lfilter(b, a, unfiltered_ecg)

        diff = abs(np.diff(filtered_ecg))

        b = np.ones(int(0.08*self.fs))
        b = b/int(0.08*self.fs)
        a = [1]

        ma = signal.lfilter(b, a, diff)

        ma[0:len(b)*2] = 0

        peaks, _ = signal.find_peaks(ma, distance=(0.25*self.fs))

        n_pks = []
        n_pks_ave = 0.0
        s_pks = []
        s_pks_ave = 0.0
        QRS = []
        RR = []
        RR_ave = 0.0

        th = 0.0

        i=0
        idx = []
        for peak in peaks:

            if ma[peak] > th:        
                QRS.append(peak)
                idx.append(i)
                s_pks.append(ma[peak])
                if len(n_pks)>8:
                    s_pks.pop(0)
                s_pks_ave = np.mean(s_pks)

                if RR_ave != 0.0:
                    if QRS[-1]-QRS[-2] > 1.5*RR_ave:
                        missed_peaks = peaks[idx[-2]+1:idx[-1]]
                        for missed_peak in missed_peaks:
                            if missed_peak-peaks[idx[-2]]>int(0.360*self.fs) and ma[missed_peak]>0.5*th:
                                QRS.append(missed_peak)
                                QRS.sort()
                                break

                if len(QRS)>2:
                    RR.append(QRS[-1]-QRS[-2])
                    if len(RR)>8:
                        RR.pop(0)
                    RR_ave = int(np.mean(RR))

            else:
                n_pks.append(ma[peak])
                if len(n_pks)>8:
                    n_pks.pop(0)
                n_pks_ave = np.mean(n_pks)

            th = n_pks_ave + 0.45*(s_pks_ave-n_pks_ave)

            i+=1

        QRS.pop(0)

        window = int(0.1*self.fs)

        r_peaks = searchBack(QRS, unfiltered_ecg, window)

        return r_peaks

    def wqrs_detector(self, unfiltered_ecg): # bad
        """
        based on W Zong, GB Moody, D Jiang
        A Robust Open-source Algorithm to Detect Onset and Duration of QRS
        Complexes
        In: 2003 IEEE
        """

        def butter_lowpass_filter(data, cutoff):
            nyq = 0.5 * self.fs
            order = 2

            normal_cutoff = cutoff / nyq

            b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
            y = signal.lfilter(b, a, data)
            return y

        def length_transfrom(x, w):
            tmp = []
            for i in range(w, len(x)):
                chunk = x[i - w:i]
                ll = np.sum(
                    np.sqrt(np.power(1 / self.fs, 2) * np.ones(w - 1) + np.power(np.diff(chunk), 2))
                )
                tmp.append(ll)
            l = [tmp[0]] * w

            return l + tmp

        def threshold(x):
            peaks = []
            u = MWA_convolve(x, 10 * self.fs)
            for i in range(len(x)):
                if (len(peaks) == 0 or i > peaks[-1] + (self.fs * 0.35)) and x[i] > u[i]:
                    peaks.append(i)
            return peaks

        y = butter_lowpass_filter(unfiltered_ecg, 15)
        y = length_transfrom(y, int(np.ceil(self.fs * 0.13)))
        return threshold(y)

    
    def christov_detector(self, unfiltered_ecg): # bad
        total_taps = 0

        b = np.ones(int(0.02*self.fs))
        b = b/int(0.02*self.fs)
        total_taps += len(b)
        a = [1]

        MA1 = signal.lfilter(b, a, unfiltered_ecg)

        b = np.ones(int(0.028*self.fs))
        b = b/int(0.028*self.fs)
        total_taps += len(b)
        a = [1]

        MA2 = signal.lfilter(b, a, MA1)

        Y = []
        for i in range(1, len(MA2)-1):
            
            diff = abs(MA2[i+1]-MA2[i-1])

            Y.append(diff)

        b = np.ones(int(0.040*self.fs))
        b = b/int(0.040*self.fs)
        total_taps += len(b)
        a = [1]

        MA3 = signal.lfilter(b, a, Y)

        MA3[0:total_taps] = 0

        ms50 = int(0.05*self.fs)
        ms200 = int(0.2*self.fs)
        ms1200 = int(1.2*self.fs)
        ms350 = int(0.35*self.fs)

        M = 0
        newM5 = 0
        M_list = []
        MM = []
        M_slope = np.linspace(1.0, 0.6, ms1200-ms200)
        F = 0
        F_list = []
        R = 0
        RR = []
        Rm = 0
        R_list = []

        MFR = 0
        MFR_list = []

        QRS = []

        for i in range(len(MA3)):

            # M
            if i < 5*self.fs:
                M = 0.6*np.max(MA3[:i+1])
                MM.append(M)
                if len(MM)>5:
                    MM.pop(0)

            elif QRS and i < QRS[-1]+ms200:
                newM5 = 0.6*np.max(MA3[QRS[-1]:i])
                if newM5>1.5*MM[-1]:
                    newM5 = 1.1*MM[-1]

            elif QRS and i == QRS[-1]+ms200:
                if newM5==0:
                    newM5 = MM[-1]
                MM.append(newM5)
                if len(MM)>5:
                    MM.pop(0)    
                M = np.mean(MM)    
            
            elif QRS and i > QRS[-1]+ms200 and i < QRS[-1]+ms1200:

                M = np.mean(MM)*M_slope[i-(QRS[-1]+ms200)]

            elif QRS and i > QRS[-1]+ms1200:
                M = 0.6*np.mean(MM)

            # F
            if i > ms350:
                F_section = MA3[i-ms350:i]
                max_latest = np.max(F_section[-ms50:])
                max_earliest = np.max(F_section[:ms50])
                F = F + ((max_latest-max_earliest)/150.0)

            # R
            if QRS and i < QRS[-1]+int((2.0/3.0*Rm)):

                R = 0

            elif QRS and i > QRS[-1]+int((2.0/3.0*Rm)) and i < QRS[-1]+Rm:

                dec = (M-np.mean(MM))/1.4
                R = 0 + dec


            MFR = M+F+R
            M_list.append(M)
            F_list.append(F)
            R_list.append(R)
            MFR_list.append(MFR)

            if not QRS and MA3[i]>MFR:
                QRS.append(i)
            
            elif QRS and i > QRS[-1]+ms200 and MA3[i]>MFR:
                QRS.append(i)
                if len(QRS)>2:
                    RR.append(QRS[-1]-QRS[-2])
                    if len(RR)>5:
                        RR.pop(0)
                    Rm = int(np.mean(RR))

        QRS.pop(0)
        r_peaks = []
        search_samples = int(0.05*self.fs)

        for i in QRS:
            if i < search_samples:
                section = unfiltered_ecg[0:search_samples]
                r_peaks.append(np.argmax(section))
            elif i+search_samples>len(unfiltered_ecg):
                section = unfiltered_ecg[i:]
                r_peaks.append(np.argmax(section)+i)
            else:
                section = unfiltered_ecg[i-search_samples:i+search_samples]
                r_peaks.append(np.argmax(section)+(i-search_samples))

        return r_peaks

    def engzee_segmenter(signal=None, sampling_rate=1000., threshold=0.48): #Can't get this to work
        """ECG R-peak segmentation algorithm.

        Follows the approach by Engelse and Zeelenberg [EnZe79]_ with the
        modifications by Lourenco *et al.* [LSLL12]_.

        Parameters
        ----------
        signal : array
            Input filtered ECG signal.
        sampling_rate : int, float, optional
            Sampling frequency (Hz).
        threshold : float, optional
            Detection threshold.

        Returns
        -------
        rpeaks : array
            R-peak location indices.

        References
        ----------
        .. [EnZe79] W. Engelse and C. Zeelenberg, "A single scan algorithm for
           QRS detection and feature extraction", IEEE Comp. in Cardiology,
           vol. 6, pp. 37-42, 1979
        .. [LSLL12] A. Lourenco, H. Silva, P. Leite, R. Lourenco and A. Fred,
           "Real Time Electrocardiogram Segmentation for Finger Based ECG
           Biometrics", BIOSIGNALS 2012, pp. 49-54, 2012

        """

        # check inputs
        if signal is None:
            raise TypeError("Please specify an input signal.")

        # algorithm parameters
        changeM = int(0.75 * sampling_rate)
        Miterate = int(1.75 * sampling_rate)
        v250ms = int(0.25 * sampling_rate)
        v1200ms = int(1.2 * sampling_rate)
        v1500ms = int(1.5 * sampling_rate)
        v180ms = int(0.18 * sampling_rate)
        p10ms = int(np.ceil(0.01 * sampling_rate))
        p20ms = int(np.ceil(0.02 * sampling_rate))
        err_kill = int(0.01 * sampling_rate)
        inc = 1
        mmth = threshold
        mmp = 0.2

        # Differentiator (1)
        y1 = [signal[i] - signal[i - 4] for i in range(4, len(signal))]

        # Low pass filter (2)
        c = [1, 4, 6, 4, 1, -1, -4, -6, -4, -1]
        y2 = np.array([np.dot(c, y1[n - 9:n + 1]) for n in range(9, len(y1))])
        y2_len = len(y2)

        # vars
        MM = mmth * max(y2[:Miterate]) * np.ones(3)
        MMidx = 0
        Th = np.mean(MM)
        NN = mmp * min(y2[:Miterate]) * np.ones(2)
        NNidx = 0
        ThNew = np.mean(NN)
        update = False
        nthfpluss = []
        rpeaks = []

        # Find nthf+ point
        while True:
            # If a previous intersection was found, continue the analysis from there
            if update:
                if inc * changeM + Miterate < y2_len:
                    a = (inc - 1) * changeM
                    b = inc * changeM + Miterate
                    Mnew = mmth * max(y2[a:b])
                    Nnew = mmp * min(y2[a:b])
                elif y2_len - (inc - 1) * changeM > v1500ms:
                    a = (inc - 1) * changeM
                    Mnew = mmth * max(y2[a:])
                    Nnew = mmp * min(y2[a:])
                if len(y2) - inc * changeM > Miterate:
                    MM[MMidx] = Mnew if Mnew <= 1.5 * \
                                        MM[MMidx - 1] else 1.1 * MM[MMidx - 1]
                    NN[NNidx] = Nnew if abs(
                        Nnew) <= 1.5 * abs(NN[NNidx - 1]) else 1.1 * NN[NNidx - 1]
                MMidx = np.mod(MMidx + 1, len(MM))
                NNidx = np.mod(NNidx + 1, len(NN))
                Th = np.mean(MM)
                ThNew = np.mean(NN)
                inc += 1
                update = False
            if nthfpluss:
                lastp = nthfpluss[-1] + 1
                if lastp < (inc - 1) * changeM:
                    lastp = (inc - 1) * changeM
                y22 = y2[lastp:inc * changeM + err_kill]
                # find intersection with Th
                try:
                    nthfplus = np.intersect1d(np.nonzero(
                        y22 > Th)[0], np.nonzero(y22 < Th)[0] - 1)[0]
                except IndexError:
                    if inc * changeM > len(y2):
                        break
                    else:
                        update = True
                        continue
                # adjust index
                nthfplus += int(lastp)
                # if a previous R peak was found:
                if rpeaks:
                    # check if intersection is within the 200-1200 ms interval. Modification: 300 ms -> 200 bpm
                    if nthfplus - rpeaks[-1] > v250ms and nthfplus - rpeaks[-1] < v1200ms:
                        pass
                    # if new intersection is within the <200ms interval, skip it. Modification: 300 ms -> 200 bpm
                    elif nthfplus - rpeaks[-1] < v250ms:
                        nthfpluss += [nthfplus]
                        continue
            # no previous intersection, find the first one
            else:
                try:
                    aux = np.nonzero(
                        y2[(inc - 1) * changeM:inc * changeM + err_kill] > Th)[0]
                    bux = np.nonzero(
                        y2[(inc - 1) * changeM:inc * changeM + err_kill] < Th)[0] - 1
                    nthfplus = int((inc - 1) * changeM) + \
                               np.intersect1d(aux, bux)[0]
                except IndexError:
                    if inc * changeM > len(y2):
                        break
                    else:
                        update = True
                        continue
            nthfpluss += [nthfplus]
            # Define 160ms search region
            windowW = np.arange(nthfplus, nthfplus + v180ms)
            # Check if the condition y2[n] < Th holds for a specified
            # number of consecutive points (experimentally we found this number to be at least 10 points)"
            i, f = windowW[0], windowW[-1] if windowW[-1] < len(y2) else -1
            hold_points = np.diff(np.nonzero(y2[i:f] < ThNew)[0])
            cont = 0
            for hp in hold_points:
                if hp == 1:
                    cont += 1
                    if cont == p10ms - 1:  # -1 is because diff eats a sample
                        max_shift = p20ms  # looks for X's max a bit to the right
                        if nthfpluss[-1] > max_shift:
                            rpeaks += [np.argmax(signal[i -
                                                        max_shift:f]) + i - max_shift]
                        else:
                            rpeaks += [np.argmax(signal[i:f]) + i]
                        break
                else:
                    cont = 0

        #rpeaks = sorted(list(set(rpeaks)))
        rpeaks = np.array(rpeaks, dtype='int')
        return rpeaks

        #return utils.ReturnTuple((rpeaks,), ('rpeaks',))
    
    def engzee_detector(self, unfiltered_ecg):
                
        f1 = 48/self.fs
        f2 = 52/self.fs
        b, a = signal.butter(4, [f1*2, f2*2], btype='bandstop')
        filtered_ecg = signal.lfilter(b, a, unfiltered_ecg)

        diff = np.zeros(len(filtered_ecg))
        for i in range(4, len(diff)):
            diff[i] = filtered_ecg[i]-filtered_ecg[i-4]

        ci = [1,4,6,4,1]        
        low_pass = signal.lfilter(ci, 1, diff)

        low_pass[:int(0.2*self.fs)] = 0
      
        ms200 = int(0.2*self.fs)
        ms1200 = int(1.2*self.fs)        
        ms160 = int(0.16*self.fs)
        neg_threshold = int(0.01*self.fs)

        M = 0
        M_list = []
        neg_m = []
        MM = []
        M_slope = np.linspace(1.0, 0.6, ms1200-ms200)

        QRS = []
        r_peaks = []

        counter = 0

        thi_list = []
        thi = False
        thf_list = []
        thf = False

        for i in range(len(low_pass)):

            # M
            if i < 5*self.fs:
                M = 0.6*np.max(low_pass[:i+1])
                MM.append(M)
                if len(MM)>5:
                    MM.pop(0)

            elif QRS and i < QRS[-1]+ms200:

                newM5 = 0.6*np.max(low_pass[QRS[-1]:i])

                if newM5>1.5*MM[-1]:
                    newM5 = 1.1*MM[-1]

            elif QRS and i == QRS[-1]+ms200:
                MM.append(newM5)
                if len(MM)>5:
                    MM.pop(0)    
                M = np.mean(MM)    
            
            elif QRS and i > QRS[-1]+ms200 and i < QRS[-1]+ms1200:

                M = np.mean(MM)*M_slope[i-(QRS[-1]+ms200)]

            elif QRS and i > QRS[-1]+ms1200:
                M = 0.6*np.mean(MM)

            M_list.append(M)
            neg_m.append(-M)


            if not QRS and low_pass[i]>M:
                QRS.append(i)
                thi_list.append(i)
                thi = True
            
            elif QRS and i > QRS[-1]+ms200 and low_pass[i]>M:
                QRS.append(i)
                thi_list.append(i)
                thi = True

            if thi and i<thi_list[-1]+ms160:
                if low_pass[i]<-M and low_pass[i-1]>-M:
                    #thf_list.append(i)
                    thf = True
                    
                if thf and low_pass[i]<-M:
                    thf_list.append(i)
                    counter += 1
                
                elif low_pass[i]>-M and thf:
                    counter = 0
                    thi = False
                    thf = False
            
            elif thi and i>thi_list[-1]+ms160:
                    counter = 0
                    thi = False
                    thf = False                                        
            
            if counter>neg_threshold:
                unfiltered_section = unfiltered_ecg[thi_list[-1]-int(0.01*self.fs):i]
                r_peaks.append(np.argmax(unfiltered_section)+thi_list[-1]-int(0.01*self.fs))
                counter = 0
                thi = False
                thf = False

        return r_peaks

    
    def matched_filter_detector(self, unfiltered_ecg):

        current_dir = pathlib.Path(__file__).resolve()
        
        if self.fs == 250:
            template_dir = current_dir.parent/'templates'/'template_250hz.csv'
            template = np.loadtxt(template_dir)
        elif self.fs == 360:
            template_dir = current_dir.parent/'templates'/'template_360hz.csv'
            template = np.loadtxt(template_dir)
        else:
            print('\n!!No template for this frequency!!\n')

        f0 = 0.1/self.fs
        f1 = 48/self.fs

        b, a = signal.butter(4, [f0*2, f1*2], btype='bandpass')

        prefiltered_ecg = signal.lfilter(b, a, unfiltered_ecg)

        matched_coeffs = template[::-1]  #time reversing template

        detection = signal.lfilter(matched_coeffs, 1, prefiltered_ecg)  # matched filter FIR filtering
        squared = detection*detection  # squaring matched filter output
        squared[:len(template)] = 0

        squared_peaks = panPeakDetect(squared, self.fs)
  
        r_peaks = searchBack(squared_peaks, unfiltered_ecg, len(template))

        return r_peaks

    
    def swt_detector(self, unfiltered_ecg):

        swt_level=3
        padding = -1
        for i in range(1000):
            if (len(unfiltered_ecg)+i)%2**swt_level == 0:
                padding = i
                break

        if padding > 0:
            unfiltered_ecg = np.pad(unfiltered_ecg, (0, padding), 'edge')
        elif padding == -1:
            print("Padding greater than 1000 required\n")    

        swt_ecg = pywt.swt(unfiltered_ecg, 'db3', level=swt_level)
        swt_ecg = np.array(swt_ecg)
        swt_ecg = swt_ecg[0, 1, :]

        squared = swt_ecg*swt_ecg

        N = int(0.12*self.fs)
        mwa = MWA(squared, N)
        mwa[:int(0.2*self.fs)] = 0

        mwa_peaks = panPeakDetect(mwa, self.fs)

        r_peaks = searchBack(mwa_peaks, unfiltered_ecg, N)

        return r_peaks


    def pan_tompkins_detector(self, unfiltered_ecg):
        
        f1 = 5/self.fs
        f2 = 15/self.fs

        b, a = signal.butter(1, [f1*2, f2*2], btype='bandpass')

        filtered_ecg = signal.lfilter(b, a, unfiltered_ecg)        

        diff = np.diff(filtered_ecg) 

        squared = diff*diff

        N = int(0.12*self.fs)
        mwa = MWA(squared, N)
        mwa[:int(0.2*self.fs)] = 0

        mwa_peaks = panPeakDetect(mwa, self.fs)

        r_peaks = searchBack(mwa_peaks, unfiltered_ecg, N)

        return r_peaks


    def two_average_detector(self, unfiltered_ecg):       
        
        f1 = 8/self.fs
        f2 = 20/self.fs

        b, a = signal.butter(2, [f1*2, f2*2], btype='bandpass')

        filtered_ecg = signal.lfilter(b, a, unfiltered_ecg)

        window1 = int(0.12*self.fs)
        mwa_qrs = MWA(abs(filtered_ecg), window1)

        window2 = int(0.6*self.fs)
        mwa_beat = MWA(abs(filtered_ecg), window2)

        blocks = np.zeros(len(unfiltered_ecg))
        block_height = np.max(filtered_ecg)

        for i in range(len(mwa_qrs)):
            if mwa_qrs[i] > mwa_beat[i]:
                blocks[i] = block_height
            else:
                blocks[i] = 0

        QRS = []

        for i in range(1, len(blocks)):
            if blocks[i-1] == 0 and blocks[i] == block_height:
                start = i
            
            elif blocks[i-1] == block_height and blocks[i] == 0:
                end = i-1

                if end-start>int(0.08*self.fs):
                    detection = np.argmax(filtered_ecg[start:end+1])+start
                    if QRS:
                        if detection-QRS[-1]>int(0.3*self.fs):
                            QRS.append(detection)
                    else:
                        QRS.append(detection)

        r_peaks = searchBack(QRS, unfiltered_ecg, window1)

        return r_peaks


def MWA(input_array, window_size):

    mwa = np.zeros(len(input_array))
    for i in range(len(input_array)):
        if i < window_size:
            section = input_array[0:i]
        else:
            section = input_array[i-window_size:i]
        
        if i!=0:
            mwa[i] = np.mean(section)
        else:
            mwa[i] = input_array[i]

    return mwa


def MWA_convolve(input_array, window_size):
    ret = np.pad(input_array, (window_size - 1, 0), 'constant', constant_values=(0, 0))
    ret = np.convolve(ret, np.ones(window_size), 'valid')

    for i in range(1, window_size):
        ret[i - 1] = ret[i - 1] / i
    ret[window_size - 1:] = ret[window_size - 1:] / window_size

    return ret

def normalise(input_array):

    output_array = (input_array-np.min(input_array))/(np.max(input_array)-np.min(input_array))

    return output_array


def searchBack(detected_peaks, unfiltered_ecg, search_samples):

    r_peaks = []
    window = search_samples

    for i in detected_peaks:
        if i<window:
            section = unfiltered_ecg[:i]
            r_peaks.append(np.argmax(section))
        else:
            section = unfiltered_ecg[i-window:i]
            r_peaks.append(np.argmax(section)+i-window)

    return np.array(r_peaks)


def panPeakDetect(detection, fs):    

    min_distance = int(0.25*fs)
    peaks, _ = signal.find_peaks(detection, distance=min_distance)      

    signal_peaks = []
    noise_peaks = []

    SPKI = 0.0
    NPKI = 0.0

    threshold_I1 = 0.0
    threshold_I2 = 0.0

    RR_missed = 0
    index = 0
    indexes = []

    missed_peaks = []
    for peak in peaks:

        if detection[peak] > threshold_I1:
               
            signal_peaks.append(peak)
            indexes.append(index)
            SPKI = 0.125*detection[signal_peaks[-1]] + 0.875*SPKI
            if RR_missed!=0:
                if signal_peaks[-1]-signal_peaks[-2]>RR_missed:
                    missed_section_peaks = peaks[indexes[-2]+1:indexes[-1]]
                    missed_section_peaks2 = []
                    for missed_peak in missed_section_peaks:
                        if missed_peak-signal_peaks[-2]>min_distance and signal_peaks[-1]-missed_peak>min_distance and detection[missed_peak]>threshold_I2:
                            missed_section_peaks2.append(missed_peak)

                    if len(missed_section_peaks2)>0:           
                        missed_peak = missed_section_peaks2[np.argmax(detection[missed_section_peaks2])]
                        missed_peaks.append(missed_peak)
                        signal_peaks.append(signal_peaks[-1])
                        signal_peaks[-2] = missed_peak   
        
        else:
            noise_peaks.append(peak)
            NPKI = 0.125*detection[noise_peaks[-1]] + 0.875*NPKI
        
        threshold_I1 = NPKI + 0.25*(SPKI-NPKI)
        threshold_I2 = 0.5*threshold_I1

        if len(signal_peaks)>8:
            RR = np.diff(signal_peaks[-9:])
            RR_ave = int(np.mean(RR))
            RR_missed = int(1.66*RR_ave)

        index = index+1

    signal_peaks.pop(0)

    return signal_peaks