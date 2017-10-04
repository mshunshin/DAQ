import mmt
import scipy.signal
import numpy as np


def calc_magic_laser(ecg_data, laser_data, no_log=False):

    ecg_detect_peaks = mmt.ecg.kathirvel_ecg(ecg_data, 1000)
    ecg_relmax_peaks = scipy.signal.argrelmax(np.abs(ecg_data), order=200)[0]

    final_peaks = []
    for peak in ecg_detect_peaks:
        final_peaks.append(mmt.find_nearest_value(ecg_relmax_peaks, peak))
    ecg_peaks_sample = np.sort(list(set(final_peaks)))

    if not no_log:
        laser_data = laser_data+10
        laser_data = np.log(laser_data) + laser_data/100
    laser_data = mmt.butter_bandpass_filter(laser_data, 0.5, 5.0, 1000, order=2)

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

    if shift > 800:
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

    laser_magic = scipy.signal.savgol_filter(laser_sum/ inc_peaks, 101, 3)

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

    if not no_log:
        laser_magic_value = 100*(np.exp((laser_ptp)/2)-1)
    elif no_log:
        laser_magic_value = laser_ptp / 2

    conf = error/(laser_ptp+0.0001)**2
    conf_pct = 100*(1-conf)
    if conf_pct < 0:
        conf_pct = 0

    print(laser_magic_value)
    return laser_magic_value

