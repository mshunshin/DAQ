import numpy as np
import scipy
import scipy.signal


def IQR(x, pct):
    upper, lower = np.percentile(x, [100-pct, pct])
    return upper - lower


def MQR(x, pct1, pct2):
    upper, lower = np.percentile(x, [pct1, pct2])
    return upper-lower


def max_max_element(a):
    return max(np.where(a==a.max())[0])


def find_nearest_value(array,value):
    idx = (np.abs(array-value)).argmin()
    return (array[idx])


def find_nearest_idx(array,value):
    idx = (np.abs(array-value)).argmin()
    return (idx)


def find_nearest_above_idx(my_array, target):
    diff = my_array - target
    mask = np.ma.less_equal(diff, 0)
    # We need to mask the negative differences and zero
    # since we are looking for values above
    if np.all(mask):
        return None # returns None if target is greater than any value
    masked_diff = np.ma.masked_array(diff, mask)
    return masked_diff.argmin()


def find_nearest_below_idx(my_array, target):
    diff = my_array - target
    mask = np.ma.greater_equal(diff, 0)
    # We need to mask the negative differences and zero
    # since we are looking for values above
    if np.all(mask):
        return None # returns None if target is greater than any value
    masked_diff = np.ma.masked_array(diff, mask)
    return masked_diff.argmax()

from collections import deque
from itertools import islice
from bisect import insort

class RunningMedian:
    'Slow running-median with O(n) updates where n is the window size'

    def __init__(self, n, iterable):
        self.it = iter(iterable)
        self.queue = deque(islice(self.it, n))
        self.sortedlist = sorted(self.queue)

    def __iter__(self):
        queue = self.queue
        sortedlist = self.sortedlist
        midpoint = len(queue) // 2
        yield sortedlist[midpoint]
        for newelem in self.it:
            oldelem = queue.popleft()
            sortedlist.remove(oldelem)
            queue.append(newelem)
            insort(sortedlist, newelem)
            yield sortedlist[midpoint]

def butter_lowpass(cutOff, fs, order=5):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = scipy.signal.butter(order, normalCutoff, btype='low', analog = False)
    return b, a

def butter_lowpass_filter(data, cutOff, fs, order=5):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    sos = scipy.signal.butter(N=order, Wn=[lowcut, highcut], btype='band', output='sos', fs=fs)
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = scipy.signal.butter(N=order, Wn=[lowcut, highcut], btype='band', output='sos', fs=fs)
    y = scipy.signal.sosfiltfilt(sos=sos, x=data)
    return y

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def lazy_property(fn):
    '''Decorator that makes a property lazy-evaluated.
    '''
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazy_property