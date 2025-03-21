import numpy as np
from scipy import signal
from sklearn.decomposition import PCA


def ampd(data):
    """Automatic multiscale-based peak detection.

    A robust peak-detection algorithm suitable for noisy periodic or quasi-periodic data,
    with no free parameters [1]_.

    Parameters
    ----------
    data : array-like
        A periodic or quasi-periodic signal, in a 1D array (or squeezable to 1D).

    Returns
    -------
    peaks : array
        Indices of peaks in `data`.

    References
    ----------
    .. [1] Scholkmann, F., Boss, J., & Wolf, M. (2012).
    An efficient algorithm for automatic peak detection in noisy periodic and quasi-periodic signals.
    *Algorithms, 5*, 588-603.


    """
    # 1. Calculate LMS.
    lms = _local_maxima_scalogram(data)

    # 2. Row-wise sum.
    gamma = lms.sum(axis=1)

    # 3. Remove all rows from lms with k > lambda.
    global_min = gamma.argmin()  # lambda

    # 4. Calculate column-wise standard deviation.
    stds = lms[:global_min + 1].std(axis=0, ddof=1)

    # 5. Return all indices i for which std_i == 0.
    return np.nonzero(stds == 0)[0]


def _local_maxima_scalogram(data):
    a = 1

    data = signal.detrend(_validate_vector(data))
    N = data.shape[0]
    L = int(np.ceil(N/2) - 1)

    # First, construct the LMS as if all the elements were r + a.
    # Then we'll change the other values to zero.
    lms = np.random.sample((L, N)) + a

    # Create shifted forward and shifted back matrices the same size as the LMS.
    shifted_back = np.empty(lms.shape)
    shifted_forward = np.empty(lms.shape)
    # The choice of zeroes seems to be based on the previous element (i -1) so we just roll the data forward 1.
    for shift_ix in range(L):
        shift = shift_ix + 1
        shifted_back[shift_ix, :] = np.roll(data, -shift)
        shifted_forward[shift_ix, :] = np.roll(data, shift)

    # We only want to check elements with k + 1 <= i <= N - k.
    # i: index in original data.
    # k: magnitude of shift.
    shift, data_ix = np.indices(lms.shape)
    shift += 1
    elements_to_check = (shift + 1 <= data_ix) & (data_ix <= N - shift)

    # To do the comparisons in one sweep we need a tiled version of data.
    data = np.tile(data, (L, 1))
    lms[elements_to_check & (data > shifted_back) & (data > shifted_forward)] = 0

    return lms


def _validate_vector(data):
    data = np.squeeze(data)
    if len(data.shape) > 1:
        raise ValueError('input cannot be cast to 1D vector')
    return data


def _validate_vectors(a, b):
    if b is None:
        a = np.squeeze(a)
        if a.ndim != 2 or a.shape[1] != 2:
            raise ValueError('with one input, array must have be 2D with two columns')
        a, b = a[:, 0], a[:, 1]

    a = np.squeeze(a)
    if a.ndim > 1:
        a = _collapse(a)

    b = np.squeeze(b)
    if b.ndim > 1:
        b = _collapse(b)

    return a, b


def _collapse(data):
    return np.squeeze(PCA(n_components=1).fit_transform(data))
