import numpy as np
import itertools

def ft_feature(data, box_size=5):

    assert(data.__class__ == np.ndarray)
    assert(data.ndim == 2)
    assert(data.dtype == np.uint8)

    box_size = (box_size//2)+1

    spectrum = data
    height = spectrum.shape[0]
    width = spectrum.shape[1]

    h_feature_map = np.zeros_like(spectrum, dtype=np.float32)
    v_feature_map = np.zeros_like(spectrum, dtype=np.float32)

    for h, w in itertools.product(np.arange(box_size, height-box_size), np.arange(box_size, width-box_size)):

        upper_box = spectrum[(h-box_size+1):(h+1), (w-box_size+1):(w+box_size)]/128
        lower_box = spectrum[h:(h+box_size), (w-box_size+1):(w+box_size)]/128

        var1 = np.sum(np.exp(2*upper_box))/(2*np.size(upper_box))
        var2 = np.sum(np.exp(2*lower_box))/(2*np.size(lower_box))

        i2var1 = 1/(2*var1)
        i2var2 = 1/(2*var2)

        J = 0.5 * np.exp(-i2var1) * (-np.log(2*var1) + np.log(2*var2) - 1 - i2var1 + var1/var2 + i2var2) +\
        0.5 * np.exp(-i2var2) * (-np.log(2*var2) + np.log(2*var1) - 1 - i2var2 + var2/var1 + i2var1)

        v_feature_map[h,w] = J

    for h, w in itertools.product(np.arange(box_size, height-box_size), np.arange(box_size, width-box_size)):

        left_box = spectrum[(h-box_size+1):(h+box_size), (w-box_size+1):(w+1)]/128
        right_box = spectrum[(h-box_size+1):(h+box_size), (w):(w+box_size)]/128

        var1 = np.sum(np.exp(2*left_box))/(2*np.size(left_box))
        var2 = np.sum(np.exp(2*right_box))/(2*np.size(right_box))

        i2var1 = 1/(2*var1)
        i2var2 = 1/(2*var2)

        J = 0.5 * np.exp(-i2var1) * (-np.log(2*var1) + np.log(2*var2) - 1 - i2var1 + var1/var2 + i2var2) +\
        0.5 * np.exp(-i2var2) * (-np.log(2*var2) + np.log(2*var1) - 1 - i2var2 + var2/var1 + i2var1)

        h_feature_map[h,w] = J

    feature_map = np.sqrt(v_feature_map**2 + h_feature_map**2)

    return(feature_map)