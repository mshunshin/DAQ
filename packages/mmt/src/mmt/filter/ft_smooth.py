import numpy as np
import itertools

def ft_smooth(data, box_size=5):

    assert(data.__class__ == np.ndarray)
    assert(data.ndim == 2)
    assert(data.dtype == np.uint8)
    
    box_width = box_size//2

    height = data.shape[0]
    width = data.shape[1]

    f_map = np.zeros_like(data, dtype=np.float32)

    for h, w in itertools.product(np.arange(box_size, height-box_size), np.arange(box_size, width-box_size)):

        box = data[(h-box_width):(h+box_width+1),(w-box_width):(w+box_width+1)]/256

        var = np.sum(np.exp(2*box))/(2*np.size(box))

        i2var = 1/(2*var)

        f_map[h,w] = np.sqrt(i2var)

    ui_map = 255-(f_map*255/np.max(f_map)).astype(np.uint8)
    
    return(ui_map)