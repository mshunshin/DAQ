import numpy as np

def minimum_error(data):

    assert(data.__class__ == np.ndarray)
    assert(data.ndim == 2)

    h, g = np.histogram(a=data, bins=np.arange(0,np.ceil(np.max(data))))

    g = g[:-1]

    C = np.cumsum(h)
    M = np.cumsum(h * g)
    S = np.cumsum(h * g**2)
    sigma_f = np.sqrt(S/C - (M/C)**2)

    Cb = C[-1] - C
    Mb = M[-1] - M
    Sb = S[-1] - S
    sigma_b = np.sqrt(Sb/Cb - (Mb/Cb)**2)

    P = C/C[-1]

    V = P * np.log(sigma_f) + (1-P)*np.log(sigma_b) - P*np.log(P) - (1-P)*np.log(1-P)

    V[~np.isfinite(V)] = np.inf

    idx = np.argmin(V)

    T = g[idx]

    return(T)