from scipy import stats

import numpy as np
import bottleneck as bn

def matt_mode(data):
    data_int = data.astype(int)
    if len(data_int) < 1:
        return 0
    return np.bincount(data_int).argmax()

def compute_corr_fast(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    mx = x.mean(axis=-1)
    my = y.mean(axis=-1)
    xm, ym = x - mx[..., None], y - my[..., None]
    r_num = np.add.reduce(xm * ym, axis=-1)
    r_den = np.sqrt(stats.ss(xm, axis=-1) * stats.ss(ym, axis=-1))
    r = r_num / r_den
    return r

def corr2_coeff(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]

    # Sum of squares across rows
    #ssA = (A_mA**2).sum(1)
    #ssB = (B_mB**2).sum(1)

    ssA = bn.ss(A_mA, axis=1)
    ssB = bn.ss(B_mB, axis=1)

    # Finally get corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))