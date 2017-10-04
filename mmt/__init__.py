from . import find_peaks
from . import fit
from . import threshold
from . import smooth
from . import filter
from . import outlier
from . import ecg
from . import stats
from . import peaks
from . import hrv

from .functions import *

def str_to_float(x):
    try:
        result = float(''.join([c for c in x if c in '1234567890.-']))
    except Exception as e:
        result = float(0)

    return(result)