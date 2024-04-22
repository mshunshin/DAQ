from .DAQ_File import DAQ_File
from .SDY_File import SDY_File
from .Bard_File import Bard_File
from .Fino_File import Fino_File

from .BaseClasses import DAQSignal, DAQContainerBP, DAQContainerBPAO, DAQContainerBoxA, DAQContainerCranial, DAQContainerECG, DAQContainerLaser

from .SyncopeClasses import SyncopeRun

from .LaserClasses import LaserAnalysis
from .Laser_GUI import LaserGui

from .ADA_GUI import ADAGui
from .AAD_LaserClasses import LaserAnalysis1

from .laser import calc_magic_laser

from .OptClasses import OptAnalysis, OptCollection
from .Opt_GUI import Opt_GUI, Opt_Selector_GUI

from .VolcanoClasses import VolcanoAnalysis

from .DAQ_GUI import DAQ_GUI




def remove_outliers(data):
    data_mean = data.mean()
    data_std = data.std()
    data[(data > (data_mean + 5 * data_std)) | (data < (data_mean - 5 * data_std))] = 0
    return data
