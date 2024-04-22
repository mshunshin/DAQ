#import matplotlib
#matplotlib.use('Qt5Agg')
import os
os.environ['PYQTGRAPH_QT_LIB'] = 'PySide6'

import sys

import numpy as np
import pandas as pd
import scipy

from PySide6 import QtWidgets

import hdy

def main():
    #database_dir = '/Volumes/Matt-Data/Box/Kyocera-Data'
    #database_fn = 'Matt-Laser.csv'
    #patient = 'elaine'
    #exp="Exp1c"

    database_dir = '/Volumes/Matt-Data/Box/LPM-Data/DFT data for analysis in paper/'
    database_fn = 'DFTregister-new.csv'
    patient = 'DKDFT013 (d.keene@imperial.ac.uk)'
    exp="Exp20"

    database_dir = '/Users/matthew/Library/CloudStorage/Dropbox/DK_PhD/DATA/DKEPS/'
    database_fn = 'EPS-DB.csv'
    patient = '1066'
    exp = "Exp1"
    mode = "normal"


    #patient = 'DKDFT048'
    #exp="Exp8"

    #DFT


    #DFT
    #database_fn = 'Haem.csv'
    #patient = 'DKDFT009'
    #exp="Exp1"

    #Conscious VT
    #database_fn = 'Haem.csv'
    #patient = 'Conscious-VT'
    #exp="Exp5"
    #exp="Exp6"

    #Conscious VT
    #database_fn = 'Haem.csv'
    #patient = 'inpockettest'
    #exp="Exp3" #RV
    #exp="Exp4" #LV


    #Conscious VT
    #database_fn = 'Haem.csv'
    #patient = 'DKEPS015'
    #exp="Exp43" #Poorly tolerated VT150 - Transcranial doppler signal

    #Conscious VT
    #database_fn = 'Haem.csv'
    #patient = 'DKEPS13'
    #exp="Exp29" #Poorly tolerated VT150 - Also how SMJ fails at end.

    #A120
    #database_fn = 'Haem.csv'
    #patient = 'DKEPS001'
    #exp="Exp17"

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("LDPM Viewer")

    pt = hdy.LaserAnalysis(database_dir=database_dir, database_fn=database_fn, patient=patient, exp=exp, mode=mode)

    m = hdy.LaserGui(pt)

    sys.exit(app.exec())

if __name__ == '__main__':
    main()
