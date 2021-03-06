#import matplotlib
#matplotlib.use('QtAgg')

import os
os.environ['PYQTGRAPH_QT_LIB'] = 'PySide6'

import sys

import numpy as np
import pandas as pd
import scipy

from PySide6 import QtWidgets

import hdy

SELECTION = "get_input"

def main():

    if SELECTION == 'get_input':
        database_dir = input("Database dir: ")
        database_fn = input("Filename of csv: ")
        patient = input("Patient: ")
        exp = input("Experiment name: ")

    elif False:
        database_dir = '/Volumes/Matt-Data/Box/Kyocera-Data'
        database_fn = 'Matt-Laser.csv'
        patient = 'elaine'
        exp="Exp1c"
    elif False:
        database_dir = '/Volumes/Matt-Data/Box/LPM-Data/DFT data for analysis in paper/'
        database_fn = 'DFTregister-new.csv'
        patient = 'DKDFT013 (d.keene@imperial.ac.uk)'
        exp="Exp20"

    elif False:
        database_dir = '/Users/matthew/Dropbox/DK_PhD/DATA/DKEPS/'
        database_fn = 'EPS-DB draft.csv'
        patient = 'DKEPS001'
        exp = "Exp1"
        mode = "normal"
    else:
        database_dir = '/Users/matthew/Dropbox/VTF MSS Review/'
        database_fn = 'clin DB.csv'
        patient = 'RR'
        exp = "Exp1"

    mode = "normal"

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("LDPM Viewer")

    pt = hdy.LaserAnalysis(database_dir=database_dir, database_fn=database_fn, patient=patient, exp=exp, mode=mode)

    m = hdy.LaserGui(pt)

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
