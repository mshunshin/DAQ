import os
import numpy as np
import pandas as pd
import scipy

import hdy

import matplotlib.pyplot as plt


database_dir = '/Volumes/Matt-Data/LPM-Data'
database_fn = 'Haem.csv'
patient = 'DKDFT003'
exp="Exp7"


pt = hdy.LaserAnalysis.from_db(database_dir=database_dir, database_fn=database_fn, patient=patient, exp=exp)
pt.process()
m = hdy.LaserGUI(pt)


print("done")

