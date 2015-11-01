import os
import numpy as np
import pandas as pd
import scipy

import hdy

import matplotlib.pyplot as plt


database_dir = '/Volumes/Matt-Data/ZIF-Data'
database_fn = 'RSP08-His-Haem.csv'

db = pd.read_csv(os.path.join(database_dir, database_fn))

patients = db['Patient']
experiments = db['Experiment']

results_list = []

for patient, exp in zip(patients, experiments):
    pt = hdy.MattOptimisation(database_dir=database_dir, database_fn=database_fn, patient=patient, exp=exp)
    pt.process()
#    pt.plot_clusters()
    results_list.extend(pt.results)

results_db = pd.DataFrame(results_list, columns=list(results_list[0].keys()))
results_fn = os.path.splitext(database_fn)[0]+'-results.csv'
results_fl = os.path.join(database_dir, "Output" ,results_fn)
results_db.to_csv(results_fl, index=False)

print("Done")