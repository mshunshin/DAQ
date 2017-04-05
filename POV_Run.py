import os
from collections import *

import numpy as np
import pandas as pd
import scipy

import hdy

import matplotlib.pyplot as plt

database_fl = '/Volumes/Matt-Data/POV-Data/Haem.csv'
database_dir = os.path.dirname(database_fl)

db = pd.read_csv(database_fl)

patients = db['Patient']
experiments = db['Experiment']

results_list = []

for patient, experiment in zip(patients, experiments):

    try:
        pt_results = OrderedDict()

        source = 'database'

        source_hints ={'database_fl': database_fl,
                       'patient': patient,
                       'exp': experiment,
                       'daq_dir': os.path.join(database_dir, patient, 'Heam')}

        pt = hdy.SyncopeRun(source=source,
                            source_hints=source_hints)

        pt.process()

        output_dir = os.path.join(database_dir, "Output", patient)

        fig_fn = experiment + "-" + "Syncope_Plot.png"
        fig_fl = os.path.join(output_dir, fig_fn)
        pt.plot(fig_fl=fig_fl)

        fig_fn = experiment + "-" + "ECG_ADV.png"
        fig_fl = os.path.join(output_dir, fig_fn)
        pt.plot_ecg_adv(fig_fl=fig_fl)

        fig_fn = experiment + "-" + "LS.png"
        fig_fl = os.path.join(output_dir, fig_fn)
        pt.ecg.plot_LombScargle(fig_fl=fig_fl)

        fig_fn = experiment + "-" + "SDNN.png"
        fig_fl = os.path.join(output_dir, fig_fn)
        pt.ecg.plot_SDNN(fig_fl=fig_fl)

        fig_fn = experiment + "-" + "BP-LS.png"
        fig_fl = os.path.join(output_dir, fig_fn)
        pt.pressure.plot_LombScargle(fig_fl=fig_fl)

        pt_results.update(pt.hints)

        LS_results = pt.ecg.LombScargle._asdict()
        LS_results.pop('power', 0)
        LS_results.pop('freqs', 0)

        BP_results = pt.pressure.LombScargle._asdict()
        BP_results.pop('bp_power', 0)
        BP_results.pop('bp_freqs', 0)

        pt_results.update(LS_results)
        pt_results['SDNN'] = pt.ecg.SDNN
        pt_results.update(BP_results)

        results_list.append(pt_results)

    except Exception as e:
        print(e)
        print("FuckUP", patient, experiment)

results_db = pd.DataFrame(results_list)
results_fl = output_dir = os.path.join(database_dir, "Output", "Results.csv")
results_db.to_csv(results_fl)