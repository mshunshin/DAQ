import os
import datetime

import numpy as np
import pandas as pd

import hdy


database_dir = '/Volumes/Matt-Data/Box/LPM-Data/DFT data for analysis in paper/'
database_fn = 'DFTregister-new.csv'

#MODE = 'double_count_fix_true'
MODE = 'normal'

db = pd.read_csv(os.path.join(database_dir, database_fn))

#db = db[db['Patient'] == 'DKDFT033']
#db = db[db['Experiment'] == 'Exp10']

patients = db['Patient']
experiments = db['Experiment']
periods = db['Period']

results_list = []



for patient, exp, period in zip(patients, experiments, periods):
    try:

        if pd.isnull(exp) or pd.isnull(patient):
            continue
        pt = hdy.LaserAnalysis(database_dir=database_dir, database_fn=database_fn, patient=patient, exp=exp, mode=MODE)
        pt.process()
        results_list.append(pt.results)
    except Exception as e:
       print("FuckUP " + patient + exp)
       print(e)

now = datetime.datetime.now().isoformat()[0:19].replace(":", "").replace("-", "")

results_db = pd.DataFrame(results_list, columns=list(results_list[0].keys()))
results_fn = os.path.splitext(database_fn)[0]+'-' + now + "-" + MODE + "-" + '-results.csv'
results_fl = os.path.join(database_dir, results_fn)
results_db.to_csv(results_fl, index=False)