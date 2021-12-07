import os
import datetime

import pandas as pd

import hdy


#database_dir = '/Volumes/Matt-Data/Box/LPM-Data/DFT data for analysis in paper/'
#database_fn = 'DFTregister-new.csv'

#database_dir = '/Volumes/Matt-Data/Dropbox/DK_PhD/DATA/Exercise_Laser/'
#database_fn = 'Exercise-DB.csv'

#database_dir = '/Volumes/Matt-Data/Dropbox/DK_PhD/DATA/VT Mechanisms and Termination/'
#database_fn = 'TREAT DB.csv'

#database_dir = '/Volumes/Matt-Data/Dropbox/DK_PhD/DATA/VT Mechanisms and Termination/'
#database_fn = 'TREAT DB.csv'

#database_dir = '/Volumes/Matt-Data/Dropbox/DK_PhD/DATA/DKEPS/'
#database_fn = 'EPS-DB.csv'

#database_dir = '/Volumes/Matt-Data/Dropbox/LDPM-DJ/DATA/'
#database_fn = 'DB.csv'

#database_dir = '/Users/matthew/Box/BSC-Shifa Data files/'

database_dir = '/users/matthew/Dropbox/VT_Mechanisms/'
database_fn = 'mech_2_monica.csv'
#database_fn = 'Tilt Test Data.csv'

MODE = 'normal'

#MODE = 'double_count_fix'

#MODE = 'double_count_fix'
#MODE = 'atrial'

db = pd.read_csv(os.path.join(database_dir, database_fn))

#db = db[db['Patient'] == 'UP PT2']
#db = db[db['Experiment'] == 'exp6']

patients = db['Patient']
experiments = db['Experiment']
periods = db['Period']

results_list = []
results_beatbybeat_list = []


for patient, exp, period in zip(patients, experiments, periods):
    try:

        if pd.isnull(exp) or pd.isnull(patient):
            continue
        pt = hdy.LaserAnalysis(database_dir=database_dir, database_fn=database_fn, patient=patient, exp=exp, mode=MODE)
        pt.process()
        results_list.append(pt.results)
        results_beatbybeat_list.append(pt.results_beatbybeat)
    except Exception as e:
       print("FuckUP " + patient + exp)
       print(e)

now = datetime.datetime.now().isoformat()[0:19].replace(":", "").replace("-", "")

results_db = pd.DataFrame(results_list, columns=list(results_list[0].keys()))
results_fn = os.path.splitext(database_fn)[0]+'-' + now + "-" + MODE + "-" + '-results.csv'
results_fl = os.path.join(database_dir, results_fn)
results_db.to_csv(results_fl, index=False)

results_beatbybeat_db = []
for d in results_beatbybeat_list:
    results_beatbybeat_db.append(pd.DataFrame(d, columns=list(d.keys())))
results_beatbybeat_db = pd.concat(results_beatbybeat_db)

results_beatbybeat_fn = os.path.splitext(database_fn)[0]+'-' + now + "-" + MODE + "-" + '-beatbybeat-results.csv'
results_beatbybeat_fl = os.path.join(database_dir, results_beatbybeat_fn)
results_beatbybeat_db.to_csv(results_beatbybeat_fl, index=False)
