import os
import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd
import scipy.stats
import scipy.signal
import scipy.interpolate

import mmt



from . import *


class OptCollection:

    source_hints_default = {'period': 'Unknown',
                            'patient': 'Unknown'}

    def __init__(self, source, source_hints):

        super().__init__()

        self.source = source
        self.source_hints = {**self.source_hints_default, **source_hints}

        self.daq_dir = source_hints['daq_dir']
        self.save_dir = source_hints['save_dir']

        self.exp_source_list = []
        self.exp_source_hints_list = []
        self.exp_title_list = []

        self.exp_list = []

        self.final_results = None
        self.final_results_db = None

        self.fusion = None

        self.patient = str(self.source_hints['patient'])
        self.period = str(self.source_hints['period'])

        if self.source.startswith('database'):
            self.database_fl = source_hints['database_fl']

            db = pd.read_csv(self.database_fl, dtype={"Patient": str})
            db_pt = db[db.Patient == self.patient]
            self.period = db_pt.Period.iloc[0]
            exps = db_pt.Experiment

            for exp in exps:
                if pd.isnull(exp):
                    continue

                exp_hints = dict(db_pt[db_pt.Experiment == exp].iloc[0])

                exp_source_hints = {}
                exp_source_hints['save_dir'] = self.save_dir
                exp_source_hints['daq_dir'] = self.daq_dir
                exp_source_hints['zip_fn'] = exp_hints['File']
                exp_source_hints['patient'] = self.patient
                exp_source_hints['exp'] = exp_hints['Experiment']
                exp_source_hints['search_for_files'] = True
                exp_source_hints['hints'] = exp_hints

                self.exp_source_hints_list.append(exp_source_hints)
                self.exp_source_list.append('dict')
                self.exp_title_list.append(exp)

        elif self.source.startswith('directory'):

            i = 1
            for file in os.listdir(self.daq_dir):
                if file.endswith(".zip"):
                    exp_source_hints = {}
                    exp_source_hints['save_dir'] = self.save_dir
                    exp_source_hints['daq_dir'] = self.daq_dir
                    exp_source_hints['zip_fn'] = file
                    exp_source_hints['patient'] = self.patient
                    exp_source_hints['exp'] = "Exp" + str(i)

                    exp_hints = {}
                    exp_hints['ECG'] = 'ecg'
                    exp_hints['BPAO'] = 'bpao'
                    exp_hints['BP'] = 'BP'
                    exp_hints['BoxA'] = 'boxa'
                    exp_hints['bp_dist'] = 'bp_dist'
                    exp_hints['bp_prox'] = 'bp_prox'

                    if source.endswith('-invasive'):
                        exp_hints['Pressure'] = 'BPAO'
                    elif source.endswith('hopehf'):
                        exp_hints['Pressure'] = 'BP'
                    elif source.endswith('-noninvasive'):
                        exp_hints['Pressure'] = 'BP'
                    elif source.endswith('bpdist'):
                        exp_hints['Pressure'] = 'bp_dist'
                    elif source.endswith('bpprox'):
                        exp_hints['Pressure'] = 'bp_prox'
                    else:
                        exp_hints['Pressure'] = 'BP'

                    exp_source_hints['hints'] = exp_hints
                    print(exp_source_hints)
                    self.exp_source_hints_list.append(exp_source_hints)
                    self.exp_source_list.append('dict')
                    self.exp_title_list.append("Exp" + str(i))

                    i += 1
        else:
            raise Exception('Unhandled source')

        for source, source_hints in zip(self.exp_source_list, self.exp_source_hints_list):

            exp = OptAnalysis(source=source, source_hints=source_hints)
            self.exp_list.append(exp)

    def calc_results(self, results_type = "SBP"):

        self.final_results = []
        self.results_type = results_type

        print()

        print("Calculating Results")

        for exp in self.exp_list:
            exp.delay.fix_codebook(ref=exp.ref, override_test=exp.override_test)
            exp.calc_results(results_type = results_type)

            self.final_results.extend(exp.results)

        self.final_results_db = pd.DataFrame(self.final_results, columns=list(self.final_results[0].keys()))

        final_results_db = self.final_results_db

        temp1 = final_results_db.groupby(['Patient', 'Period', 'Exp', 'Target', 'Direction', 'Rep', 'Order'])[['SBP']].mean()
        temp2 = temp1.unstack().reset_index()

        temp2['Benefit'] = np.where(temp2['Direction'] == 'AO',
                                    temp2['SBP']['Post'] - temp2['SBP']['Pre'],
                                    temp2['SBP']['Pre'] - temp2['SBP']['Post'])
        temp2['Target_num'] = [mmt.str_to_float(x) for x in temp2['Target']]

        temp3 = temp2[temp2['Target_num'] > 0].groupby(['Target_num'])

        temp3 = temp3.agg(benefit_mean=('Benefit', 'mean'), benefit_sem=('Benefit', 'sem'),
                          benefit_n=('Benefit', 'count'))
        print(temp3)

        tmp = ['Target_num']
        # tmp.extend(temp3.columns.get_level_values(1))
        temp3 = temp3.reset_index()
        # temp3.columns = tmp

        temp3['benefit_upper'] = (temp3['benefit_sem'] * scipy.stats.t.ppf(0.975, df=temp3['benefit_n']))
        temp3['benefit_lower'] = (temp3['benefit_sem'] * scipy.stats.t.ppf(0.975, df=temp3['benefit_n']))

        print(temp3)
        print(temp2)

        x = np.array(temp2['Target_num'])
        y = np.array(temp2['Benefit'])

        x_mask = x > 1

        x = x[x_mask]
        y = y[x_mask]

        x_sort = np.argsort(x)

        x = x[x_sort]
        y = y[x_sort]

        x_all = x.copy()
        y_all = y.copy()

        if self.fusion:
            x_mask = x < self.fusion
            x = x[x_mask]
            y = y[x_mask]

        import statsmodels.api as sm

        X = np.column_stack((x, x ** 2))
        X = sm.add_constant(X)

        model = sm.OLS(y, X)
        results = model.fit()

        # optim_fit = np.polyfit(x=x,y=y,deg=2)

        # Yep - doing stats in python is awful.
        optim_fitter = np.poly1d(results.params[::-1])

        # And it gets worse.
        from statsmodels.stats.outliers_influence import summary_table

        st, results_data, ss2 = summary_table(results, alpha=0.05)

        # fittedvalues = results_data[:, 2]
        # predict_mean_se = results_data[:, 3]

        predict_mean_ci_low, predict_mean_ci_upp = results_data[:, 4:6].T

        # predict_ci_low, predict_ci_upp = results_data[:, 6:8].T

        xs = np.linspace(min(x), max(x), 100)
        ys = optim_fitter(xs)

        optim_delay_arg = np.argmax(ys)
        optim_delay = xs[optim_delay_arg]
        optim_response = ys[optim_delay_arg]

        self.results_plot = {"x": x,
                             "y": y,
                             "xs": xs,
                             "ys": ys,
                             "x_all": x_all,
                             "y_all": y_all,
                             "predict_mean_ci_low": predict_mean_ci_low,
                             "predict_mean_ci_upp": predict_mean_ci_upp,
                             "optim_delay": optim_delay,
                             "optim_response": optim_response,
                             "error_df": temp3}


    def save_settings(self):
        try:
            # Make sure you have send the data down###
            patient = []
            period = []
            experiment = []
            file = []
            ecg = []
            bpao = []
            bp = []
            bp_prox = []
            bp_dist = []
            boxa = []
            ref = []
            pressure = []
            override_test = []
            transitions = []

            for pt in self.exp_list:
                patient.append(self.patient)
                period.append(self.period)
                experiment.append(pt.exp)
                file.append(os.path.split(pt.daq_fl)[1])
                ecg.append(pt.hints.get('ECG', ''))
                bpao.append(pt.hints.get('BPAO', ''))
                boxa.append(pt.hints.get('BoxA', ''))
                bp.append(pt.hints.get('BP', ''))
                bp_prox.append(pt.hints.get('bp_prox', ''))
                bp_dist.append(pt.hints.get('bp_dist', ''))
                ref.append(pt.ref)
                pressure.append(pt.hints.get('Pressure', ''))
                override_test.append(pt.override_test)
                transitions.append(" ".join(map(str, pt.final_transitions_sample)))

            db = pd.DataFrame(OrderedDict([('Patient', patient),
                                           ('Period', period),
                                           ('Experiment', experiment),
                                           ('File', file),
                                           ('ECG', ecg),
                                           ('BPAO', bpao),
                                           ('BP', bp),
                                           ('bp_prox', bp_prox),
                                           ('bp_dist', bp_dist),
                                           ('BoxA', boxa),
                                           ('Pressure', pressure),
                                           ('Ref', ref),
                                           ('Override_Test', override_test),
                                           ('Transitions', transitions)]))

            now = datetime.datetime.now().isoformat()[0:19].replace(":", "").replace("-", "")

            settings_fn = self.patient + "-" + self.period + "-" + now + "-Haem.csv"
            results_fn = self.patient + "-" + self.period + "-" + now + "-Results.csv"
            summary_fn = self.patient + "-" + self.period + "-" + now + "-Summary.csv"

            settings_fl = os.path.join(self.save_dir, settings_fn)
            results_fl = os.path.join(self.save_dir, results_fn)
            summary_fl = os.path.join(self.save_dir, summary_fn)

            print("Saving Settings: ", settings_fl)
            db.to_csv(settings_fl, index=False)
            print("Saving Results: ", results_fl)
            self.final_results_db.to_csv(results_fl, index=False)
            print("Saving Summary:", summary_fl)

            rp = self.results_plot
            res_sum = pd.DataFrame({'Delay': rp['x_all'], 'Benefit': rp['y_all']})
            res_sum.to_csv(summary_fl, index=False)
        except Exception as e:
            print(e)

class OptAnalysis:

    beat_window = 10

    def __init__(self, source, source_hints):
        print(source)
        print(source_hints)

        source_hints_default = {'patient': 'Unknown',
                                'search_for_files': False}

        hints_default = {'ECG': 'ecg',
                         'BP': 'BP',
                         'BPAO': 'bpao',
                         'Pressure': 'BP',
                         'Period': 'Unknown',
                         'BoxA': 'boxa',
                         'Method': None,
                         'Transitions': ""}

        self.source = source

        self.source_hints = {**source_hints_default, **source_hints}

        self.final_transitions_dir_use = False


        if source == 'database':
            ####Provide####
            self.database_fl = self.source_hints['database_fl']
            self.patient = self.source_hints['patient']

            self.exp = self.source_hints['exp']

            self.daq_dir = self.source_hints['daq_dir']
            self.search_for_files = self.source_hints['search_for_files']

            self.save_dir = self.source_hints['save_dir']
            ####Provide####

            db = pd.read_csv(self.database_fl)

            hints = db[(db.Patient == self.patient) & (db.Experiment == self.exp)].iloc[0]
            self.hints = {**hints_default, **hints}

            self.zip_fn = self.hints['File']
            self.period = self.source['Period']
            self.ecg_source = self.hints['ECG']
            self.pressure_source = self.hints[self.hints['Pressure']]
            self.delay_source = self.hints['BoxA']

        if source == 'dict':
            #########Provide#####
            self.patient = self.source_hints['patient']
            self.exp = self.source_hints['exp']

            self.zip_fn = self.source_hints['zip_fn']
            self.daq_dir = self.source_hints['daq_dir']
            self.search_for_files = self.source_hints['search_for_files']

            self.save_dir = self.source_hints['save_dir']

            hints = self.source_hints.get('hints', {})
            self.hints = {**hints_default, **hints}
            ######Provide########
            self.period = self.hints['Period']
            self.ecg_source = self.hints['ECG']
            self.pressure_source = self.hints[self.hints['Pressure']]
            self.delay_source = self.hints['BoxA']

        if pd.isnull(self.hints['Transitions']) or not self.hints['Transitions']:
            self.hints['Transitions'] = ""

        self.results = []

        self.delay_transitions_sample = []
        self.ecg_transitions_sample = []
        self.hints_transitions_sample = []
        self.pressure_cutoff = 0

        self.daq_exp = DAQ_File(self.daq_dir, self.zip_fn,
                                search_for_files=self.search_for_files)

        self.daq_fl = self.daq_exp.zip_fl

        self.sampling_rate = self.daq_exp.sampling_rate

        self.ecg = DAQContainerECG(data=getattr(self.daq_exp, self.ecg_source), sampling_rate=self.sampling_rate)
        self.ecg.calc_ecg_peaks()
        self.ecg.calc_peaks_correlation()

        self.set_pressure(self.pressure_source)

        self.delay = DAQContainerBoxA(data=getattr(self.daq_exp, self.delay_source, self.daq_exp.blank), sampling_rate=self.sampling_rate)
        self.delay.calc_sample_code()

        try:
            self.ref_idx = self.delay.codebook_default_labels.index(self.hints['Ref'])
            self.ref = self.hints['Ref']
        except Exception as e:
            print(e)
            self.ref = "140ms"
            self.ref_idx = self.delay.codebook_default_labels.index(self.ref)

        try:
            self.override_test_idx = self.delay.codebook_default_labels.index(self.hints['Override_Test'])
            self.override_test = self.hints['Override_Test']
        except Exception as e:
            print(e)
            found_labels = [self.delay.codebook_default_labels[i] for i in self.delay.sample_code_set]
            tmp = [x for x in found_labels if x != self.ref]
            if len(tmp) == 1:
                self.override_test = tmp[0]
                self.override_test_idx = self.delay.codebook_default_labels.index(self.override_test)
            else:
                self.override_test = None
                self.override_test_idx = None

        if self.hints['Transitions']:
            self.calc_final_transitions_hints()
        else:
            self.calc_final_transitions_delay()

        if len(np.unique(self.delay.sample_code)) > 2:
            print("More than two togglebox settings")
        elif len(np.unique(self.delay.sample_code)) < 2:
            print("Only one togglebox setting")

    def set_pressure(self, pressure_source):

        self.pressure_source = pressure_source

        if "bp_prox" in self.pressure_source or "bp_dist" in self.pressure_source:
            pressure_scale = 100/2.5
        else:
            pressure_scale = 100

        if self.pressure_source.upper() == "BP":
            try:
                self.pressure = DAQContainerBPAO(data=getattr(self.daq_exp, "BP"), sampling_rate=self.sampling_rate, scale=pressure_scale)
                self.pressure_source = "BP"
            except Exception as e:
                self.pressure = DAQContainerBPAO(data=getattr(self.daq_exp, "bp"), sampling_rate=self.sampling_rate, scale=pressure_scale)
                self.pressure_source = 'bp'
        else:
            self.pressure = DAQContainerBPAO(data=getattr(self.daq_exp, self.pressure_source), sampling_rate=self.sampling_rate, scale=pressure_scale)

        self.pressure.calc_peaks()

    def calc_final_transitions_hints(self):
        if self.hints['Transitions']:
            self.hints_transitions_sample = list(map(int, self.hints['Transitions'].split(" ")))
            self.final_transitions_sample = self.hints_transitions_sample

        self.final_transitions_dir = []
        for transition in self.final_transitions_sample:
            self.final_transitions_dir.append("AO")

    def calc_final_transitions_delay(self):
        self.delay.calc_transitions()
        self.delay_transitions_sample = list(self.delay.transitions_sample)
        self.final_transitions_sample = self.delay_transitions_sample

        self.final_transitions_dir = []
        for transition in self.final_transitions_sample:
            self.final_transitions_dir.append("AO")

    def calc_final_transitions_ecg(self):

        self.delay.calc_transitions()
        self.ecg.calc_peaks_cluster(target_clusters=2)
        self.ecg.calc_transitions()

        delay_transitions_sample = self.delay.transitions_sample
        ecg_transitions_sample = self.ecg.transitions_sample

        if not self.final_transitions_dir_use:
            final_transitions_sample = []

            search_idx = 0

            for i, transition in enumerate(delay_transitions_sample):

                if search_idx < len(ecg_transitions_sample):
                    temp = mmt.find_nearest_idx(ecg_transitions_sample[search_idx:], transition)
                    candidate_ecg_transition_idx = temp + search_idx
                    final_transitions_sample.append(ecg_transitions_sample[candidate_ecg_transition_idx])
                    search_idx = candidate_ecg_transition_idx + 1
                else:
                    break
        else:
            final_transitions_sample = list(ecg_transitions_sample)
            print(final_transitions_sample)

        self.ecg_transitions_sample = np.array(final_transitions_sample) - 100
        self.final_transitions_sample = list(self.ecg_transitions_sample)

        self.final_transitions_dir = []
        for transition in self.final_transitions_sample:
            if len(self.final_transitions_dir) % 2 == 0:
                self.final_transitions_dir.append("AO")
            else:
                self.final_transitions_dir.append("OA")


    def calc_ecg_delay_code_map(self):
        ecg_match = list(set(self.ecg.good_peaks_code))
        code_match = {}
        for code in ecg_match:
            smpls = self.ecg.good_peaks_sample[self.ecg.good_peaks_code == code]
            code_match[code] = mmt.stats.matt_mode(self.delay.sample_code[smpls])

        self.ecg_peaks_code_map_delay = np.array([code_match.get(x) for x in self.ecg.good_peaks_code])
        self.ecg_peaks_code_delay = np.array([self.delay.codebook.get(x) for x in self.ecg_peaks_code_map_delay])

    def calc_results(self, results_type = "SBP"):

        self.results_type = results_type

        sampling_rate = self.ecg.sampling_rate

        sample_window = 10 * sampling_rate
        sample_blank_window = 3 * sampling_rate
        assert(sample_blank_window < sample_window)

        beat_window = self.beat_window
        results = []
        final_transitions_sample = self.final_transitions_sample
        num_transitions = len(final_transitions_sample)
        max_sample = len(self.pressure.data)
        pressure_peaks_sample = self.pressure.peaks_sample
        pressure_peaks_value = self.pressure.peaks_value

        if not "Laser" in results_type:

            if self.pressure_cutoff:
                print(pressure_peaks_value)
                print(self.pressure_cutoff)
                mask = pressure_peaks_value > self.pressure_cutoff
                print(mask)
                pressure_peaks_sample = pressure_peaks_sample[mask]
                pressure_peaks_value = pressure_peaks_value[mask]

        for i in range(num_transitions):

            # pre_transition_delay_sample_start = None
            # pre_transition_delay_sample_end = None
            # post_transition_delay_sample_start = None
            # post_transition_delay_sample_end = None


            if num_transitions <= 0:
                continue

            elif num_transitions > 0:
                transition_sample = final_transitions_sample[i]
                pre_transition_delay_sample_start = max(0, transition_sample - sample_window)
                pre_transition_delay_sample_end = max(0, transition_sample - sample_blank_window)
                post_transition_delay_sample_start = min(max_sample, transition_sample + sample_blank_window)
                post_transition_delay_sample_end = min(max_sample, transition_sample + sample_window)
                if post_transition_delay_sample_start == post_transition_delay_sample_end:
                    continue


            if num_transitions == 1:
                pre_transition_sample = 0
                post_transition_sample = max_sample

            elif num_transitions > 1:

                if i == 0:
                    pre_transition_sample = 0
                    post_transition_sample = final_transitions_sample[i+1]

                elif i == num_transitions - 1:
                    pre_transition_sample = final_transitions_sample[i-1]
                    post_transition_sample = max_sample

                else:
                    pre_transition_sample = final_transitions_sample[i-1]
                    post_transition_sample = final_transitions_sample[i+1]

            if not self.final_transitions_dir_use:

                pre_code_int = mmt.stats.matt_mode(self.delay.sample_code[pre_transition_delay_sample_start:pre_transition_delay_sample_end])
                post_code_int = mmt.stats.matt_mode(self.delay.sample_code[post_transition_delay_sample_start:post_transition_delay_sample_end])

                pre_code = self.delay.codebook_labels[pre_code_int]
                post_code = self.delay.codebook_labels[post_code_int]

                if pre_code == "ref":
                    direction = "AO"
                    target = post_code
                elif post_code == "ref":
                    direction = "OA"
                    target = pre_code
                else:
                    direction = ""
                    target = ""

            elif self.final_transitions_dir_use:
                if self.final_transitions_dir[i] == 'AO':
                    direction = "AO"
                    pre_code = "ref"
                    post_code = self.override_test
                    target = self.override_test
                elif self.final_transitions_dir[i] == 'OA':
                    direction = "OA"
                    pre_code = self.override_test
                    post_code = "ref"
                    target = self.override_test


            if 'SBP' in results_type:

                pre_pressure_beats = np.where((pressure_peaks_sample > pre_transition_sample) & (pressure_peaks_sample < transition_sample))[0][-beat_window:]
                post_pressure_beats = np.where((pressure_peaks_sample > transition_sample) & (pressure_peaks_sample < post_transition_sample))[0][:beat_window]

                pre_post_pressure_beats = np.concatenate((pre_pressure_beats, post_pressure_beats))
                pre_post_indicators = np.concatenate((np.repeat('Pre', len(pre_pressure_beats)), np.repeat('Post', len(post_pressure_beats))))
                pre_post_codes = np.concatenate((np.repeat(pre_code, len(pre_pressure_beats)), np.repeat(post_code, len(post_pressure_beats))))
                fbeats = np.concatenate((np.arange(-len(pre_pressure_beats), 0), np.arange(0, len(post_pressure_beats))))

                for pre_post_indicator, pre_post_code, fbeat, pressure_beat in zip(pre_post_indicators, pre_post_codes, fbeats, pre_post_pressure_beats):

                    row = OrderedDict()

                    row['Patient'] = self.patient
                    row['Exp'] = self.exp
                    row['File'] = self.daq_fl
                    row['Rep'] = "Rep" + str(i)
                    row['Direction'] = direction
                    row['Target'] = target
                    row['Delay'] = pre_post_code
                    row['Order'] = pre_post_indicator
                    row['Beat'] = pressure_beat
                    row['FBeat'] = fbeat
                    row['SBP'] = pressure_peaks_value[pressure_beat]
                    row['Period'] = self.period

                    results.append(row)

            elif 'Laser' in results_type:

                try:

                    fix_pre_transition_sample = np.max([pre_transition_sample, transition_sample-7000])
                    fix_post_transition_sample = np.min([post_transition_sample, transition_sample+7000])

                    ecg_data_pre = self.ecg.data[fix_pre_transition_sample:transition_sample-1000]
                    laser_data_pre = self.pressure.data[fix_pre_transition_sample:transition_sample-1000]

                    ecg_data_post = self.ecg.data[transition_sample+1000: fix_post_transition_sample]
                    laser_data_post = self.pressure.data[transition_sample+1000: fix_post_transition_sample]

                    if "Magic" in results_type:
                        pre_pressure_beats = calc_magic_laser(ecg_data_pre, laser_data_pre)
                        post_pressure_beats = calc_magic_laser(ecg_data_post, laser_data_post)
                    elif "Mean" in results_type:
                        pre_pressure_beats = np.mean(laser_data_pre)
                        post_pressure_beats = np.mean(laser_data_post)
                    elif "Magic2" in results_type:
                        pre_pressure_beats = calc_magic_laser(ecg_data_pre, laser_data_pre, no_log=True)
                        post_pressure_beats = calc_magic_laser(ecg_data_post, laser_data_post, no_log=True)

                    pre_post_pressure_beats = np.array([pre_pressure_beats, post_pressure_beats])
                    pre_post_indicators = np.concatenate((np.repeat('Pre', 1), np.repeat('Post', 1)))
                    pre_post_codes = np.concatenate((np.repeat(pre_code, 1), np.repeat(post_code, 1)))
                    fbeats = np.array([-1,0])

                    for pre_post_indicator, pre_post_code, fbeat, pressure_beat in zip(pre_post_indicators, pre_post_codes, fbeats, pre_post_pressure_beats):

                        row = OrderedDict()

                        row['Patient'] = self.patient
                        row['Exp'] = self.exp
                        row['File'] = self.daq_fl
                        row['Rep'] = "Rep" + str(i)
                        row['Direction'] = direction
                        row['Target'] = target
                        row['Delay'] = pre_post_code
                        row['Order'] = pre_post_indicator
                        row['Beat'] = pressure_beat
                        row['FBeat'] = fbeat
                        row['SBP'] = pressure_beat
                        row['Period'] = self.period

                        results.append(row)

                except Exception as e:
                    print(e)


        self.results = results


