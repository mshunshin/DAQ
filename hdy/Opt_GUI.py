import os
import datetime

import numpy as np
import pandas as pd

import hdy

from PySide6 import QtGui, QtCore, QtWidgets

import pyqtgraph as pg

class Opt_Selector_GUI(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.init_gui()
        self.show()

        self.source = None
        self.source_hints = {}

    def init_gui(self):

        self.select_db_btn = QtWidgets.QPushButton("Load Database")
        self.select_db_btn.clicked.connect(self.select_db)

        self.select_dir_noninvasive_btn = QtWidgets.QPushButton("Load Directory Non-invasive")
        self.select_dir_noninvasive_btn.clicked.connect(self.select_dir_noninvasive)

        self.select_dir_invasive_btn = QtWidgets.QPushButton("Load Directory Invasive")
        self.select_dir_invasive_btn.clicked.connect(self.select_dir_invasive)

        self.select_dir_bpprox_btn = QtWidgets.QPushButton("Load Directory BP Prox")
        self.select_dir_bpprox_btn.clicked.connect(self.select_dir_bpprox)

        self.select_dir_bpdist_btn = QtWidgets.QPushButton("Load Directory BP Dist")
        self.select_dir_bpdist_btn.clicked.connect(self.select_dir_bpdist)

        self.select_layout = QtWidgets.QVBoxLayout()
        self.select_layout.addWidget(self.select_db_btn)
        self.select_layout.addWidget(self.select_dir_noninvasive_btn)
        self.select_layout.addWidget(self.select_dir_invasive_btn)
        self.select_layout.addWidget(self.select_dir_bpprox_btn)
        self.select_layout.addWidget(self.select_dir_bpdist_btn)

        self.setLayout(self.select_layout)


    def select_db(self):
        self.database_fl = str(QtWidgets.QFileDialog.getOpenFileName(self, "Select Optimisation Database")[0])
        self.source = 'database'

        self.source_hints['database_fl'] = self.database_fl
        database_dir = os.path.dirname(self.database_fl)
        self.source_hints['save_dir'] = database_dir
        patient = pd.read_csv(self.database_fl).Patient.iloc[0]
        self.source_hints['patient'] = patient
        self.source_hints['daq_dir'] = database_dir

        print(self.database_fl)
        self.close()

    def select_dir_noninvasive(self):
        self.directory = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory - Non-invasive"))
        self.source = 'directory-noninvasive'
        self.source_hints['daq_dir'] = self.directory
        self.source_hints['save_dir'] = self.directory
        print(self.directory)
        self.close()

    def select_dir_invasive(self):
        self.directory = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory - Invasive"))
        self.source = 'directory-invasive'
        self.source_hints['daq_dir'] = self.directory
        self.source_hints['save_dir'] = self.directory
        print(self.directory)
        self.close()


    def select_dir_bpprox(self):
        self.directory = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory - BP Prox"))
        self.source = 'directory-bpprox'
        self.source_hints['daq_dir'] = self.directory
        self.source_hints['save_dir'] = self.directory
        print(self.directory)
        self.close()


    def select_dir_bpdist(self):
        self.directory = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory - BP Dist"))
        self.source = 'directory-bpdist'
        self.source_hints['daq_dir'] = self.directory
        self.source_hints['save_dir'] = self.directory
        print(self.directory)
        self.close()


class TimeAxisItem(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tickStrings(self, values, scale, spacing):
        if spacing > 1000:
            result = [self.ms_to_mmss(value) for value in values]
        else:
            result = [self.ms_to_mmssmm(value) for value in values]
        return result

    def ms_to_mmss(self, value):
        sec = int(value/1000)%60
        min = int(value/(1000*60))
        return "{min}:{sec:02d}".format(min=min, sec=sec)

    def ms_to_mmssmm(self, value):
        ms = int(value)%1000
        sec = int(value/1000)%60
        min = int(value/(1000*60))
        return "{min}:{sec:02d}.{ms:03d}".format(min=min, sec=sec, ms=ms)



class DelayAxisItem(pg.AxisItem):
    def __init__(self, codebook, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.codebook = codebook

    def tickStrings(self, values, scale, spacing):
        return [self.codebook[int(value)] for value in values]


class Opt_GUI(QtWidgets.QWidget):

    def __init__(self, opt_collection: hdy.OptCollection, parent=None):

        super().__init__(parent=parent)

        self.exp_collection = opt_collection

        self.exp_list = self.exp_collection.exp_list
        self.exp_title_list = self.exp_collection.exp_title_list
        self.exp_widget_list = []

        self.init_gui()

        for exp, title in zip(self.exp_list,
                              self.exp_title_list):

            exp_widget = OptimWidget(exp)
            self.switcher.addTab(exp_widget, title)
            self.exp_widget_list.append(exp_widget)

        self.init_results_gui()

    def init_gui(self):
        pg.setConfigOptions(antialias=True)
        self.main_layout = QtWidgets.QHBoxLayout()
        self.btn_layout = QtWidgets.QVBoxLayout()
        self.plot_layout = QtWidgets.QVBoxLayout()

        self.calc_btn = QtWidgets.QPushButton("Calc Results")
        self.calc_btn.clicked.connect(self.calc_results)

        self.save_settings_btn = QtWidgets.QPushButton("Save Results and Settings")
        self.save_settings_btn.clicked.connect(self.save_settings)

        self.screenshot_btn = QtWidgets.QPushButton("Take Screenshot")
        self.screenshot_btn.clicked.connect(self.screenshot)

        self.patient_lbl = QtWidgets.QLabel("Patient:")
        self.patient_txt = QtWidgets.QLineEdit(self.exp_collection.patient)

        self.period_lbl = QtWidgets.QLabel("Period")
        self.period_txt = QtWidgets.QLineEdit(self.exp_collection.period)

        self.fusion_lbl = QtWidgets.QLabel("Fusion")
        self.fusion_cb = QtWidgets.QComboBox()

        fusions = list(range(20,360, 20))
        fusions.append(None)

        for value in fusions:
            self.fusion_cb.addItem(str(value), value)

        fusion_idx = self.fusion_cb.findData(None)
        if fusion_idx != -1:
            self.fusion_cb.setCurrentIndex(fusion_idx)

        self.type_lbl = QtWidgets.QLabel("Optimisation Type")
        self.type_cb = QtWidgets.QComboBox()

        self.type_cb.addItem("SBP", "SBP")
        self.type_cb.addItem("Laser Mean", "Laser_Mean")
        self.type_cb.addItem("Laser Magic", "Laser_Magic")
        self.type_cb.addItem("Laser Magic 2", "Laser_Magic2")

        self.optim_delay_lbl = QtWidgets.QLabel("")
        self.optim_response_lbl = QtWidgets.QLabel("")

        self.matt_lbl = QtWidgets.QLabel()
        self.matt_lbl.setText("Dr. Matthew Shun-Shin\nBHF Clinical Fellow\nImperial College London")

        self.btn_layout.addWidget(self.calc_btn)
        self.btn_layout.addWidget(self.save_settings_btn)
        self.btn_layout.addWidget(self.screenshot_btn)
        self.btn_layout.addStretch(1)
        self.btn_layout.addWidget(self.patient_lbl)
        self.btn_layout.addWidget(self.patient_txt)
        self.btn_layout.addWidget(self.period_lbl)
        self.btn_layout.addWidget(self.period_txt)
        self.btn_layout.addStretch(1)
        self.btn_layout.addWidget(self.fusion_lbl)
        self.btn_layout.addWidget(self.fusion_cb)
        self.btn_layout.addWidget(self.type_lbl)
        self.btn_layout.addWidget(self.type_cb)
        self.btn_layout.addStretch(1)
        self.btn_layout.addWidget(self.optim_delay_lbl)
        self.btn_layout.addWidget(self.optim_response_lbl)
        self.btn_layout.addStretch(1)
        self.btn_layout.addWidget(self.matt_lbl)
        self.btn_layout.addStretch(1)

        self.switcher = QtWidgets.QTabWidget()
        self.plot_layout.addWidget(self.switcher)

        self.main_layout.addLayout(self.btn_layout, 0)
        self.main_layout.addLayout(self.plot_layout, 1)

        self.setLayout(self.main_layout)
        self.show()

    def init_results_gui(self):
        self.results_pw = pg.PlotWidget()
        self.results_pi = self.results_pw.getPlotItem()

        self.switcher.addTab(self.results_pw, 'Results')

    def widget_to_exp(self):

        print("Updating Settings")

        patient = self.patient_txt.text()
        period = self.period_txt.text()

        self.exp_collection.patient = patient
        self.exp_collection.period = period

        fusion_index = self.fusion_cb.currentIndex()
        fusion = self.fusion_cb.itemData(fusion_index)

        self.exp_collection.fusion = fusion

        for exp, exp_widget in zip(self.exp_list, self.exp_widget_list):

            exp.patient = patient
            exp.period = period

            transitions = []

            for transition in exp_widget.transitions_lst:
                transitions.append(int(transition.value()))

            exp.final_transitions_sample = np.sort(np.array(transitions))

            exp.override_test_idx = exp_widget.test_delay_cb.itemData(exp_widget.test_delay_cb.currentIndex())
            exp.ref_idx = exp_widget.ref_delay_cb.itemData(exp_widget.ref_delay_cb.currentIndex())

            if exp.override_test_idx is not None:
                exp.override_test = exp.delay.codebook_default_labels[exp.override_test_idx]
            else:
                exp.override_test = None

            if exp.ref_idx is not None:
                exp.ref = exp.delay.codebook_default_labels[exp.ref_idx]
            else:
                exp.ref = None

    def calc_results(self):
        print("Calculating Results")

        try:
            self.widget_to_exp()

            self.exp_collection.calc_results(results_type = self.type_cb.currentData())

            rp = self.exp_collection.results_plot

            self.results_pi.clear()

            zero_line_pen = pg.mkPen('FFFFFF', width=2)
            zero_line = pg.InfiniteLine(pos=0, angle=0, pen=zero_line_pen, movable=False)
            self.results_pi.addItem(zero_line)

            err_df = rp['error_df']
            print(err_df)
            error_pen = pg.mkPen('w', width=2)
            error_bars = pg.ErrorBarItem(x=err_df['Target_num'],
                                         y=err_df['benefit_mean'],
                                         top=err_df['benefit_upper'],
                                         bottom=err_df['benefit_lower'],
                                         pen=error_pen,
                                         beam=5)

            self.results_pi.addItem(error_bars)

            self.results_plt = self.results_pi.scatterPlot()
            self.results_plt.setData(x=rp['x_all'], y=rp['y_all'], symbol='o', symbolBrush='3333FF', symbolPen=None, pen=None)

            self.results_mean_plt = self.results_pi.scatterPlot()
            self.results_mean_plt.setData(x=err_df['Target_num'],
                                          y=err_df['benefit_mean'],
                                          symbol='o',
                                          symbolBrush='33FF33',
                                          symbolPen=None,
                                          pen=None)

            optim_pen = pg.mkPen("33FF33", width=2)
            self.optim_plt = self.results_pi.plot()
            self.optim_plt.setData(x=rp['xs'], y=rp['ys'], pen=optim_pen, symbol=None)

            self.optim_upper_plt = self.results_pi.plot()
            self.optim_upper_plt.setData(x=rp['x'], y=rp['predict_mean_ci_upp'], pen="FF444488", symbol=None)

            self.optim_lower_plt = self.results_pi.plot()
            self.optim_lower_plt.setData(x=rp['x'], y=rp['predict_mean_ci_low'], pen="FF444488", symbol=None)

            f = pg.FillBetweenItem(self.optim_upper_plt, self.optim_lower_plt, "FF444488")
            self.results_pi.addItem(f)

            self.optim_delay_lbl.setText("Optimum Delay: {0:.0f}ms".format(rp['optim_delay']))
            self.optim_response_lbl.setText("Response at Optimum: {0:.0f}mmHg".format(rp['optim_response']))
        except Exception as e:
            print(e)

    def screenshot(self):
        print("Saving screenshot")

        try:
            img = self.grab()
            save_dir = self.exp_collection.save_dir
            now = datetime.datetime.now().isoformat()[0:19].replace(":", "").replace("-", "")
            screenshot_fn = self.patient_txt.text() + "-" + self.period_txt.text() + "-" + now + ".png"
            screenshot_fl = os.path.join(save_dir, screenshot_fn)
            os.makedirs(save_dir, exist_ok=True)
            img.save(screenshot_fl, "png")
            print("Screenshot saved: {0}".format(screenshot_fl))
        except Exception as e:
            print(e)
            print("Failed to save screenshot")

    def save_settings(self):
        print("Updating experiment")
        #self.widget_to_exp()
        #print("Calculating results")
        #self.exp_collection.calc_results()
        print("Saving settings and results")
        self.exp_collection.save_settings()
        print("Done")



class OptimWidget(QtWidgets.QWidget):

    def __init__(self, optim_exp: hdy.OptAnalysis, parent=None):

        super().__init__(parent=parent)

        self.transitions_lst = []
        self.last_clicked_transition = None

        self.transitions_type = []
        self.transitions_type_lst = []

        self.last_clicked_transition = None

        self.optim_exp = optim_exp
        self.setup_gui()
        self.setup_plots()
        self.setup_transitions()
        self.update_data()
        self.show()
        self.setup_toggles()

    def setup_gui(self):

        pg.setConfigOptions(antialias=True)

        self.plot_layout = QtWidgets.QVBoxLayout()
        self.btn_layout = QtWidgets.QVBoxLayout()
        self.main_layout = QtWidgets.QHBoxLayout()

        self.bg_toggle = QtWidgets.QCheckBox("White BG")
        self.bg_toggle.stateChanged.connect(self.bg_toggle_changed)

        self.pressure_source_lbl = QtWidgets.QLabel("Pressure Source")
        self.pressure_source_cb = QtWidgets.QComboBox()

        for source in self.optim_exp.daq_exp.sources:
            self.pressure_source_cb.addItem(str(source), str(source))

        self.pressure_source_cb.setCurrentIndex(self.optim_exp.daq_exp.sources.index(self.optim_exp.pressure_source))
        self.pressure_source_cb.currentIndexChanged.connect(self.pressure_source_changed)

        self.remove_transition_btn = QtWidgets.QPushButton("Remove Transition")
        self.remove_transition_btn.clicked.connect(self.remove_transition)

        self.add_transition_btn = QtWidgets.QPushButton("Add Transition")
        self.add_transition_btn.clicked.connect(self.add_transition)

        self.toggle_transitions_dir_use_btn = QtWidgets.QPushButton("Toggle Manual Direction")
        self.toggle_transitions_dir_use_btn.clicked.connect(self.toggle_transitions_dir_use)

        self.invert_transition_btn = QtWidgets.QPushButton("Invert Transition")
        self.invert_transition_btn.clicked.connect(self.invert_transition)

        self.set_transitions_hints_btn = QtWidgets.QPushButton("Set Transitions from Database")
        self.set_transitions_hints_btn.clicked.connect(self.set_transitions_hints)

        self.set_transitions_delay_btn = QtWidgets.QPushButton("Set Transitions from Box")
        self.set_transitions_delay_btn.clicked.connect(self.set_transitions_delay)

        self.set_transitions_ecg_btn = QtWidgets.QPushButton("Set Transitions from ECG")
        self.set_transitions_ecg_btn.clicked.connect(self.set_transitions_ecg)

        self.toggle_dpdt_btn = QtWidgets.QPushButton("Toggle dP/Dt")
        self.toggle_dpdt_btn.clicked.connect(self.toggle_dpdt)

        self.ref_delay_lbl = QtWidgets.QLabel("Ref Delay Setting")
        self.ref_delay_cb = QtWidgets.QComboBox()

        for i, value in enumerate(self.optim_exp.delay.codebook_default_labels):
            self.ref_delay_cb.addItem(value, i)
        self.ref_delay_cb.addItem("No Reference", None)

        if self.optim_exp.ref_idx is not None:
            self.ref_delay_cb.setCurrentIndex(self.optim_exp.ref_idx)
        else:
            self.ref_delay_cb.setCurrentIndex(i+1)

        self.test_delay_lbl = QtWidgets.QLabel("Test Delay Override")
        self.test_delay_cb = QtWidgets.QComboBox()

        for i, value in enumerate(self.optim_exp.delay.codebook_default_labels):
            self.test_delay_cb.addItem(value, i)
        self.test_delay_cb.addItem("Use Box", None)

        if self.optim_exp.override_test_idx is not None:
            self.test_delay_cb.setCurrentIndex(self.optim_exp.override_test_idx)
        else:
            self.test_delay_cb.setCurrentIndex(i+1)

        self.btn_layout.addWidget(self.bg_toggle)
        self.btn_layout.addWidget(self.remove_transition_btn)
        self.btn_layout.addWidget(self.add_transition_btn)
        self.btn_layout.addStretch(1)
        self.btn_layout.addWidget(self.toggle_transitions_dir_use_btn)
        self.btn_layout.addWidget(self.invert_transition_btn)
        self.btn_layout.addStretch(1)
        self.btn_layout.addWidget(self.set_transitions_hints_btn)
        self.btn_layout.addWidget(self.set_transitions_delay_btn)
        self.btn_layout.addWidget(self.set_transitions_ecg_btn)
        self.btn_layout.addStretch(1)
        self.btn_layout.addWidget(self.pressure_source_lbl)
        self.btn_layout.addWidget(self.pressure_source_cb)
        self.btn_layout.addWidget(self.toggle_dpdt_btn)
        self.btn_layout.addStretch(1)
        self.btn_layout.addWidget(self.ref_delay_lbl)
        self.btn_layout.addWidget(self.ref_delay_cb)
        self.btn_layout.addWidget(self.test_delay_lbl)
        self.btn_layout.addWidget(self.test_delay_cb)
        self.btn_layout.addStretch(2)

        self.ecg_pw = pg.PlotWidget(axisItems={'bottom':TimeAxisItem(orientation='bottom')})
        self.ecg_pi = self.ecg_pw.getPlotItem()
        self.ecg_pi.setLabel(axis='left',text="ECG")
        self.ecg_plt = self.ecg_pi.plot()
        self.ecg_peak_plt = self.ecg_pi.plot()

        self.pressure_pw = pg.PlotWidget(axisItems={'bottom':TimeAxisItem(orientation='bottom')})
        self.pressure_pi = self.pressure_pw.getPlotItem()
        self.pressure_pi.setLabel(axis='left',text="BP")
        self.pressure_plt = self.pressure_pi.plot()
        self.pressure_peak_plt = self.pressure_pi.plot()

        self.boxa_pw = pg.PlotWidget(axisItems={'bottom':TimeAxisItem(orientation='bottom'),
                                                'left':DelayAxisItem(orientation='left', codebook=self.optim_exp.delay.codebook_default_labels)})
        self.boxa_pi = self.boxa_pw.getPlotItem()
        self.boxa_pi.setLabel(axis='left',text="BoxA")
        self.boxa_plt = self.boxa_pi.plot()

        self.overview_pw = pg.PlotWidget(axisItems={'bottom':TimeAxisItem(orientation='bottom')})
        self.overview_pi = self.overview_pw.getPlotItem()
        self.overview_pi.setLabel(axis='left',text="Overview")
        self.overview_plt = self.overview_pi.plot()

        self.zipfl_lbl = QtWidgets.QLabel()

        self.plot_layout.addWidget(self.ecg_pw, stretch=1)
        self.plot_layout.addWidget(self.pressure_pw, stretch=1)
        self.plot_layout.addWidget(self.boxa_pw, stretch=1)
        self.plot_layout.addWidget(self.overview_pw, stretch=1)
        self.plot_layout.addWidget(self.zipfl_lbl)
        self.zipfl_lbl.setText(str(self.optim_exp.daq_exp.zip_fl))
        self.plot_layout.setSpacing(0)
        self.plot_layout.setContentsMargins(0,0,0,0)

        self.main_layout.addLayout(self.btn_layout)
        self.main_layout.addLayout(self.plot_layout)

        self.setLayout(self.main_layout)

    def setup_toggles(self):
        self.bg_toggle.setChecked(True)
        #self.bg_toggle.setChecked(False)

    def setup_plots(self):
        self.overview_region = pg.LinearRegionItem()
        self.overview_region.setZValue(10)
        self.overview_pi.addItem(self.overview_region)
        self.overview_region.sigRegionChanged.connect(self.overview_region_changed)
        self.overview_region.setRegion((0,self.optim_exp.ecg.data.shape[0]))

        self.ecg_pw.sigRangeChanged.connect(self.overview_region_update)
        self.ecg_pw.setYRange(-1,1)

        self.pressure_pw.sigRangeChanged.connect(self.overview_region_update)


        self.pressure_cutoff_il = pg.InfiniteLine(pos=self.optim_exp.pressure_cutoff, angle=0, pen='#0000FF', movable=True)
        self.pressure_pw.addItem(self.pressure_cutoff_il)
        self.pressure_cutoff_il.sigPositionChangeFinished.connect(self.clicked_pressure_cutoff_il)

        self.boxa_pw.sigRangeChanged.connect(self.overview_region_update)

        self.ecg_pi.getAxis('left').setWidth(w=60)
        self.ecg_pi.getAxis('left').setStyle(showValues=False)
        self.pressure_pi.getAxis('left').setWidth(w=60)
        self.boxa_pi.getAxis('left').setWidth(w=60)
        self.overview_pi.getAxis('left').setWidth(w=60)
        self.overview_pi.getAxis('left').setStyle(showValues=False)


    def setup_transitions(self):

        for transition_il in self.transitions_lst:
            self.ecg_pi.removeItem(transition_il)

        for text in self.transitions_type_lst:
            self.ecg_pi.removeItem(text)

        self.transitions_lst = []
        self.transitions_type_lst = []

        for transition, transition_dir in zip(self.optim_exp.final_transitions_sample, self.optim_exp.final_transitions_dir):
            transition_il = pg.InfiniteLine(pos=transition, angle=90, pen='#0FA00F', movable=True)
            self.ecg_pi.addItem(transition_il)
            self.transitions_lst.append(transition_il)
            transition_il.sigPositionChangeFinished.connect(self.clicked_transition)

            if self.optim_exp.final_transitions_dir_use:
                if transition_dir == 'AO':
                    transition_text = pg.TextItem(html='<div style="text-align:center; color:#000;"><pre>   ON<br><br>OFF   </pre></div>', color="000", anchor=(0.5,0.5))
                else:
                    transition_text = pg.TextItem(html='<div style="text-align:center; color:#000;"><pre>ON   <br><br>   OFF</pre></div>', color="000", anchor=(0.5,0.5))

                self.ecg_pi.addItem(transition_text)
                transition_text.setPos(transition, 0)
                self.transitions_type_lst.append(transition_text)

        if self.last_clicked_transition:
            obj = self.transitions_lst[self.last_clicked_transition]
            obj.setPen('b')

    def update_data(self):
        #There is a performance bug in QT where pen widths that are not 0 are really slow.
        ecg_pen = pg.mkPen('#0FA00F', width=1)
        pressure_pen = pg.mkPen('r', width=1)
        boxa_pen = pg.mkPen('#732F9B', width=1)

        self.ecg_plt.setData(self.optim_exp.ecg.data,
                     pen=ecg_pen, symbol=None,
                     antialise=True, autoDownsample=True, downsampleMethod='peak', clipToView=True)

        #self.ecg_peak_plt.setData(x=self.optim_exp.ecg.good_peaks_sample, y=self.optim_exp.ecg.good_peaks_value, symbol='o', pen=None)

        self.overview_plt.setData(self.optim_exp.ecg.data,
                          pen=ecg_pen, symbol=None,
                          antialise=True, autoDownsample=True, clipToView=True)

        self.pressure_plt.setData(self.optim_exp.pressure.data, pen=pressure_pen, antialise=True, autoDownsample=True, clipToView=True)
        self.pressure_peak_plt.setData(x=self.optim_exp.pressure.peaks_sample, y=self.optim_exp.pressure.peaks_value, symbol='o', pen=None)

        self.boxa_plt.setData(self.optim_exp.delay.sample_code, pen=boxa_pen, antialise=True, autoDownsample=True, clipToView=True)


    def overview_region_changed(self):
        self.overview_region.setZValue(10)
        minX, maxX = self.overview_region.getRegion()

        self.ecg_pw.plotItem.setXRange(minX, maxX, padding=0)
        self.pressure_pw.plotItem.setXRange(minX, maxX, padding=0)
        self.boxa_pw.plotItem.setXRange(minX, maxX, padding=0)

    def overview_region_update(self, window, viewRange):
        rgn = viewRange[0]
        self.overview_region.setRegion(rgn)


    def bg_toggle_changed(self, state):

        p = self.palette()

        if state == QtCore.Qt.Checked:
            p.setColor(self.backgroundRole(), QtCore.Qt.white)
            p.setColor(self.foregroundRole(), QtCore.Qt.black)
            self.ecg_pw.setBackground("w")
            self.pressure_pw.setBackground("w")
            self.boxa_pw.setBackground("w")
            self.overview_pw.setBackground("w")
        else:
            p.setColor(self.backgroundRole(), QtCore.Qt.black)
            p.setColor(self.foregroundRole(), QtCore.Qt.white)
            self.ecg_pw.setBackground("k")
            self.pressure_pw.setBackground("k")
            self.boxa_pw.setBackground("k")
            self.overview_pw.setBackground("k")

        self.setPalette(p)

    def clicked_transition(self, obj):
        idx = np.argmax([x == obj for x in self.transitions_lst])

        self.last_clicked_transition = idx
        self.optim_exp.final_transitions_sample[idx] = int(obj.getPos()[0])
        self.setup_transitions()

    def clicked_pressure_cutoff_il(self):
        self.optim_exp.pressure_cutoff = int(self.pressure_cutoff_il.getPos()[1])

    def remove_transition(self):
        try:
            if self.last_clicked_transition is not None:
                idx = self.last_clicked_transition
                print(idx)
                print(self.optim_exp.final_transitions_sample)
                del self.optim_exp.final_transitions_sample[idx]
                del self.optim_exp.final_transitions_dir[idx]
                print(self.optim_exp.final_transitions_sample)
        except Exception as e:
            print(e)

        self.last_clicked_transition = None
        self.setup_transitions()


    def set_transitions_delay(self):
        self.optim_exp.calc_final_transitions_delay()
        self.optim_exp.final_transitions_dir_use = False
        self.last_clicked_transition = None
        self.setup_transitions()

    def set_transitions_ecg(self):
        self.optim_exp.calc_final_transitions_ecg()
        self.optim_exp.final_transitions_dir_use = False
        self.last_clicked_transition = None
        self.setup_transitions()

    def set_transitions_hints(self):
        self.optim_exp.calc_final_transitions_hints()
        self.optim_exp.final_transitions_dir_use = False
        self.last_clicked_transition = None
        self.setup_transitions()

    def pressure_source_changed(self):
        new_pressure_source = self.pressure_source_cb.currentData()
        self.optim_exp.set_pressure(new_pressure_source)
        self.update_data()
        self.pressure_pw.autoRange()

    def toggle_dpdt(self):
        if self.optim_exp.pressure.data_type == 'clean':
            self.optim_exp.pressure.set_dpdt()
            self.optim_exp.pressure.calc_peaks(dpdt=True)
        else:
            self.optim_exp.pressure.set_clean()
            self.optim_exp.pressure.calc_peaks(dpdt=False)
        self.update_data()
        self.pressure_pw.autoRange()

    def toggle_transitions_dir_use(self):
        self.optim_exp.final_transitions_dir_use = not self.optim_exp.final_transitions_dir_use
        self.setup_transitions()

    def add_transition(self):
        vr = self.pressure_pi.getViewBox().viewRange()
        xpos = (vr[0][0] + vr[0][1])/2

        self.optim_exp.final_transitions_sample = list(self.optim_exp.final_transitions_sample)

        self.optim_exp.final_transitions_sample.append(int(xpos))
        self.optim_exp.final_transitions_dir.append("AO")

        self.setup_transitions()
        self.last_clicked_transition = None

    def invert_transition(self):
        try:
            if self.last_clicked_transition is not None:
                idx = self.last_clicked_transition
                if self.optim_exp.final_transitions_dir[idx] == 'AO':
                    self.optim_exp.final_transitions_dir[idx] = 'OA'
                else:
                    self.optim_exp.final_transitions_dir[idx] = 'AO'
        except Exception as e:
            print(e)

        self.setup_transitions()


