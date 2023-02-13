#LPM GUI FOR AAD

import os
import datetime

import numpy as np

import scipy
import scipy.signal
import scipy.fftpack
import scipy.interpolate
os.environ['PYQTGRAPH_QT_LIB'] = 'PySide6'

import mmt

from . import *
import hdy.LaserClasses as lc
from PySide6 import QtCore, QtWidgets, QtGui
#from PyQt5 import QtCore, QtWidgets, QtGui

import pyqtgraph as pg


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


class MagiQuantLaser(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        self.initUI()


    def initUI(self):
        self.statusBar().showMessage('Ready')

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')

        openFile = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open new File')
        openFile.triggered.connect(self.showFileOpen)

        fileMenu.addAction(openFile)

        menubar.addMenu(fileMenu)

        self.lasergui = LaserGui()
        self.setCentralWidget(self.lasergui)
        self.setWindowTitle('MagiQuant Laser')

        self.show()


    def showFileOpen(self):
        file_dir = QtWidgets.QFileDialog.getExistingDirectory(self, 'Open file', '/Users')
        self.mattcon_widget.file_selector.load_dir(file_dir)


class ADAGui(QtWidgets.QWidget):

    def __init__(self, laser_exp: object, parent: object = None) -> object:

        super().__init__(parent=parent)

        self.laser_exp = laser_exp
        self.setup_gui()
        self.setup_toggles()
        self.setup_plots()
        self.update_data()
        self.resize(1200, 800)
        self.show()

    def setup_gui(self):

        pg.setConfigOptions(antialias=True)

        self.setWindowTitle('AAD Research View')
        self.setWindowIcon(QtGui.QIcon('web.png'))

        self.plot_layout = QtWidgets.QVBoxLayout()
        self.btn_layout = QtWidgets.QVBoxLayout()
        self.minibtn_layout = QtWidgets.QVBoxLayout()
        self.turnbtn_layout = QtWidgets.QHBoxLayout()
        self.main_layout = QtWidgets.QHBoxLayout()

        self.bg_toggle = QtWidgets.QCheckBox("White BG")
        self.bg_toggle.stateChanged.connect(self.bg_toggle_changed)

        self.laser1_toggle = QtWidgets.QCheckBox("View Laser1")
        self.laser1_toggle.stateChanged.connect(self.laser1_toggle_changed)

        self.laser2_toggle = QtWidgets.QCheckBox("View Laser2")
        self.laser2_toggle.stateChanged.connect(self.laser2_toggle_changed)

        self.ecg3_toggle = QtWidgets.QCheckBox("View 3 lead ECG")
        self.ecg3_toggle.stateChanged.connect(self.ecg3_toggle_changed)

        self.rvshock_toggle = QtWidgets.QCheckBox("View RV Shock")
        self.rvshock_toggle.stateChanged.connect(self.rvshock_toggle_changed)

        self.rvbip_toggle = QtWidgets.QCheckBox("View RV Bipolar")
        self.rvbip_toggle.stateChanged.connect(self.rvbip_toggle_changed)

        self.lvlead_toggle = QtWidgets.QCheckBox("View LV Lead")
        self.lvlead_toggle.stateChanged.connect(self.lvlead_toggle_changed)

        self.ralead_toggle = QtWidgets.QCheckBox("View RA")
        self.ralead_toggle.stateChanged.connect(self.ralead_toggle_changed)

        self.calc_move_btn = QtWidgets.QPushButton("Move Calc ROI")
        self.calc_move_btn.clicked.connect(self.calc_region_move)

        self.calc_6s_btn = QtWidgets.QPushButton("Set 6s")
        self.calc_6s_btn.clicked.connect(self.calc_6s)

        self.calc_btn = QtWidgets.QPushButton("Calc")
        self.calc_btn.clicked.connect(self.calc_results)

        self.screenshot_btn = QtWidgets.QPushButton("Take Screenshot")
        self.screenshot_btn.clicked.connect(self.screenshot)

        self.hrv_toggle = QtWidgets.QCheckBox("Show RR Interval")
        self.hrv_toggle.stateChanged.connect(self.hrv_toggle_changed)

        self.laser_combo = QtWidgets.QComboBox()
        self.laser_combo.addItems("Raw Filtered Log-Filtered".split())
        self.laser_combo.currentIndexChanged.connect(self.update_data)

        self.begin_lbl = QtWidgets.QLabel()
        self.end_lbl = QtWidgets.QLabel()
        self.laser1_value = QtWidgets.QLabel()
        self.laser2_value = QtWidgets.QLabel()
        self.hr_value = QtWidgets.QLabel()
        self.rr_value = QtWidgets.QLabel()
        self.sbp_value = QtWidgets.QLabel()
        self.map_value = QtWidgets.QLabel()
        self.matt_lbl = QtWidgets.QLabel()

        self.btn_layout.addWidget(self.bg_toggle)
        self.btn_layout.addWidget(self.laser1_toggle)
        self.btn_layout.addWidget(self.laser2_toggle)
        self.btn_layout.addWidget(self.ecg3_toggle)
        self.btn_layout.addWidget(self.rvshock_toggle)
        self.btn_layout.addWidget(self.rvbip_toggle)
        self.btn_layout.addWidget(self.lvlead_toggle)
        self.btn_layout.addWidget(self.ralead_toggle)
        self.btn_layout.addWidget(self.hrv_toggle)
        self.btn_layout.addWidget(self.laser_combo)
        self.btn_layout.addWidget(self.calc_move_btn)
        self.btn_layout.addWidget(self.calc_6s_btn)
        self.btn_layout.addWidget(self.calc_btn)
        self.btn_layout.addWidget(self.screenshot_btn)
        self.btn_layout.addWidget(self.begin_lbl)
        self.btn_layout.addWidget(self.end_lbl)
        self.btn_layout.addWidget(self.laser1_value)
        self.btn_layout.addWidget(self.laser2_value)
        self.btn_layout.addWidget(self.rr_value)
        self.btn_layout.addWidget(self.hr_value)
        self.btn_layout.addWidget(self.sbp_value)
        self.btn_layout.addWidget(self.map_value)
        self.turnbtn_layout.addWidget(self.matt_lbl)

        self.ecg_pw = pg.PlotWidget(axisItems={'bottom':TimeAxisItem(orientation='bottom')})
        self.ecg_pi = self.ecg_pw.getPlotItem()
        self.ecg_pi.setLabel(axis='left',text="ECG")
        self.ecg_plt = self.ecg_pi.plot()
        self.ecg_peak_plt = self.ecg_pi.plot()

        self.pressure_pw = pg.PlotWidget(axisItems={'bottom':TimeAxisItem(orientation='bottom')})
        self.pressure_pi = self.pressure_pw.getPlotItem()
        self.pressure_pi.setLabel(axis='left',text="BP")
        self.pressure_plt = self.pressure_pi.plot()

        self.laser1_pw = pg.PlotWidget(axisItems={'bottom':TimeAxisItem(orientation='bottom')})
        self.laser1_pi = self.laser1_pw.getPlotItem()
        self.laser1_pi.setLabel(axis='left',text="LDPM1")
        self.laser1_plt = self.laser1_pi.plot()
        self.envelope1_plt = self.laser1_pi.plot()

        self.laser2_pw = pg.PlotWidget(axisItems={'bottom':TimeAxisItem(orientation='bottom')})
        self.laser2_pi = self.laser2_pw.getPlotItem()
        self.laser2_pi.setLabel(axis='left',text="LDPM2")
        self.laser2_plt = self.laser2_pi.plot()
        self.envelope2_plt = self.laser2_pi.plot()

        self.ecg3_pw = pg.PlotWidget(axisItems={'bottom':TimeAxisItem(orientation='bottom')})
        self.ecg3_pi = self.ecg3_pw.getPlotItem()
        self.ecg3_pi.setLabel(axis='left',text="3-Lead ECG")
        self.ecg3_plt = self.ecg3_pi.plot()

        self.ralead_pw = pg.PlotWidget(axisItems={'bottom':TimeAxisItem(orientation='bottom')})
        self.ralead_pi = self.ralead_pw.getPlotItem()
        self.ralead_pi.setLabel(axis='left',text="RA")
        self.ralead_plt = self.ralead_pi.plot()

        self.rvshock_pw = pg.PlotWidget(axisItems={'bottom':TimeAxisItem(orientation='bottom')})
        self.rvshock_pi = self.rvshock_pw.getPlotItem()
        self.rvshock_pi.setLabel(axis='left',text="RV Shock")
        self.rvshock_plt = self.rvshock_pi.plot()

        self.rvbip_pw = pg.PlotWidget(axisItems={'bottom':TimeAxisItem(orientation='bottom')})
        self.rvbip_pi = self.rvbip_pw.getPlotItem()
        self.rvbip_pi.setLabel(axis='left',text="RV Bip")
        self.rvbip_plt = self.rvbip_pi.plot()

        self.lvlead_pw = pg.PlotWidget(axisItems={'bottom':TimeAxisItem(orientation='bottom')})
        self.lvlead_pi = self.lvlead_pw.getPlotItem()
        self.lvlead_pi.setLabel(axis='left',text="LV")
        self.lvlead_plt = self.lvlead_pi.plot()

        self.overview_pw = pg.PlotWidget(axisItems={'bottom':TimeAxisItem(orientation='bottom')})
        self.overview_pi = self.overview_pw.getPlotItem()
        self.overview_pi.setLabel(axis='left',text="Overview")
        self.overview_plt = self.overview_pi.plot()

        self.zipfl_lbl = QtWidgets.QLabel()

        self.plot_layout.addWidget(self.ecg_pw, stretch=1)
        self.plot_layout.addWidget(self.pressure_pw, stretch=1)
        self.plot_layout.addWidget(self.laser1_pw, stretch=1)
        self.plot_layout.addWidget(self.laser2_pw, stretch=1)
        self.plot_layout.addWidget(self.ecg3_pw, stretch=1)
        self.plot_layout.addWidget(self.rvshock_pw, stretch=1)
        self.plot_layout.addWidget(self.rvbip_pw, stretch=1)
        self.plot_layout.addWidget(self.lvlead_pw, stretch=1)
        self.plot_layout.addWidget(self.ralead_pw, stretch=1)
        self.plot_layout.addWidget(self.overview_pw, stretch=1)
        self.plot_layout.addWidget(self.zipfl_lbl)
        self.zipfl_lbl.setText(str(self.laser_exp.zip_fl))

        self.plot_layout.setSpacing(0)
        self.plot_layout.setContentsMargins(0,0,0,0)


        self.minibtn_layout.addLayout(self.turnbtn_layout)
        self.main_layout.addLayout(self.btn_layout)
        self.plot_layout.addLayout(self.minibtn_layout)
        self.main_layout.addLayout(self.plot_layout)

        self.setLayout(self.main_layout)

    def setup_toggles(self):
        self.bg_toggle.setChecked(True)
        #self.bg_toggle.setChecked(False)
        self.laser1_toggle.setChecked(True)
        self.laser2_toggle.setChecked(True)
        self.ecg3_toggle.setChecked(True)
        self.rvshock_toggle.setChecked(True)
        self.rvbip_toggle.setChecked(True)
        self.lvlead_toggle.setChecked(True)
        self.ralead_toggle.setChecked(True)
        self.hrv_toggle.setChecked(False)

    def setup_plots(self):
        self.overview_region = pg.LinearRegionItem()
        self.overview_region.setZValue(10)
        self.overview_pi.addItem(self.overview_region)
        self.overview_region.sigRegionChanged.connect(self.overview_region_changed)
        self.overview_region.setRegion((0,self.laser_exp.ecg.data.shape[0]))

        self.calc_region = pg.LinearRegionItem()
        self.calc_region.setZValue(-10)
        self.ecg_pi.addItem(self.calc_region)
        self.calc_region.sigRegionChanged.connect(self.calc_region_changed)

        self.ecg_pw.sigRangeChanged.connect(self.overview_region_update)
        self.pressure_pw.sigRangeChanged.connect(self.overview_region_update)
        self.laser1_pw.sigRangeChanged.connect(self.overview_region_update)
        self.laser2_pw.sigRangeChanged.connect(self.overview_region_update)
        self.ecg3_pw.sigRangeChanged.connect(self.overview_region_update)
        self.rvshock_pw.sigRangeChanged.connect(self.overview_region_update)
        self.rvbip_pw.sigRangeChanged.connect(self.overview_region_update)
        self.lvlead_pw.sigRangeChanged.connect(self.overview_region_update)
        self.ralead_pw.sigRangeChanged.connect(self.overview_region_update)

        self.ecg_pi.getAxis('left').setWidth(w=50)
        self.ecg_pi.getAxis('left').setStyle(showValues=False)
        self.pressure_pi.getAxis('left').setWidth(w=50)
        self.laser1_pi.getAxis('left').setWidth(w=50)
        self.laser2_pi.getAxis('left').setWidth(w=50)
        self.ecg3_pi.getAxis('left').setWidth(w=50)
        self.ecg3_pi.getAxis('left').setStyle(showValues=False)
        self.rvshock_pi.getAxis('left').setWidth(w=50)
        self.rvshock_pi.getAxis('left').setStyle(showValues=False)
        self.rvbip_pi.getAxis('left').setWidth(w=50)
        self.rvbip_pi.getAxis('left').setStyle(showValues=False)
        self.lvlead_pi.getAxis('left').setWidth(w=50)
        self.lvlead_pi.getAxis('left').setStyle(showValues=False)
        self.ralead_pi.getAxis('left').setWidth(w=50)
        self.ralead_pi.getAxis('left').setStyle(showValues=False)
        self.overview_pi.getAxis('left').setWidth(w=50)
        self.overview_pi.getAxis('left').setStyle(showValues=False)

    def update_data(self):

        samples = self.laser_exp.pressure.data.shape[0]

        #self.pressure_plt.setData(self.laser_exp.pressure.data, pen='r', antialise=True, autoDownsample=True, clipToView=True)
        self.pressure_plt.setData(x=np.arange(samples), y=self.laser_exp.pressure.data, pen='#fc0303', antialise=True, autoDownsample=True, clipToView=True)

        self.ecg3_plt.setData(x=np.arange(samples), y=self.laser_exp.ecg3.data, pen='#02B83A', antialise=True, autoDownsample=True, clipToView=True)
        self.rvshock_plt.setData(x=np.arange(samples), y=self.laser_exp.rvshock.data, pen='#732F9B', antialise=True, autoDownsample=True, clipToView=True)
        self.rvbip_plt.setData(x=np.arange(samples), y=self.laser_exp.rvbip.data, pen='#732F9B', antialise=True, autoDownsample=True, clipToView=True)
        self.lvlead_plt.setData(x=np.arange(samples), y=self.laser_exp.lvlead.data, pen='#732F9B', antialise=True, autoDownsample=True, clipToView=True)
        self.ralead_plt.setData(x=np.arange(samples), y=self.laser_exp.ralead.data, pen='#732F9B', antialise=True, autoDownsample=True, clipToView=True)

        if self.laser_combo.currentIndex() == 0:
            laser1_data = self.laser_exp.laser1.data
            laser2_data = self.laser_exp.laser2.data
        elif self.laser_combo.currentIndex() == 1:
            laser1_data = mmt.butter_bandpass_filter(self.laser_exp.laser1.data, 0.5, 25, 1000, order=2)
            laser2_data = mmt.butter_bandpass_filter(self.laser_exp.laser2.data, 0.5, 25, 1000, order=2)
            envelope1_data = np.abs(scipy.signal.hilbert(laser1_data))
            envelope1_data = mmt.butter_lowpass_filter(envelope1_data, 1, 1000, 2)
            envelope2_data = np.abs(scipy.signal.hilbert(laser2_data))
            envelope2_data = mmt.butter_lowpass_filter(envelope2_data, 1, 1000, 2)
        elif self.laser_combo.currentIndex() == 2:
            laser1_data = mmt.butter_bandpass_filter(np.log(self.laser_exp.laser1.data+20) + self.laser_exp.laser1.data/100, 0.5, 25, 1000, order=2)
            laser2_data = mmt.butter_bandpass_filter(np.log(self.laser_exp.laser2.data+20) + self.laser_exp.laser2.data/100, 0.5, 25, 1000, order=2)
            envelope1_data = np.abs(scipy.signal.hilbert(laser1_data))
            envelope1_data = mmt.butter_lowpass_filter(envelope1_data, 1, 1000, 2)
            envelope2_data = np.abs(scipy.signal.hilbert(laser2_data))
            envelope2_data = mmt.butter_lowpass_filter(envelope2_data, 1, 1000, 2)

        envelope1_data = np.array([])
        envelope2_data = np.array([])

        #self.laser1_plt.setData(laser1_data, pen='#3273F5', antialise=True, autoDownsample=True, clipToView=True)
        #self.laser2_plt.setData(laser2_data, pen='#3273F5', antialise=True, autoDownsample=True, clipToView=True)
        self.laser1_plt.setData(x=np.arange(samples), y=laser1_data, pen='#03c6fc', pensize=50, antialise=True, autoDownsample=True, clipToView=True)
        self.laser2_plt.setData(x=np.arange(samples), y=laser2_data, pen='#03c6fc', antialise=True, autoDownsample=True, clipToView=True)
        #self.envelope1_plt.setData(x=np.arange(envelope1_data.shape[0]), y=envelope1_data, pen='g', antialise=True, autoDownsample=True, clipToView=True)
        #self.envelope2_plt.setData(x=np.arange(envelope2_data.shape[0]), y=envelope2_data, pen='g', antialise=True, autoDownsample=True, clipToView=True)

        self.overview_plt.setData(x=np.arange(samples), y=self.laser_exp.ecg.data, pen='#0FA00F', symbol=None, antialise=True, autoDownsample=True, clipToView=True)

        self.ecg_plt.setData(x=np.arange(samples), y=self.laser_exp.ecg.data, pen='02B83A', symbol=None, antialise=True, autoDownsample=True, clipToView=True)


        if not self.hrv_toggle.isChecked():
            print("off")
            self.ecg_peak_plt.setData(x=[], y=[],
                                      pen=None, symbol=None,
                                      antialise=True, autoDownsample=False, downsampleMethod='peak', clipToView=False)

        elif self.hrv_toggle.isChecked():
            self.ecg_peak_plt.setData(x=self.ecg_peaks_x, y=self.ecg_peaks_y,
                                      pen=None, symbol='o', size=4, pxMode=True,
                                      antialise=True, autoDownsample=False, downsampleMethod='peak', clipToView=False)

    def overview_region_changed(self):
        self.overview_region.setZValue(10)
        minX, maxX = self.overview_region.getRegion()

        self.ecg_pw.plotItem.setXRange(minX, maxX, padding=0)
        self.pressure_pw.plotItem.setXRange(minX, maxX, padding=0)
        self.laser1_pw.plotItem.setXRange(minX, maxX, padding=0)
        self.laser2_pw.plotItem.setXRange(minX, maxX, padding=0)
        self.ecg3_pw.plotItem.setXRange(minX, maxX, padding=0)
        self.rvshock_pw.plotItem.setXRange(minX, maxX, padding=0)
        self.rvbip_pw.plotItem.setXRange(minX, maxX, padding=0)
        self.lvlead_pw.plotItem.setXRange(minX, maxX, padding=0)
        self.ralead_pw.plotItem.setXRange(minX, maxX, padding=0)

    def overview_region_update(self, window, viewRange):
        rgn = viewRange[0]
        self.overview_region.setRegion(rgn)

    def laser1_toggle_changed(self, state):
        if state == QtCore.Qt.Checked:
            self.laser1_pw.show()
        else:
            self.laser1_pw.hide()

    def laser2_toggle_changed(self, state):
        if state == QtCore.Qt.Checked:
            self.laser2_pw.show()
        else:
            self.laser2_pw.hide()

    def ecg3_toggle_changed(self, state):
        if state == QtCore.Qt.Checked:
            self.ecg3_pw.show()
        else:
            self.ecg3_pw.hide()

    def rvshock_toggle_changed(self, state):
        if state == QtCore.Qt.Checked:
            self.rvshock_pw.show()
        else:
            self.rvshock_pw.hide()

    def rvbip_toggle_changed(self, state):
        if state == QtCore.Qt.Checked:
            self.rvbip_pw.show()
        else:
            self.rvbip_pw.hide()

    def lvlead_toggle_changed(self, state):
        if state == QtCore.Qt.Checked:
           self.lvlead_pw.show()
        else:
           self.lvlead_pw.hide()

    def ralead_toggle_changed(self, state):
        if state == QtCore.Qt.Checked:
            self.ralead_pw.show()
        else:
            self.ralead_pw.hide()

    def calc_region_changed(self):
        roi_begin, roi_end = self.calc_region.getRegion()
        self.begin_lbl.setText("Begin: " + str(int(roi_begin)))
        self.end_lbl.setText("End: " + str(int(roi_end)))

    def calc_region_move(self):
        x, y = self.overview_region.getRegion()
        print(x,y)

        rgn = (x+(y-x)/4, y-(y-x)/4)
        self.calc_region.setRegion(rgn)

    def bg_toggle_changed(self, state):

        p = self.palette()

        if state == QtCore.Qt.Checked:
            p.setColor(self.backgroundRole(), QtCore.Qt.white)
            p.setColor(self.foregroundRole(), QtCore.Qt.black)
            self.ecg_pw.setBackground("w")
            self.pressure_pw.setBackground("w")
            self.laser1_pw.setBackground("w")
            self.laser2_pw.setBackground("w")
            self.ecg3_pw.setBackground("w")
            self.rvshock_pw.setBackground("w")
            self.rvbip_pw.setBackground("w")
            self.lvlead_pw.setBackground("w")
            self.ralead_pw.setBackground("w")
            self.overview_pw.setBackground("w")
        else:
            p.setColor(self.backgroundRole(), QtCore.Qt.black)
            p.setColor(self.foregroundRole(), QtCore.Qt.white)
            self.ecg_pw.setBackground("k")
            self.pressure_pw.setBackground("k")
            self.laser1_pw.setBackground("k")
            self.laser2_pw.setBackground("k")
            self.ecg3_pw.setBackground("k")
            self.rvshock_pw.setBackground("k")
            self.rvbip_pw.setBackground("k")
            self.lvlead_pw.setBackground("k")
            self.ralead_pw.setBackground("k")
            self.overview_pw.setBackground("k")

        self.setPalette(p)

    def calc_results(self):
        roi_begin, roi_end = self.calc_region.getRegion()
        self.laser_exp.begin = int(roi_begin)
        self.laser_exp.end = int(roi_end)

        try:
            self.laser_exp.process()
            print(self.laser_exp.results)
        except Exception as e:
            print("Problem in calculation")
            print(e)
        else:
            laser1_magic = self.laser_exp.results['Laser1_Magic']
            if not laser1_magic:
                laser1_magic = float("NaN")

            laser2_magic = self.laser_exp.results['Laser2_Magic']
            if not laser2_magic:
                laser2_magic = float("NaN")

            laser1_conf = self.laser_exp.results['Laser1_Conf']
            if not laser1_conf:
                laser1_conf = float("NaN")

            laser2_conf = self.laser_exp.results['Laser2_Conf']
            if not laser2_conf:
                laser2_conf = float("NaN")

            laser1_text = f"Laser1: {laser1_magic:.3f} conf:{laser1_conf:.3f}"
            laser2_text = f"Laser2: {laser2_magic:.3f} conf:{laser2_conf:.3f}"

            self.laser1_value.setText(laser1_text)
            self.laser2_value.setText(laser2_text)
            self.rr_value.setText("RR (Bipolar ECG): " + str(int(np.mean(np.diff(self.laser_exp.ecg.peaks_sample)))))
            self.hr_value.setText("HR: " + str(int(60000 / np.mean(np.diff(self.laser_exp.ecg.peaks_sample)))))
            try:
                self.sbp_value.setText("SBP: " + str(self.laser_exp.results['SBP_Mean']))
                self.map_value.setText("MAP: " + str(self.laser_exp.results['MAP_Mean']))

            #     if self.laser_exp.results['dSBP_Mean'] != "":
            #         self.distsbp_value.setText("dSBP: " + str(self.laser_exp.results['dSBP_Mean']))
            #         self.distmap_value.setText("dMAP: " + str(self.laser_exp.results['dMAP_Mean']))

            except Exception as e:
                print(e)
            try:
                laser1_array_ys = 100 * ((self.laser_exp.laser1_magic_data_all))
            except:
                laser1_array_ys = np.zeros((1, 1000))

            try:
                laser1_sum = 100 * ((self.laser_exp.laser1_magic_data))
            except:
                laser1_sum = np.zeros((1000))

            try:
                laser2_array_ys = 100 * ((self.laser_exp.laser2_magic_data_all))
            except:
                laser2_array_ys = np.zeros((1, 1000))

            try:
                laser2_sum = 100 * ((self.laser_exp.laser2_magic_data))
            except:
                laser2_sum = np.zeros((1000))

            dual_laser_window = DualLaserWindow(laser1_array_ys=laser1_array_ys,
                                                laser1_sum=laser1_sum,
                                                laser2_array_ys=laser2_array_ys,
                                                laser2_sum=laser2_sum,
                                                parent=self)


            dual_laser_window.show()
    def calc_6s(self):
        roi_begin, roi_end = self.calc_region.getRegion()
        roi_end = roi_begin + 6000
        self.calc_region.setRegion((roi_begin, roi_end))

    def screenshot(self):
        print("Saving Screenshot")

        try:
            img = self.grab()

            now = datetime.datetime.now().isoformat()[0:19].replace(":", "").replace("-", "")

            f_dir = os.path.join(self.laser_exp.database_dir, "Output")
            f_fn = self.laser_exp.patient + "-" + self.laser_exp.exp + "-" + str(self.laser_exp.begin) + "-" + str(self.laser_exp.end) + "-" + now + ".png"
            f_fl = os.path.join(f_dir, f_fn)

            os.makedirs(f_dir, exist_ok=True)
            img.save(f_fl, "png")
        except Exception as e:
            print("Saving Failed")
            print(e)
        else:
            print("Image Saved")

    def calc_hrv(self):
        ecg_hint = self.laser_exp.hints['Period']
        print(ecg_hint)

        self.laser_exp.ecg.calc_ecg_peaks(ecg_hint=ecg_hint)
        self.hrv_xs = self.laser_exp.ecg.peaks_sample[1:]
        self.hrv_ys = 60000/np.diff(self.laser_exp.ecg.peaks_sample)
        self.ecg_peaks_x = self.laser_exp.ecg.peaks_sample
        self.ecg_peaks_y = self.laser_exp.ecg.peaks_value


    def hrv_toggle_changed(self, state):
        if state == QtCore.Qt.Checked:
            if not hasattr(self, "hrv_xs"):
                self.calc_hrv()

        self.update_data()

class DualLaserWindow(QtWidgets.QDialog):
    def __init__(self, laser1_array_ys, laser1_sum, laser2_array_ys, laser2_sum, parent=None):
        super().__init__(parent=parent)
        p = self.palette()
        p.setColor(self.backgroundRole(), QtCore.Qt.white)
        p.setColor(self.foregroundRole(), QtCore.Qt.black)

        laser_sum_pen = pg.mkPen('k', width=8)

        laser1_array_xs = np.empty_like(laser1_array_ys)
        laser1_array_xs[:] = np.arange(laser1_array_ys.shape[1])[np.newaxis,:]
        laser1_multilines = MultiLine(laser1_array_xs, laser1_array_ys, pen_args={'color': (150,150,150), 'width': 2})

        laser1_sum_xs = np.arange(laser1_sum.shape[0])

        laser2_array_xs = np.empty_like(laser2_array_ys)
        laser2_array_xs[:] = np.arange(laser2_array_ys.shape[1])[np.newaxis,:]
        laser2_multilines = MultiLine(laser2_array_xs, laser2_array_ys, pen_args={'color': (150,150,150), 'width': 2})

        laser2_sum_xs = np.arange(laser2_sum.shape[0])

        self.laser1_pw = pg.PlotWidget()
        self.laser1_pw.setBackground('w')

        self.laser2_pw = pg.PlotWidget()
        self.laser2_pw.setBackground('w')

        self.plot_layout = QtWidgets.QVBoxLayout()
        self.plot_layout.addWidget(self.laser1_pw)
        self.plot_layout.addWidget(self.laser2_pw)

        self.setLayout(self.plot_layout)

        self.laser1_pw.addItem(laser1_multilines)
        self.laser1_pw.plot(laser1_sum_xs, laser1_sum, pen=laser_sum_pen)

        self.laser2_pw.addItem(laser2_multilines)
        self.laser2_pw.plot(laser2_sum_xs, laser2_sum, pen=laser_sum_pen)



class MultiLine(QtWidgets.QGraphicsPathItem):
    def __init__(self, x, y, pen_args={'color': 'k'}):
        """x and y are 2D arrays of shape (Nplots, Nsamples)"""
        connect = np.ones(x.shape, dtype=bool)
        connect[:,-1] = 0 # don't draw the segment between each trace
        self.path = pg.arrayToQPath(x.flatten(), y.flatten(), connect.flatten())
        QtWidgets.QGraphicsPathItem.__init__(self, self.path)
        self.setPen(pg.mkPen(**pen_args))
    def shape(self): # override because QGraphicsPathItem.shape is too expensive.
        return QtWidgets.QGraphicsItem.shape(self)
    def boundingRect(self):
        return self.path.boundingRect()
