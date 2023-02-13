import os

from PySide6 import QtGui, QtCore, QtWidgets

import pyqtgraph as pg

from .DAQ_File import DAQ_File


class DAQ_GUI(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.daq_exp_widget = None

        self.statusBar().showMessage('Ready')

        self.open_file_btn = QtWidgets.QPushButton("Open DAQ File")
        self.open_file_btn.clicked.connect(self.showFileOpen)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')

        openFile = QtGui.QAction(QtGui.QIcon('open.png'), 'Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open new File')
        openFile.triggered.connect(self.showFileOpen)

        fileMenu.addAction(openFile)
        menubar.addMenu(fileMenu)

        self.setWindowTitle('MattCon 2000')
        self.setCentralWidget(self.open_file_btn)
        self.resize(1000, 800)
        self.show()

    def showFileOpen(self):
        daq_exp_fl = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/Users')

        if daq_exp_fl is None:
            return

        if True:
            daq_exp_dir, daq_exp_fn = os.path.split(daq_exp_fl[0])
            daq_exp = DAQ_File(daq_exp_dir, daq_exp_fn, search_for_files=False)
            self.daq_exp_widget = DAQ_Exp_Widget(daq_exp)
            self.setCentralWidget(self.daq_exp_widget)
        #except Exception as e:
        #    print("Error loading daq file")
        #    print(e)


class DAQ_Exp_Widget(QtWidgets.QWidget):

    def __init__(self, daq_exp, parent = None):
        super().__init__(parent=parent)

        self.plot_w_list = []
        self.ecg_plot_num = None

        self.daq_exp = daq_exp
        self.setup()

        self.show()

    def setup(self):

        self.control_layout = QtWidgets.QVBoxLayout()

        self.pos_lbl = QtWidgets.QLabel()
        self.zipfl_lbl = QtWidgets.QLabel()

        self.matt_lbl = QtWidgets.QLabel()
        self.matt_lbl.setText("Dr. Matthew Shun-Shin\nBHF Clinical Fellow\nImperial College London")

        self.control_layout.addWidget(self.pos_lbl)
        self.control_layout.addStretch(1)
        self.control_layout.addWidget(self.matt_lbl)
        self.control_layout.addStretch(10)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0,0,0,0)

        for i, source in enumerate(self.daq_exp.sources):
            if source == 'ecg':
                self.ecg_plot_num = i

            data = getattr(self.daq_exp, source)

            plot_w = pg.PlotWidget()
            plot_pi = plot_w.getPlotItem()
            plot_pi.plot(data, pen="r", antialise=True, autoDownsample=True, downsampleMethod='peak', clipToView=False)
            plot_pi.setClipToView(True)
            plot_pi.showAxis('bottom', False)
            plot_pi.setLabel('left', source)
            plot_pi.sigRangeChanged.connect(self.update_region)
            #plot_pi.setAutoVisible(y=True)
            plot_pi.getAxis('left').setWidth(75)

            self.plot_w_list.append(plot_w)

            self.layout.addWidget(plot_w)

        if True:
            source = 'ecg'
            data = getattr(self.daq_exp, source)

            plot_w = pg.PlotWidget()
            plot_pi = plot_w.getPlotItem()
            plot_pi.plot(data, pen="r", antialise=True, autoDownsample=True, downsampleMethod='peak', clipToView=False)
            plot_pi.setClipToView(True)
            plot_pi.showAxis('bottom', False)
            plot_pi.setLabel('left', source)
            #plot_pi.setAutoVisible(y=True)
            plot_pi.getAxis('left').setWidth(75)

            self.overview_region = pg.LinearRegionItem()
            self.overview_region.setZValue(10)
            self.overview_region.sigRegionChanged.connect(self.region_updated)

            plot_pi.addItem(self.overview_region, ignoreBounds=True)

            self.layout.addWidget(plot_w)

        if self.ecg_plot_num is not None:
            i = self.ecg_plot_num
            self.vLine = pg.InfiniteLine(angle=90, movable=False)
            self.hLine = pg.InfiniteLine(angle=0, movable=False)
            self.plot_w_list[i].addItem(self.vLine, ignoreBounds=True)
            self.plot_w_list[i].addItem(self.hLine, ignoreBounds=True)
            self.ecg_vb = self.plot_w_list[i].getPlotItem().vb
            self.plot_w_list[i].scene().sigMouseMoved.connect(self.mouse_moved)

        self.main_layout = QtWidgets.QHBoxLayout()
        self.main_layout.addLayout(self.control_layout)
        self.main_layout.addLayout(self.layout)
        self.layout.addWidget(self.zipfl_lbl)
        self.zipfl_lbl.setText(str(self.daq_exp.zip_fl))

        self.setLayout(self.main_layout)


    def update_region(self, window, viewRange):
        rgn = viewRange[0]
        self.overview_region.setRegion(rgn)

    def region_updated(self):
        self.overview_region.setZValue(10)
        minX, maxX = self.overview_region.getRegion()
        for plot_w in self.plot_w_list:
            plot_w.setXRange(minX, maxX, padding=0)

    def mouse_moved(self, evt):
        if self.ecg_plot_num is None:
            return

        i = self.ecg_plot_num

        pos = evt.toPoint()

        if self.plot_w_list[i].sceneBoundingRect().contains(pos):
            mousePoint = self.ecg_vb.mapSceneToView(pos)
            index = int(mousePoint.x())
            if index > 0:
                self.pos_lbl.setText("X: {pos}".format(pos=int(mousePoint.x())))
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())
