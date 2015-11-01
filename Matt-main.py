import numpy as np
import scipy
import scipy.signal
import matplotlib.pyplot as plt

import mmt

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.Point import Point

## You need to find out the calibration constants for the pd and pa. Also, check pd and pa are the correct way around.
## If only multiple, then obv their ratio is ok, but if there is a consant offset will affect calculation
## P.s.
## Whoever designed this file format was insane.
## Storing data as unsigned ints.
## Insane interleaving

def find_nearest(a, a0):
    return a[np.abs(a-a0).argmin()]

class SDYFile(object):
    def __init__(self, sdy_fl):
        self.plot_order = False #If set to false then red is drawn over green
        self.sampling_rate = 200 ###Check this - should be a multiple of 4!! this is only a guess
        with open(sdy_fl, 'rb') as sdy_f:
            sdy_f.seek(0)
            self.file_type = np.fromfile(sdy_f, dtype=np.uint32, count=1)
            self.date_time = np.fromfile(sdy_f, dtype=np.uint32, count=2)
            self.exam_type = np.fromfile(sdy_f, dtype=np.int32, count=1)
            #3=pressure, 4=Flow, 5=Combo
            
            details = ['last_name',
                      'first_name',
                      'middle_initial',
                      'gender',
                      'patient_id',
                      'physcian',
                      'date_of_birth',
                      'procedure',
                      'procedure_id',
                      'accession',
                      'ffr',
                      'ffr_suid',
                      'refering_physcian',
                      'additional_pt_history',
                      'ivus_suid',
                      'department',
                      'institution',
                      'cathlab_id']
            
            pt_details = {}
            
            for detail in details:
                temp = sdy_f.read(512)
                pt_details[detail] = temp.decode('utf-16').replace('\x00', '').strip()
            
            self.pt_details = pt_details
            
            temp = np.fromfile(sdy_f, dtype=np.uint16, count=-1)
            temp_width = len(temp)/(44+1079) #44 +1079 channels of data - don't ask me why
            frame = temp.reshape((temp_width, 44+1079)) #The axis to the right is the one which cycles the fastest through the data
            self.frame = frame
            self.pd = PressureData(SDYFile.clean(frame[:,(5,7,9,11)].ravel()), sampling_rate=self.sampling_rate) #Yep, this uninterleaves the data a second dime
            self.pa = PressureData(SDYFile.clean(frame[:, (16,17,18,19)].ravel()), sampling_rate=self.sampling_rate)
            self.ecg = ECGData(SDYFile.clean(frame[:, (24,26,28,30)].ravel()), sampling_rate=self.sampling_rate)
            self.flow = FlowData(SDYFile.clean(frame[:,(32,32,33,33)].ravel()), sampling_rate=self.sampling_rate)
            self.calc1 = frame[:, (34,34,35,35)].ravel() #Only represented by 2 interleaved channels - duplicate to match sampling rate
            self.calc2 = frame[:, (36,36,37,37)].ravel()
            self.calc3 = frame[:, (38,38,39,39)].ravel()
            
            self.delay_pd = 0
            self.roi_start = 0
            self.roi_end = self.pd.data.shape[0]
            
    @staticmethod
    def clean(data):
        data_mean = data.mean()
        data_std = data.std()
        data[(data>(data_mean+5*data_std))| (data<(data_mean - 5*data_std))] = 0
        return(data)
    
    def process(self, start, end):
        self.roi_start = start
        self.roi_end = end
        self.pa.find_peaks(self.roi_start, self.roi_end)
        self.pa.find_valleys(self.roi_start, self.roi_end)
        self.pd.find_peaks(self.roi_start, self.roi_end)
        self.pa.find_valleys(self.roi_start, self.roi_end)
        self.calc_ffr()
    
    def calc_ffr(self):
        
        start = self.roi_start
        end = self.roi_end
        sampling_rate = self.sampling_rate
        window = sampling_rate * 10 ##I.e. a 10 second smoother. Check what we should use here
        
        pa = self.pa.data[start:end]
        pd = self.pd.data[start:end]
        
        pa_mean = scipy.ndimage.generic_filter(pa, np.mean, size=window) #This is slow - find a built in
        pd_mean = scipy.ndimage.generic_filter(pd, np.mean, size=window) #This is slow - find a built in
        ffr_live = pd_mean / pa_mean
        ffr = np.min(ffr_live) #Perhaps use lower 95% CI
        
        self.pa_mean = pa_mean
        self.pd_mean = pd_mean
        self.ffr_live = ffr_live
        self.ffr = ffr
    
    def plot_raw(self):
        plt.ioff()
        plt.plot(self.ecg.data, "g")
        plt.plot(self.pa.data, "r")
        plt.plot(self.pd.data, "b")
        plt.plot(self.calc1)
        plt.plot(self.calc2)
        plt.plot(self.calc3)
        plt.show()

        
    def pyqtgraph_plot_raw(self):
        app = QtGui.QApplication([])
        win = pg.GraphicsWindow()
        win.setWindowTitle('pyqtgraph example: crosshair')

        pg.setConfigOptions(antialias=True)

        label = pg.LabelItem(justify='right')
        win.addItem(label)
        p1 = win.addPlot(row=1, col=0)
        p2 = win.addPlot(row=2, col=0)
        p3 = win.addPlot(row=3, col=0)

        region = pg.LinearRegionItem()
        region.setZValue(10)

        p3.addItem(region, ignoreBounds=True)

        #pg.dbg()
        p1.setAutoVisible(y=True)


        #create numpy arrays
        #make the numbers large to show that the xrange shows data from 10000 to all the way 0
        data1 = self.pa.data
        data2 = self.pd.data
        data_ffr = self.ffr_live

        if self.plot_order == True:
            p1.plot(data1, pen="r")
            p1.plot(data2, pen="g")
        else:
            p1.plot(data2, pen="g")            
            p1.plot(data1, pen="r")

        p2.plot(data_ffr, pen='w')
        if self.plot_order == True:
            p3.plot(data1, pen="r")
            p3.plot(data2, pen="g")
        else:
            p3.plot(data2, pen="g")
            p3.plot(data1, pen="r")            

        def update():
            region.setZValue(10)
            minX, maxX = region.getRegion()
            p1.setXRange(minX, maxX, padding=0)
            p2.setXRange(minX, maxX, padding=0)

        region.sigRegionChanged.connect(update)

        def updateRegion(window, viewRange):
            rgn = viewRange[0]
            region.setRegion(rgn)

        p1.sigRangeChanged.connect(updateRegion)

        region.setRegion([1000, 2000])

        #cross hair
        vLine = pg.InfiniteLine(angle=90, movable=False)
        hLine = pg.InfiniteLine(angle=0, movable=False)
        p1.addItem(vLine, ignoreBounds=True)
        p1.addItem(hLine, ignoreBounds=True)


        vb = p1.vb

        def mouseMoved(evt):
            pos = evt[0]  ## using signal proxy turns original arguments into a tuple
            if p1.sceneBoundingRect().contains(pos):
                mousePoint = vb.mapSceneToView(pos)
                index = int(mousePoint.x())
                if index > 0 and index < len(data1):
                    label.setText("<span style='font-size: 12pt'>x=%0.1f,   <span style='color: red'>y1=%0.1f</span>,   <span style='color: green'>y2=%0.1f</span>" % (mousePoint.x(), data1[index], data2[index]))
                vLine.setPos(mousePoint.x())
                hLine.setPos(mousePoint.y())



        proxy = pg.SignalProxy(p1.scene().sigMouseMoved, rateLimit=60, slot=mouseMoved)

        app.exec_()

                
    def plot_ffr(self):
        plt.ioff()
        plt.plot(self.pa.data[self.roi_start:self.roi_end], "r")
        plt.plot(self.pa_mean, "r")
        plt.plot(self.pa.peaks_x-self.roi_start, self.pa.peaks_y, "ro")

        plt.plot(self.pd.data[self.roi_start:self.roi_end], "b")
        plt.plot(self.pd.peaks_x-self.roi_start, self.pd.peaks_y, "bo")
        plt.plot(self.pd_mean, "b")

        plt.plot(acq.ffr_live*100, "g")
        plt.show()
        
    @property
    def segment_xs(self):
        return np.arange(self.roi_start, self.roi_end)
    
    def align(self):
        # You could just use the built in correlation, but need to minimise amount of data correlated to speed up
        start = self.roi_start
        end = self.roi_end
        sampling_rate = self.sampling_rate
        
        window_half = np.min([sampling_rate * 5, end-start])//2
        max_delay = sampling_rate //2
        
        ref_start = end-start//2 - window_half
        ref_end = ref_start + 2*window_half
        ref_data = self.pa.data[ref_start:ref_end]
        
        search_delays = np.arange(-max_delay, max_delay)
        corrcoef_delays = np.zeros_like(search_delays)
        
        for i, delay in enumerate(search_delays):
            search_data = self.pd.data[ref_start-de]
    
class PressureData(object):
    def __init__(self, data, sampling_rate):
        self.sampling_rate = sampling_rate
        self.data = scipy.signal.savgol_filter(data, 31, 5)
        
    def find_peaks(self, start, end):
        #Have to limit the range it as takes ages - dont look at the bits you are not interested in.
        #This is my amazing super pressure wave peak finding algorithm.
        #I works *exceptionally* well.
        #Smooth data (again) then find peaks using continuous wavelet tranform with some optimised test widths.
        #This robustly gets you proper peaks, but not the actual peak - as if peak is assymetric is offset.
        #Then find local maximum (on a sensible scale) of the original dataset.
        #Then iterate through the robust cwt peaks and find the nearest local maximum
        #Once you have optimised it for a few traces with a fixed sample rate and noise profile is as solid as a rock.
        #Patent Pending Matthew Shun-Shin
                
        data = self.data[start:end]
        data_smooth = scipy.signal.savgol_filter(data, 51, 3)
        widths = np.array([40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 225])
        
        cwt_peaks = scipy.signal.find_peaks_cwt(data_smooth, widths)
        relmax_peaks = scipy.signal.argrelmax(data, order=50)[0]
        
        final_peaks = set()
        
        for peak in cwt_peaks:
            final_peaks.add(find_nearest(relmax_peaks, peak))
        
        self.peaks_x = np.array(list(final_peaks)) + start
        
    def find_valleys(self, start, end):
        #Have to limit the range it as takes ages - dont look at the bits you are not interested in.
        #This is my amazing super pressure wave peak finding algorithm.
        #I works *exceptionally* well.
        #Smooth data (again) then find peaks using continuous wavelet tranform with some optimised test widths.
        #This robustly gets you proper peaks, but not the actual peak - as if peak is assymetric is offset.
        #Then find local maximum (on a sensible scale) of the original dataset.
        #Then iterate through the robust cwt peaks and find the nearest local maximum
        #Once you have optimised it for a few traces with a fixed sample rate and noise profile is as solid as a rock.
        #Patent Pending Matthew Shun-Shin
                
        data = -self.data[start:end]
        data_smooth = scipy.signal.savgol_filter(data, 51, 3)
        widths = np.array([40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 225])
        
        cwt_valleys = scipy.signal.find_peaks_cwt(data_smooth, widths)
        relmax_valleys = scipy.signal.argrelmax(data, order=50)[0]
        
        final_valleys = set()
        
        for valley in cwt_valleys:
            final_valleys.add(find_nearest(relmax_valleys, valley))
        
        self.valleys_x = np.array(list(final_valleys)) + start
        
    @property
    def peaks_y(self):
        return self.data[self.peaks_x]
    
    @property
    def valleys_y(self):
        return self.data[self.valleys_x]
    
class FlowData(object):
    def __init__(self, data, sampling_rate):
        self.data = data
        self.sampling_rate = sampling_rate
    
    def find_peaks(self):
        #Put flow detection algorithm here
        self.peaks_x = [1]
        
    @property
    def peaks_y(self):
        return self.data[self.peaks_x]
    
class ECGData(object):
    def __init__(self, data, sampling_rate):
        self.data = data
        self.sampling_rate = sampling_rate
        
    def find_peaks(self, start, end):
        #Put ecg detection algorithm here
        #Put pan tompkinson.
        #I.e.
        #Filer - band pass
        #Square Data
        #Sliding window integration
        #Then shift it back as this pushs the peaks on
        #Then simple peak detection
        #With heristics.
        self.peaks_x = [1]
        
    @property
    def peaks_y(self):
        return self.data[self.peaks_x]
        
        
sdy_fl = "../../IFR Downloaded/19-08-15/CMStudy_2015_07_22_120053.sdy"
acq = SDYFile(sdy_fl)
acq.calc_ffr()
acq.pyqtgraph_plot_raw()

print("hello")