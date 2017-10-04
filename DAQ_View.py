import sys
import os
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'

import hdy

from PyQt5 import QtWidgets

def main():

    app = QtWidgets.QApplication(sys.argv)

    window = hdy.DAQ_GUI()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
