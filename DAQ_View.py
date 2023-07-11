import sys
import os
os.environ['PYQTGRAPH_QT_LIB'] = 'PySide6'
import hdy

from PySide6 import QtWidgets


def main():

    app = QtWidgets.QApplication(sys.argv)

    window = hdy.DAQ_GUI()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
