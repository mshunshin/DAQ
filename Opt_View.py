import sys
import os
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'

from PyQt5 import QtWidgets

import hdy


def main():

    app = QtWidgets.QApplication(sys.argv)

    selector = hdy.Opt_Selector_GUI()

    app.exec_()

    print(selector.source)
    print(selector.source_hints)

    opt_collection = hdy.OptCollection(source=selector.source, source_hints=selector.source_hints)
    opt_gui = hdy.Opt_GUI(opt_collection)

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()