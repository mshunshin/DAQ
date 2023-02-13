import sys
import os
import logging

from PySide6 import QtWidgets

import hdy


def main():
    logging.basicConfig(level=logging.INFO)
    app = QtWidgets.QApplication(sys.argv)

    selector = hdy.Opt_Selector_GUI()

    app.exec_()

    print(selector.source)
    print(selector.source_hints)

    opt_collection = hdy.OptCollection(source=selector.source, source_hints=selector.source_hints)
    opt_gui = hdy.Opt_GUI(opt_collection)

    sys.exit(app.exec())


if __name__ == '__main__':
    main()