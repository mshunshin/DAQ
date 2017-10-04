import numpy as np
import pandas as pd

class Bard_File():
    def __init__(self, bard_fl):
        self.bard_fl = bard_fl
        print("hello")
        with open(bard_fl) as f:
            while True:
                line = f.readline()
                if line.startswith("[Data]"):
                    break

            self.pot = Bard_File.fast_load_txt(f).T

    @classmethod
    def fast_load_txt(file_f):
        return(np.array(pd.read_csv(file_f, delimiter=',', dtype=np.float, header=None).values))

