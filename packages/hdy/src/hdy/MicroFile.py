from pathlib import Path

import numpy as np
import pandas as pd


class MicroFile():
    def __init__(self, file_dir, file_fn):

        file_path = Path(file_dir) / file_fn

        metadata = dict()
        skip_lines = None
        lines_read = 0
        with open(file_path) as f:
            while True:
                line: str = f.readline()
                lines_read = lines_read + 1

                if line.startswith("Date/Time,"):
                    metadata['datetime'] = line.strip("Date/Time,")
                elif line.startswith("Protocol,"):
                    metadata['protocol'] = line.strip("Protocol,")
                elif line.startswith("S1A,"):
                    metadata['S1A'] = line.strip("S1A,")
                elif line.startswith("S2AV,"):
                    metadata['S2AV'] = line.strip("S2AV,")
                elif line.startswith("Decrement,"):
                    metadata['Decrement,'] = line.strip("Decrement,")
                elif line.startswith("Time [ms]"):
                    skip_lines = lines_read - 1
                    break
                else:
                    pass

        data = pd.read_csv(file_path, skiprows=skip_lines)
        self.data = data
        self.timecode = np.array(data['Time [ms]'])
        self.sampling_rate = int(1000/np.mean(np.diff(np.array(data['Time [ms]']))))
        self.ppg = np.array(data[' PPG [uv]'])
        self.ecg = np.array(data[' CH1 [uv]'])
        self.ecg2 = np.array(data[' CH2 [uv]'])
        self.flag = np.array(data[' Flag'])
        self.boxa = np.array(data[' A-AV Type'])
        self.zip_fl = str(file_path)
        self.blank = np.zeros_like(getattr(self, "ecg"))
        self.sources = ["ppg"]

        print(f"{metadata['S2AV']}: {file_fn}")


if __name__ == "__main__":
    file_dir = "/Volumes/Matt-Temp/AAV_DATA_RECORDING/"
    file_name = "AAV_DATA_20240520-00000000-0003.CSV"

    d = MicroFile(file_dir=file_dir, file_fn=file_name)

    print(d)

