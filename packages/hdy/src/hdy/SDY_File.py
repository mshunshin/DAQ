import numpy as np


class SDY_File:
    chan_start = np.array([0])
    chan_unknown1 = np.array([1])
    chan_blank1 = np.array([2, 3])
    chan_unknown2 = np.array([4, 6, 8, 10])
    chan_pd = np.array([5, 7, 9, 11])
    chan_unknown_pa = np.array([12, 13, 14, 15])  # Sometime pa, sometimes noise
    chan_pa = np.array([16, 17, 18, 19])
    chan_unknown3 = np.array([20, 21, 22, 23])
    chan_blank2 = np.array([25, 27, 31])
    chan_ecg = np.array([24, 26, 28, 30])
    chan_ecg_marker = np.array([29])
    chan_flow = np.array([32, 33])
    chan_calc1 = np.array([34])
    chan_blank3 = np.array([35])
    chan_calc2 = np.array([36, 37])
    chan_calc3 = np.array([38, 39])
    chan_calc4 = np.array([40])
    chan_blank4 = np.array([41])
    chan_calc5 = np.array([42])
    chan_end = np.array([43])

    # 1112 - ?
    # 1114 - ?
    # 1118 - ?
    # 1120 - on off
    # 1121 - ?

    def __init__(self, sdy_fl):
        self.plot_order = False
        self.sampling_rate = 50  ###Check this is only a guess
        with open(sdy_fl, 'rb') as sdy_f:
            sdy_f.seek(0)
            self.file_type = np.fromfile(sdy_f, dtype=np.uint32, count=1)
            self.date_time = np.fromfile(sdy_f, dtype=np.uint32, count=2)
            self.exam_type = np.fromfile(sdy_f, dtype=np.int32, count=1)
            # 3=pressure, 4=Flow, 5=Combo

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
            temp_width = len(temp) // (44 + 1079)  # 44 +1079 channels of data - don't ask me why
            temp_len = temp_width * (44 + 1079)
            rem = len(temp) % (44 + 1079)
            sdy_f.read(rem * 2)

            frame = temp[:temp_len].reshape(
                (temp_width, 44 + 1079))  # The axis to the right is the one which cycles the fastest through the data
            self.frame = frame

            pd_raw = frame[:, self.chan_pd].mean(axis=1)
            self.pd_raw = np.pad(pd_raw[3:], (0, 3), 'edge')

            self.pa_raw = frame[:, self.chan_pa].astype(np.float).mean(axis=1)
            self.ecg_raw = frame[:, self.chan_ecg].astype(np.float).mean(axis=1)
            self.flow_raw = frame[:, self.chan_flow].astype(np.float).mean(axis=1)
            self.calc1_raw = frame[:, self.chan_calc1].astype(np.float).mean(axis=1)
            self.calc2_raw = frame[:, self.chan_calc2].astype(np.float).mean(axis=1)
            self.calc3_raw = frame[:, self.chan_calc3].astype(np.float).mean(axis=1)
            self.spectrum_raw = (frame[:, 88:344] + frame[:, 344:600]) / 2
