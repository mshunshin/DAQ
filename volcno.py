import os
import sys
import numpy as np

from pathlib import Path

import matplotlib
matplotlib.use('qt5agg')

import matplotlib.pyplot as plt

import hdy



SDY_PATH = Path("/Volumes/Ashur Pro2/dicom_database/sdy1/CMStudy_2019_11_12_144859.sdy")
SDY_PATH = Path("/Volumes/Matt-Data/test2.sdy")

d = hdy.SDY_File(SDY_PATH)

#plt.imshow(np.frombuffer(d.spectrum_a[10000:10200,:].tobytes(), dtype=np.uint16).reshape((200,256)).T)

temp = np.zeros((400, 256), dtype=np.uint16)
temp[::2, :] = d.spectrum_a[10300:10500, :]
temp[1::2, :] = d.spectrum_b[10300:10500, :]

from scipy.cluster.vq import vq

code_book = np.array([
    64,
    2,
    4,
    10,
    100,
    1000,
    10000
])

code_book_a = 0 + (np.arange(0, 128) * 64)  #11072
code_book_b = 8128 + (np.arange(0, 64) * 128) #33088
code_book_c = 16192 + (np.arange(0, 64) * 256)

code_book = np.concatenate((code_book_a, code_book_b, code_book_c))

temp_shape = temp.shape

temp2, _ = vq(temp.ravel(), code_book)
temp2 = np.reshape(temp2, temp_shape)

temp3 = temp/256
temp4 = (temp*2)**0.5

plt.imshow(temp2.T, cmap='gray', vmin=0, vmax=255)
plt.show()
plt.imshow(temp3.T, cmap='gray', vmin=0, vmax=255)
plt.show()
plt.imshow(temp4.T, cmap='gray', vmin=0, vmax=255)
plt.show()