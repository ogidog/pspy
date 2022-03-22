# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 14:57:34 2021

@author: anyuser
"""

import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import upfirdn
from numpy import convolve as np_convolve
from scipy.signal import fftconvolve, lfilter, firwin, filtfilt, resample_poly, iirdesign
from scipy.signal import convolve as sig_convolve
from scipy.ndimage import convolve1d
from scipy.signal.windows import gaussian
import matplotlib.pyplot as plt

xpuw = loadmat('C:\\Users\\Ryzen\\Documents\\PYTHON\\stampsexport\\ps_plot_v-d.mat')
xmuw = loadmat('C:\\Users\\Ryzen\\Documents\\MATLAB\\stampsexport\\ps_plot_v-d.mat')

eq_ind = []
for i in range(len(xmuw['lonlat'])):
    for j in range(len(xpuw['lonlat'])):
        if xmuw['lonlat'][i,0] == xpuw['lonlat'][j,0] and xmuw['lonlat'][i,1] == xpuw['lonlat'][j,1]:
            eq_ind.append([i, j])

dph = np.zeros(len(eq_ind))
for i in range(len(eq_ind)):
    dph[i] = (xpuw['ph_disp'][eq_ind[i][1]] - xmuw['ph_disp'][eq_ind[i][0]]) / xmuw['ph_disp'][eq_ind[i][0]]

plt.figure()
plt.plot(xpuw['lonlat'][:,0], xpuw['lonlat'][:,1], 'r.', alpha = 1.0)
plt.plot(xmuw['lonlat'][:,0], xmuw['lonlat'][:,1], 'b.', alpha = 0.5)