# -*- coding: utf-8 -*-
"""
Created on Mon May 31 11:03:31 2021

@author: Ryzen
"""

import numpy as np
from scipy.io import loadmat, savemat

# pse = loadmat('C:\\Users\\Ryzen\\Documents\\PYTHON\\stampsexport_pse\\PATCH_1\\ps1.mat')
# zry = loadmat('C:\\Users\\Ryzen\\Documents\\PYTHON\\stampsexport_zry\\PATCH_1\\ps1.mat')
# mat = loadmat('C:\\Users\\Ryzen\\Documents\\PYTHON\\stampsexport_mat\\PATCH_1\\ps1.mat')

# pse = loadmat('C:\\Users\\Ryzen\\Documents\\PYTHON\\stampsexport_pse\\PATCH_1\\ph1.mat')
# zry = loadmat('C:\\Users\\Ryzen\\Documents\\PYTHON\\stampsexport_zry\\PATCH_1\\ph1.mat')
# mat = loadmat('C:\\Users\\Ryzen\\Documents\\PYTHON\\stampsexport_mat\\PATCH_1\\ph1.mat')

# pse = loadmat('C:\\Users\\Ryzen\\Documents\\PYTHON\\stampsexport_pse\\PATCH_1\\ph1.mat')

# pse = np.load('C:\\Users\\Ryzen\\Documents\\PYTHON\\stampsexport_pse\\PATCH_1\\xy2.npy')
# zry = np.load('C:\\Users\\Ryzen\\Documents\\PYTHON\\stampsexport_zry\\PATCH_1\\xy2.npy')

fname = 'select'
zry = loadmat('C:\\Users\\Ryzen\\Documents\\PYTHON\\stampsexport\\PATCH_1\\' + fname + '1.mat')
mat = loadmat('C:\\Users\\Ryzen\\Documents\\PYTHON\\stampsexport_mat\\PATCH_1\\' + fname + '1.mat')

vname = 'keep_ix'
d = (np.abs(mat[vname]) - np.abs(zry[vname]))
maxi = np.argmax(d)
maxv = np.max(d)
mean = np.mean(d)

