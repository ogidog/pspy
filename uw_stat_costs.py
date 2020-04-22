import numpy as np

from scipy.io import loadmat
from utils import *


def uw_stat_costs(*args):
    # params: unwrap_method,variance,subset_ifg_index

    if len(args) < 1:
        unwrap_method = '3D'

    costscale = 100
    nshortcycle = 200
    maxshort = 32000

    print('Unwrapping in space...\n')

    uw = loadmat('uw_grid.mat')
    ui = loadmat('uw_interp.mat')
    ut = loadmat('uw_space_time.mat')

    if len(args) < 2:
        variance = []

    if len(args) < 3:
        subset_ifg_index = np.array([i for i in range(len(uw['ph'][0]))])

    predef_flag = 'n'
    if 'predef_ix' in ut.keys() and len(ut['predef_ix']) > 0:
        predef_flag = 'y'

    nrow = len(uw['nzix'])
    ncol = len(uw['nzix'][0])

    x, y = np.nonzero(uw['nzix'].T)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    nzix = np.nonzero(uw['nzix'].T.flatten())[0].reshape(-1, 1)
    z = np.array([i for i in range(uw['n_ps'][0][0])])

    colix = ui['colix']
    rowix = ui['rowix']
    Z = ui['Z']

    print()
