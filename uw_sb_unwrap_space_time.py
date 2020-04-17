import numpy as np

from scipy.io import loadmat, savemat
from utils import *


def uw_sb_unwrap_space_time(day, ifgday_ix, unwrap_method, time_win, la_flag, bperp, n_trial_wraps, prefilt_win,
                            scf_flag, temp, n_temp_wraps, max_bperp_for_temp_est):
    print('\nUnwrapping in time-space...')

    uw = loadmat('uw_grid.mat');
    ui = loadmat('uw_interp.mat');

    n_ifg = uw['n_ifg'][0][0]
    n_ps = uw['n_ps'][0][0]
    nzix = uw['nzix']
    ij = uw['ij']

    if 'ph_uw_predef' in uw.keys():
        predef_flag = 'n'
    else:
        predef_flag = 'y'

    n_image = len(day)
    master_ix = np.where(day == 0)[0][0]
    nrow, ncol = np.shape(ui['Z'])

    day_pos_ix = np.where(day > 0)
    tempdummy = np.min(day[day_pos_ix])
    I = np.argmin(day[day_pos_ix])
    dph_space = np.multiply(uw['ph'][ui['edgs'][:, 2] - 1, :], np.conj(uw['ph'][ui['edgs'][:, 1] - 1, :]))
    if predef_flag == 'y':
        not_supported_param(predef_flag, 'y')
        # dph_space_uw=uw.ph_uw_predef(ui.edgs(:,3),:)-uw.ph_uw_predef(ui.edgs(:,2),:);
        # predef_ix=~isnan(dph_space_uw);
        # dph_space_uw=dph_space_uw(predef_ix);
    else:
        predef_ix = []

    uw.clear()
    tempdummy = -1

    dph_space = dph_space / np.abs(dph_space)

    ifreq_ij = [];
    jfreq_ij = [];

    # TODO: убрать
    # diff = compare_complex_objects(dph_space, 'dph_space')

    print()
