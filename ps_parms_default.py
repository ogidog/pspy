import os, sys
from datetime import datetime
from scipy.io import loadmat


def ps_parms_default():
    parmfile = 'parms.mat';
    parent_flag = 0;

    if os.path.exists('./parms.mat'):
        parms = loadmat(parmfile);
    else:
        if os.path.exists('../parms.mat'):
            parmfile = '../parms.mat';
            parms = loadmat(parmfile);
            parent_flag = 1;
        else:
            parms = {}
            parms['Created'] = datetime.today().strftime('%Y-%m-%d')
            parms['small_baseline_flag'] = 'n'

    parmfields_before = parms.keys();
    num_fields = len(parmfields_before);

    if 'max_topo_err' not in parmfields_before:
        parms['max_topo_err'] = 20

    if 'quick_est_gamma_flag' not in parmfields_before:
        parms['quick_est_gamma_flag'] = 'y'

    if 'select_reest_gamma_flag' not in parmfields_before:
        parms['select_reest_gamma_flag'] = 'y'

    if 'filter_grid_size' not in parmfields_before:
        parms['filter_grid_size'] = 50

    if 'filter_weighting' not in parmfields_before:
        parms['filter_weighting'] = 'P-square'

    if 'gamma_change_convergence' not in parmfields_before:
        parms['gamma_change_convergence'] = 0.005

    if 'gamma_max_iterations' not in parmfields_before:
        parms['gamma_max_iterations'] = 3

    if 'slc_osf' not in parmfields_before:
        parms['slc_osf'] = 1

    if 'slc_osf' not in parmfields_before:
        parms['slc_osf'] = 1

    if 'clap_win' not in parmfields_before:
        parms['clap_win'] = 32

    if 'clap_low_pass_wavelength' not in parmfields_before:
        parms['clap_low_pass_wavelength'] = 800



    # TODO: save

sys.exit(0)
