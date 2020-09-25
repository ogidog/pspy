import os, sys
import numpy as np
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

    if 'clap_alpha' not in parmfields_before:
        parms['clap_alpha'] = 1

    if 'clap_beta' not in parmfields_before:
        parms['clap_beta'] = 0.3

    if 'select_method' not in parmfields_before:
        parms['select_method'] = 'DENSITY'

    if 'density_rand' not in parmfields_before:
        if parms['small_baseline_flag'] == 'y':
            parms['density_rand'] = 2
        else:
            parms['density_rand'] = 20

    if 'percent_rand' not in parmfields_before:
        if parms['small_baseline_flag'] == 'y':
            parms['percent_rand'] = 1
        else:
            parms['percent_rand'] = 20

    if 'gamma_stdev_reject' not in parmfields_before:
        parms['gamma_stdev_reject'] = 0

    if 'weed_alpha' in parmfields_before:
        del parms['weed_alpha']

    if 'weed_time_win' not in parmfields_before:
        parms['weed_time_win'] = 730

    if 'weed_max_noise' not in parmfields_before:
        parms['weed_max_noise'] = float('inf')

    if 'weed_standard_dev' not in parmfields_before:
        if parms['small_baseline_flag'] == 'y':
            parms['weed_standard_dev'] = float('inf')
        else:
            parms['weed_standard_dev'] = 1.0

    if 'weed_zero_elevation' not in parmfields_before:
        parms['weed_zero_elevation'] = 'n'

    if 'weed_neighbours' not in parmfields_before:
        parms['weed_neighbours'] = 'n'

    if 'unwrap_method' not in parmfields_before:
        if parms['small_baseline_flag'] == 'y':
            parms['unwrap_method'] = '3D_QUICK'
        else:
            parms['unwrap_method'] = '3D'

    if 'unwrap_patch_phase' not in parmfields_before:
        parms['unwrap_patch_phase'] = 'n'

    if 'unwrap_ifg_index' in parmfields_before:
        try:
            ps = loadmat('ps2.mat');
        except:
            try:
                ps = loadmat('ps1.mat');
            except:
                print('')

        if 'ps' in globals().keys() and parms['unwrap_ifg_index'] != 'all':
            parms['drop_ifg_index'] = np.setdiff1d([*range(ps['n_ifg'])], parms['unwrap_ifg_index'])

        del parms['unwrap_ifg_index']
        num_fields = 0

    if 'drop_ifg_index' not in parmfields_before:
        parms['drop_ifg_index'] = []

    if 'unwrap_la_error_flag' not in parmfields_before:
        parms['unwrap_la_error_flag'] = 'y'

    if 'unwrap_spatial_cost_func_flag' not in parmfields_before:
        parms['unwrap_spatial_cost_func_flag'] = 'n'

    if 'unwrap_prefilter_flag' not in parmfields_before:
        parms['unwrap_prefilter_flag'] = 'y'

    if 'unwrap_grid_size' not in parmfields_before:
        parms['unwrap_grid_size'] = 200

    if 'unwrap_gold_n_win' not in parmfields_before:
        parms['unwrap_gold_n_win'] = 32

    if 'unwrap_alpha' not in parmfields_before:
        parms['unwrap_alpha'] = 8

    if 'unwrap_time_win' not in parmfields_before:
        parms['unwrap_time_win'] = 730

    if 'unwrap_gold_alpha' not in parmfields_before:
        parms['unwrap_gold_alpha'] = 0.8

    if 'unwrap_hold_good_values' not in parmfields_before:
        parms['unwrap_hold_good_values'] = 'n'

    if 'recalc_index' in parmfields_before:
        try:
            ps = loadmat('ps2.mat');
        except:
            try:
                ps = loadmat('ps1.mat')
            except:
                print()
        if 'ps' in globals().keys() and parms['recalc_index'] != 'all':
            if parms['small_baseline_flag'] == 'y':
                parms['scla_drop_index'] = np.setdiff1d([*range(ps['n_image'])], parms['recalc_index'])
            else:
                parms['scla_drop_index'] = np.setdiff1d([*range(ps['n_ifg'])], parms['recalc_index'])
        del parms['recalc_index']

        if 'sb_recalc_index' in parmfields_before:
            if 'ps' in globals().keys() and parms['sb_recalc_index'] != 'all':
                parms['sb_scla_drop_index'] = np.setdiff1d([*range(ps['n_ifg'])], parms['sb_recalc_index'])
            del parms['sb_recalc_index']
        num_fields = 0

    if 'scla_drop_index' not in parmfields_before:
        parms['scla_drop_index'] = []

    if 'scn_wavelength' not in parmfields_before:
        parms['scn_wavelength'] = 100

    if 'scn_time_win' not in parmfields_before:
        parms['scn_time_win'] = 365

    if 'scn_deramp_ifg' not in parmfields_before:
        parms['scn_deramp_ifg'] = []

    if 'scn_kriging_flag' not in parmfields_before:
        parms['scn_kriging_flag'] = 'n'

    if 'ref_lon' not in parmfields_before:
        parms['ref_lon'] = [float('-inf'), float('inf')]

    # TODO: save

# sys.exit(0)
