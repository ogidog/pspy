import os, sys, time
import numpy as np

from scipy.io import loadmat, savemat
from getparm import get_parm_value as getparm
from uw_3d import uw_3d

from utils import compare_objects, not_supported_param, compare_mat_with_number_values

def tr(matrix):
    out = np.transpose(matrix)
    return(out)

def v2c(v):
    m = len(v)
    out = np.reshape(v, (m, 1))
    return(out)

def v2r(v):
    m = len(v)
    out = np.reshape(v, (1, m))
    return(out)

def ps_unwrap(*args):
    
    begin_time = time.time()
    
    if len(args) == 1: # To debug
        path_to_task = args[0] + os.sep
    
    else: # To essential run
        path_to_task = ''
    
    print()    
    print('********** Stage 6 *********')
    print('****** Unwrap phases *******')
    print('')
    print('Work dir:', os.getcwd() + os.sep)
    
    op = loadmat(path_to_task + 'parms.mat', squeeze_me = True)

    psver = str(2) # str(loadmat(path_to_patch + 'psver.mat', squeeze_me = True)['psver'])

    small_baseline_flag = op['small_baseline_flag']
    unwrap_patch_phase = op['unwrap_patch_phase']
    scla_deramp = op['scla_deramp']
    subtr_tropo = op['subtr_tropo']
    aps_name = op['tropo_method']

    psname = path_to_task + 'ps' + psver + '.mat'
    rcname = path_to_task + 'rc' + psver + '.mat'
    pmname = path_to_task + 'pm' + psver + '.mat'
    bpname = path_to_task + 'bp' + psver + '.mat'
    goodname = path_to_task + 'phuw_good' + psver + '.mat'

    if small_baseline_flag != 'y':
        sclaname = path_to_task + 'scla_smooth' + psver + '.mat'
        apsname = path_to_task + 'tca' + psver + '.mat'
        phuwname = path_to_task + 'phuw' + psver + '.mat'
    else:
        print('* Not supported param [small_baseline_flag] = ', op['small_baseline_flag'])
        sys.exit(0)
        
    ps = loadmat(psname, squeeze_me = True);

    drop_ifg_index = op['drop_ifg_index']
    unwrap_ifg_index = np.setdiff1d(np.arange(ps['n_ifg']), drop_ifg_index - 1)

    if os.path.exists(bpname):
        bp = loadmat(bpname, squeeze_me = True)
    else:
        bperp = ps['bperp']
        if small_baseline_flag != 'y':
            bperp = np.concatenate((bperp[:ps['master_ix']], bperp[ps['master_ix']:]), axis = 0)
        
        bp['bperp_mat'] = np.tile(bperp.T, (ps['n_ps'], 1))

    if small_baseline_flag != 'y':
        bperp_mat = np.concatenate((bp['bperp_mat'][:, :ps['master_ix'] - 1], np.zeros(ps['n_ps']).reshape(-1, 1), bp['bperp_mat'][:, ps['master_ix'] - 1:]), axis = 1)
    else:
        sys.exit(0)

    if unwrap_patch_phase == 'y':
        pm = loadmat(pmname, squeeze_me = True)
        ph_w = pm['ph_patch'] / np.abs(pm['ph_patch'])

        if small_baseline_flag != 'y':
            ph_w = np.concatenate((ph_w[:, :ps['master_ix'] - 1], np.ones((ps['n_ps'], 1)), ph_w[:, ps['master_ix'] - 1:]), axis = 1)
    else:
        rc = loadmat(rcname, squeeze_me = True)
        ph_w = rc['ph_rc']

        if os.path.exists(pmname):
            pm = loadmat(pmname, squeeze_me = True)
            if 'K_ps' in pm.keys():
                if len(pm['K_ps']) != 0:
                    ph_w = np.multiply(ph_w, np.exp(np.multiply(complex(0.0, 1.0) * np.tile(v2c(pm['K_ps']), (1, ps['n_ifg'])), bperp_mat)))

    ix = np.array([ph_w[i, :] != 0 for i in range(len(ph_w))])
    ph_w[ix] = ph_w[ix] / np.abs(ph_w[ix])

    scla_subtracted_sw = 0
    ramp_subtracted_sw = 0

    options = {'master_day': ps['master_day']}
    unwrap_hold_good_values = op['unwrap_hold_good_values']
    if small_baseline_flag != 'y' or os.path.exists(phuwname):
        unwrap_hold_good_values = 'n';
        print('* Code to hold good values skipped')

    if unwrap_hold_good_values == 'y':
        print('* Not supported param [unwrap_hold_good_values] = ', unwrap_hold_good_values)
        sys.exit(0)

    if small_baseline_flag != 'y' and os.path.exists(sclaname):
        print('* Subtracting scla and master aoe...')
        scla = loadmat(sclaname, squeeze_me = True)
        if len(scla['K_ps_uw']) == ps['n_ps']:
            scla_subtracted_sw = 1
            ph_w = ph_w * np.exp(np.multiply(complex(0.0, -1.0) * np.tile(v2c(scla['K_ps_uw']), (1, ps['n_ifg'])), bperp_mat))
            ph_w = np.multiply(ph_w, np.tile(np.exp(complex(0.0, -1.0) * v2c(scla['C_ps_uw'])), (1, ps['n_ifg'])))

            if scla_deramp == 'y' and 'ph_ramp' in scla.keys() and len(scla['ph_ramp']) == ps['n_ps']:
                ramp_subtracted_sw = 1
                ph_w = ph_w * np.exp(complex(0.0, -1.0) * scla['ph_ramp'])
        else:
            print('* Wrong number of PS in scla - subtraction skipped')
            os.remove(sclaname)

    if small_baseline_flag == 'y' and os.path.exists(sclaname):
        print('* Not supported param [small_baseline_flag] = ', small_baseline_flag)
        sys.exit(0)

    if os.path.exists(apsname) and subtr_tropo == 'y':
        print('* Not supported param [subtr_tropo] = ', subtr_tropo)
        sys.exit(0)

    options['time_win'] = op['unwrap_time_win']
    options['unwrap_method'] = op['unwrap_method']
    options['grid_size'] = op['unwrap_grid_size']
    options['prefilt_win'] = op['unwrap_gold_n_win']
    options['goldfilt_flag'] = op['unwrap_prefilter_flag']
    options['gold_alpha'] = op['unwrap_gold_alpha']
    options['la_flag'] = op['unwrap_la_error_flag']
    options['scf_flag'] = op['unwrap_spatial_cost_func_flag']

    max_topo_err = op['max_topo_err']
    _lambda = op['lambda']

    rho = 830000
    if 'mean_incidence' in ps.keys():
        inc_mean = ps['mean_incidence']
    else:
        laname = path_to_task + 'la' + str(psver) + '.mat'
        if os.path.exists(laname):
            la = loadmat(laname, squeeze_me = True)
            inc_mean = np.mean(la['la']) + 0.052
        else:
            inc_mean = 21 * np.pi / 180.0
    max_K = max_topo_err / (_lambda * rho * np.sin(inc_mean) / 4 / np.pi)

    bperp_range = np.amax(ps['bperp']) - np.amin(ps['bperp'])
    options['n_trial_wraps'] = (bperp_range * max_K / (2 * np.pi))
    print('* Value n_trial_wraps = ', options['n_trial_wraps'])

    if small_baseline_flag == 'y':
        sys.exit(0)

    else:
        lowfilt_flag = 'n'
        ifgday_ix = np.concatenate((np.ones((ps['n_ifg'], 1)) * ps['master_ix'], np.array([x for x in range(ps['n_ifg'])]).reshape(-1, 1)), axis = 1).astype('int')
        master_ix = np.sum(ps['master_day'] > ps['day']) + 1
        unwrap_ifg_index = np.setdiff1d(unwrap_ifg_index, master_ix - 1)
        day = ps['day'] - ps['master_day']

    if unwrap_hold_good_values == 'y':
        sys.exit(0)

    v1 = ph_w[:, unwrap_ifg_index]
    v2 = ps['xy']
    v3 = v2c(day)
    v4 = ifgday_ix[unwrap_ifg_index, :]
    v5 = ps['bperp'][unwrap_ifg_index]
    ph_uw_some, msd_some = uw_3d(v1, v2, v3, v4, v5, options)

    ph_uw = np.zeros((ps['n_ps'], ps['n_ifg']))
    msd = np.zeros((ps['n_ifg'], 1))
    ph_uw[:, unwrap_ifg_index] = ph_uw_some
    msd[unwrap_ifg_index] = msd_some

    if scla_subtracted_sw == 1 and small_baseline_flag != 'y':
        print('* Adding back SCLA and master AOE')
        scla = loadmat(sclaname, squeeze_me = True)
        ph_uw = ph_uw + (np.multiply(np.tile(v2c(scla['K_ps_uw']), (1, ps['n_ifg'])), bperp_mat))
        ph_uw = ph_uw + np.tile(v2c(scla['C_ps_uw']), (1, ps['n_ifg']))
        if ramp_subtracted_sw:
            ph_uw = ph_uw + scla['ph_ramp']

    if scla_subtracted_sw == 1 and small_baseline_flag == 'y':
        not_supported_param('small_baseline_flag', 'y')

    if os.path.exists(apsname) and subtr_tropo == 'y':
        not_supported_param('subtr_tropo', 'y')

    if unwrap_patch_phase == 'y':
        not_supported_param('unwrap_patch_phase', 'y')

    ph_uw[:, np.setdiff1d(np.array([*range(ps['n_ifg'])]), unwrap_ifg_index)] = 0

    savemat(phuwname, {'ph_uw': ph_uw, 'msd': msd})

    print('Done at', int(time.time() - begin_time), 'sec')

if __name__ == "__main__":
    # For testing
    test_path = 'C:\\Users\\Ryzen\\Documents\\PYTHON\\stampsexport'
    ps_unwrap(test_path)    