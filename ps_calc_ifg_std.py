import os, sys
import numpy as np
from datetime import date

from scipy.io import loadmat, savemat
from getparm import get_parm_value as getparm

from utils import compare_objects, compare_mat_with_number_values, not_supported_value, not_supported, not_supported_param


def ps_calc_ifg_std(*args):
    
    print('* Estimating noise standard deviation (degrees).')

    if len(args) == 1: # To debug
        path_to_task = args[0] + os.sep
    
    else: # To essential run
        path_to_task = ''

    psver = str(2)
    
    op = loadmat(path_to_task + 'parms.mat', squeeze_me = True)
    small_baseline_flag = op['small_baseline_flag']

    psname = path_to_task + 'ps' + str(psver) + '.mat'
    phname = path_to_task + 'ph' + str(psver) + '.mat'
    pmname = path_to_task + 'pm' + str(psver) + '.mat'
    bpname = path_to_task + 'bp' + str(psver) + '.mat'
    ifgstdname = path_to_task + 'ifgstd' + str(psver) + '.mat'

    ps = loadmat(psname, squeeze_me = True)
    pm = loadmat(pmname, squeeze_me = True)
    bp = loadmat(bpname, squeeze_me = True)

    if os.path.exists(phname):
        phin = loadmat(phname, squeeze_me = True)
        ph = phin['ph']
        phin.clear()
    else:
        ph = ps['ph']

    n_ps = len(ps['xy'])
    master_ix = sum(ps['master_day'] > ps['day'])

    if small_baseline_flag == 'y':
        not_supported_param('small_baseline_flag', 'y')
        sys.exit(0)
        
    else:
        bperp_mat = np.concatenate((bp['bperp_mat'][:, 0:ps['master_ix'] - 1], np.zeros((ps['n_ps'], 1)), bp['bperp_mat'][:, ps['master_ix'] - 1:]), axis = 1)
        
        ph_patch = np.concatenate((pm['ph_patch'][:, 0:master_ix], np.ones((n_ps, 1)), pm['ph_patch'][:, master_ix:]), axis = 1)
        
        a = ph * np.conj(ph_patch)
        b = np.tile(np.reshape(pm['K_ps'], (len(pm['K_ps']), 1)), (1, ps['n_ifg'])) * bperp_mat
        
        ph_diff = np.angle(a * np.exp(-1j * (b + np.tile(np.reshape(pm['C_ps'], (len(pm['C_ps']), 1)), (1, ps['n_ifg'])))))

    ifg_std = (np.sqrt(sum(ph_diff ** 2) / n_ps) * 180 / np.pi).reshape(-1, 1)

    if small_baseline_flag == 'y':
        not_supported_param('small_baseline_flag', 'y')
        sys.exit(0)
        
    else:
        for i in range(ps['n_ifg']):
            print('{} {} {}\n'.format(i + 1, date.fromordinal(ps['day'][i] - 366), np.round(ifg_std[i], decimals = 2)))

    ifgstd2 = {'ifg_std': ifg_std}
    savemat(ifgstdname, ifgstd2)
    
if __name__ == "__main__":
    # For testing
    test_path = 'C:\\Users\\Ryzen\\Documents\\PYTHON\\stampsexport'
    ps_calc_ifg_std(test_path)    