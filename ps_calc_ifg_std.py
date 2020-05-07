import os
import numpy as np
from datetime import date

from scipy.io import loadmat, savemat
from getparm import get_parm_value as getparm

from utils import compare_objects, compare_mat_file, not_supported_value, not_supported, not_supported_param


def ps_calc_ifg_std():
    print('\nEstimating noise standard deviation (degrees)...\n')

    small_baseline_flag = getparm('small_baseline_flag')[0][0]

    psver = loadmat('psver.mat')['psver'][0][0]
    psname = 'ps' + str(psver)
    phname = 'ph' + str(psver)
    pmname = 'pm' + str(psver)
    bpname = 'bp' + str(psver)
    ifgstdname = 'ifgstd' + str(psver)

    ps = loadmat(psname + '.mat')
    pm = loadmat(pmname + '.mat')
    bp = loadmat(bpname + '.mat')

    if os.path.exists(phname + '.mat'):
        phin = loadmat(phname + '.mat')
        ph = phin['ph']
        phin.clear()
    else:
        ph = ps['ph']

    n_ps = len(ps['xy'])
    master_ix = sum(ps['master_day'] > ps['day'])[0]

    if small_baseline_flag == 'y':
        not_supported_param('small_baseline_flag', 'y')
        # ph_diff=angle(ph.*conj(pm.ph_patch).*exp(-j*(repmat(pm.K_ps,1,ps.n_ifg).*bp.bperp_mat)));
    else:
        bperp_mat = np.concatenate((bp['bperp_mat'][:, 0:ps['master_ix'][0][0] - 1], np.zeros((ps['n_ps'][0][0], 1)),
                                    bp['bperp_mat'][:, ps['master_ix'][0][0] - 1:]), axis=1)
        ph_patch = np.concatenate((pm['ph_patch'][:, 0:master_ix], np.ones((n_ps, 1)),
                                   pm['ph_patch'][:, master_ix:]), axis=1)
        ph_diff = np.angle(ph * np.conj(ph_patch) * np.exp(-1j * (
                np.tile(pm['K_ps'], (1, ps['n_ifg'][0][0])) * bperp_mat + np.tile(pm['C_ps'],
                                                                                  (1, ps['n_ifg'][0][0])))))

    ifg_std = (np.sqrt(sum(ph_diff ** 2) / n_ps) * 180 / np.pi).reshape(-1, 1)

    if small_baseline_flag == 'y':
        not_supported_param('small_baseline_flag', 'y')
        # for i=1:ps.n_ifg:
        #    fprintf('%3d %s_%s %3.2f\n',i,datestr(ps.ifgday(i,1)),datestr(ps.ifgday(i,2)),ifg_std(i))
    else:
        for i in range(ps['n_ifg'][0][0]):
            print('{} {} {}\n'.format(i + 1, date.fromordinal(ps['day'][i][0] - 366),
                                      np.round(ifg_std[i][0], decimals=2)))

    print('\n')

    ifgstd2 = {'ifg_std': ifg_std}
    savemat(ifgstdname + '.mat', ifgstd2)
