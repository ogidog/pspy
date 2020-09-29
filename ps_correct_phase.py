import os
import numpy as np

from scipy.io import loadmat, savemat
from utils import not_supported_param, compare_mat_with_number_values, compare_objects

from getparm import get_parm_value as getparm


def ps_correct_phase():
    print('Correcting phase for look angle error...\n')

    small_baseline_flag = getparm('small_baseline_flag')[0][0]

    psver = loadmat('psver.mat')['psver'][0][0]
    psname = 'ps' + str(psver)
    phname = 'ph' + str(psver)
    pmname = 'pm' + str(psver)
    rcname = 'rc' + str(psver)
    bpname = 'bp' + str(psver)

    ps = loadmat(psname + '.mat')
    pm = loadmat(pmname + '.mat')
    bp = loadmat(bpname + '.mat')

    if os.path.exists(phname + '.mat'):
        phin = loadmat(phname)
        ph = phin['ph']
        phin = {}
    else:
        ph = ps['ph']

    K_ps = pm['K_ps']
    C_ps = pm['C_ps']
    master_ix = (sum(ps['master_day'] > ps['day']) + 1)[0] - 1

    if small_baseline_flag == 'y':
        not_supported_param('small_baseline_flag', 'y')
        # %ph_rc=ph.*exp(-j*(repmat(K_ps,1,ps.n_ifg).*bp.bperp_mat));  % subtract range error
        # ph_rc=ph.*exp(-j*(repmat(K_ps,1,ps.n_ifg).*bp.bperp_mat));  % subtract range error
        # save(rcname,'ph_rc');
    else:
        bperp_mat = np.concatenate((bp['bperp_mat'][:, 0:ps['master_ix'][0][0] - 1], np.zeros((ps['n_ps'][0][0], 1)),
                                    bp['bperp_mat'][:, ps['master_ix'][0][0] - 1:]), axis=1)
        ph_rc = np.multiply(ph,
                            (np.exp(
                                complex(0, -1) * np.add(np.multiply(np.tile(K_ps, (1, ps['n_ifg'][0][0])), bperp_mat),
                                                        np.tile(C_ps, (1, ps['n_ifg'][0][0]))))))
        ph_reref = np.concatenate((pm['ph_patch'][:, 0:master_ix], np.ones((ps['n_ps'][0][0], 1)),
                                   pm['ph_patch'][:, master_ix:]), axis=1)
        rc2 = {'ph_rc': ph_rc, 'ph_reref': ph_reref}
        savemat(rcname + '.mat', rc2)
