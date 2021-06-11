import os, sys, time
import shutil
from datetime import date, datetime

import numpy as np
from scipy.io import loadmat, savemat
from scipy.optimize import fmin

from ps_deramp import ps_deramp
from ps_setref import ps_setref
from ggf.matlab_funcs import lscov

def lscov_m(A, B, w = None):

    if w is None:
        Aw = A.copy()
        Bw = np.transpose(B.copy())
    else:
        W = np.sqrt(np.diag(np.array(w).flatten()))
        Aw = np.dot(W, A)
        Bw = np.dot(B.T, W)

    x, residuals, rank, s = np.linalg.lstsq(Aw, Bw.T, rcond = 1e-10)
    return(x)

def lscov_p(A, B, w):
    W = np.diag(w)
    solve = (((np.linalg.inv((A.T.dot(np.linalg.inv(W)).dot(A)))).dot(A.T)).dot(np.linalg.inv(W))).dot(B)
    return(solve)

def f(x, *args):
    d = args[0]
    G = args[1]
    return sum(abs(np.array([d]).T - np.array([G.dot(x)]).T))

def ps_calc_scla(*args):
    
    begin_time = time.time()
    
    if len(args) == 1: # To debug
        path_to_task = args[0] + os.sep
    
    else: # To essential run
        path_to_task = ''
    
    print()    
    print('******** Stage 7-1 *********')
    print('****** Estimate scla *******')
    print('')
    print('Work dir:', os.getcwd() + os.sep)

    psver = str(2) # str(loadmat(path_to_patch + 'psver.mat', squeeze_me = True)['psver'])

    op = loadmat(path_to_task + 'parms.mat', squeeze_me = True)
    
    # print('Estimating spatially-correlated look angle error...')
    
    drop_ifg_index = op['drop_ifg_index'] # Critical point: difference between indexing 0-base in python and 1-base in matlab
    scla_method = op['scla_method']
    scla_deramp = op['scla_deramp']
    subtr_tropo = op['subtr_tropo']
    scla_drop_index = op['scla_drop_index']
    coest_mean_vel = 1
    
    psname = path_to_task + 'ps' + psver + '.mat'
    bpname = path_to_task + 'bp' + psver + '.mat'
    meanvname = path_to_task + 'mv' + psver + '.mat'
    ifgstdname = path_to_task + 'ifgstd' + psver + '.mat'
    
    phuwname = path_to_task + 'phuw' + psver + '.mat'
    sclaname = path_to_task + 'scla' + psver + '.mat'

    if os.path.exists(meanvname):
        os.system('rm -f ' + meanvname)

    ps = loadmat(psname, squeeze_me = True)

    if os.path.exists(bpname):
        bp = loadmat(bpname, squeeze_me = True)
    else:
        bperp = ps['bperp']
        bperp = np.concatenate((bperp[:ps['master_ix'] - 1], bperp[ps['master_ix']:]), axis = 0)
        bp['bperp_mat'] = np.tile(bperp.T, (ps['n_ps'], 1))
    
    uw = loadmat(phuwname, squeeze_me = True)

    unwrap_ifg_index = np.setdiff1d(np.arange(0, ps['n_ifg']), drop_ifg_index)

    if subtr_tropo == 'y':
        print('* Not supported param <subtr_tropo> =', op['subtr_tropo'])
        sys.exit(0)

    if scla_deramp == 'y':
        print('* Deramping ifgs.')

        [ph_all, ph_ramp] = ps_deramp(ps.copy(), uw['ph_uw'].copy(), 1)
        uw['ph_uw'] = np.subtract(uw['ph_uw'], ph_ramp)

    else:
        ph_ramp = []

    unwrap_ifg_index = np.setdiff1d(unwrap_ifg_index, scla_drop_index)

    ref_ps = ps_setref({}, path_to_task, op)
    uw['ph_uw'] = np.subtract(uw['ph_uw'], np.tile(np.nanmean(uw['ph_uw'][ref_ps, :], 0), (ps['n_ps'], 1)))

    bperp_mat = np.append(np.append(bp['bperp_mat'][:, 0:ps['master_ix'] - 1], np.zeros((ps['n_ps'], 1)), 1), bp['bperp_mat'][:, ps['master_ix'] - 1:], 1)

    day = np.diff((ps['day'][unwrap_ifg_index]), axis = 0)
    ph = np.diff(uw['ph_uw'][:, unwrap_ifg_index], 1)
    bperp = np.diff(bperp_mat[:, unwrap_ifg_index], 1)

    bprint = np.mean(bperp, 0)
    # print('PS_CALC_SCLA: {} ifgs used in estimation:'.format(len(ph[0])))

    for i in range(len(ph[0])):
        t1 = date.fromordinal(ps['day'][unwrap_ifg_index[i]] - 366)
        t2 = date.fromordinal(ps['day'][unwrap_ifg_index[i + 1]] - 366)
        t3 = day[i]
        t4 = np.round(bprint[i])
        # print('PS_CALC_SCLA:', t1, 'to', t2, t3, 'days', t4, 'm')

    K_ps_uw = np.zeros((ps['n_ps'], 1))

    (s1, s2) = np.shape(ph)
    t1 = np.ones((s2, 1))
    t2 = np.reshape(np.mean(bperp, axis = 0), (s2, 1))
    t3 = np.reshape(day, (s2, 1))
    
    if coest_mean_vel == 0 or len(unwrap_ifg_index) < 4:
        G = np.hstack((t1, t2))
    else:
        G = np.hstack((t1, t2, t3))

    ifg_vcm = np.eye(ps['n_ifg'])

    if os.path.exists(ifgstdname):
        ifgstd = loadmat(ifgstdname, squeeze_me = True)
        ifg_vcm = np.diag(ifgstd['ifg_std'] * np.pi / 180) ** 2

    ifg_vcm_use = np.ones(s2)
    
    m = lscov(G, ph.T, ifg_vcm_use)
    m = np.reshape(m, (3, len(m) // 3))
    K_ps_uw = m.T[:,1].flatten()
    
    if scla_method == 'L1':
        print('* SCLA Method L1-norm.')
        for i in range(ps['n_ps']):
            d = ph[i, :]
            m2 = m[:, i]
            m2 = fmin(f, m2, args = (d, G), disp = False)
            K_ps_uw[i] = m2[1]

    (s1, s2) = np.shape(bperp_mat)
    ph_scla = np.tile(np.reshape(K_ps_uw, (s1, 1)), (1, s2)) * bperp_mat

    unwrap_ifg_index = np.setdiff1d(unwrap_ifg_index, ps['master_ix'] - 1)
    if coest_mean_vel == 0:
        C_ps_uw = np.array([np.mean(uw['ph_uw'][:, unwrap_ifg_index] - ph_scla[:, unwrap_ifg_index], 1)]).T
    else:
        t1 = np.reshape(np.ones(len(unwrap_ifg_index)), (len(unwrap_ifg_index), 1))
        t2 = np.reshape(ps['day'][unwrap_ifg_index] - ps['day'][ps['master_ix'] - 1], (len(unwrap_ifg_index), 1))
        G = np.hstack((t1, t2))
        
        B = np.transpose(np.array(uw['ph_uw'][:, unwrap_ifg_index] - ph_scla[:, unwrap_ifg_index]))
        v = (ifgstd['ifg_std'][unwrap_ifg_index] * np.pi / 180) ** 2
        
        m = lscov_p(G, B, v).T
        C_ps_uw = m[:,0].flatten()

    scla = {
        'ph_scla': ph_scla,
        'K_ps_uw': K_ps_uw,
        'C_ps_uw': C_ps_uw,
        'ph_ramp': ph_ramp,
        'ifg_vcm': ifg_vcm}  

    if os.path.exists(sclaname):
        olddatenum = os.path.getmtime(sclaname)
        shutil.move(sclaname, sclaname + datetime.fromtimestamp(olddatenum).strftime('_%Y%m%d_%H%M%S'))
        
    savemat(sclaname, scla, oned_as = 'column')
    
    print('Done at', int(time.time() - begin_time), 'sec')
    
if __name__ == "__main__":
    # For testing
    test_path = 'C:\\Users\\Ryzen\\Documents\\PYTHON\\stampsexport'
    ps_calc_scla(test_path)