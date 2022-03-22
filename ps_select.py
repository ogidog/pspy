#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:06:44 2020

@author: anyuser
"""

import os
# import sys
import time
import numpy as np

from multiprocessing import Pool

from scipy.io import loadmat, savemat
from scipy.signal import convolve2d
from scipy.signal.windows import gaussian
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from numpy.linalg import solve, lstsq

def v2c(v):
    m = len(v)
    out = np.reshape(v, (m, 1))
    return(out)

def v2r(v):
    m = len(v)
    out = np.reshape(v, (1, m))
    return(out)

def ps_topofit_orig(args):
    
    cpxphase = args[0].flatten() 
    bperp =    args[1].flatten() 
    n_trial_wraps = args[2]
    asym = 0

    ix = cpxphase != 0  # if signal of one image is 0, dph set to 0
    ix = ix.flatten()
    cpxphase = cpxphase[ix]
    
    bperp = bperp[ix]
    bperp_range = np.max(bperp) - np.min(bperp)
    
    trial_mult = np.arange(-np.ceil(8 * n_trial_wraps), np.ceil(8 * n_trial_wraps) + 1) + asym * 8 * n_trial_wraps
    n_trials = len(trial_mult)
    trial_phase = bperp / bperp_range * np.pi / 4
    trial_phase_mat = np.exp(np.matmul(-1j * np.reshape(trial_phase, (len(trial_phase.flatten()),1)), np.reshape(trial_mult, (1,len(trial_mult.flatten())))))
    cpxphase_mat = np.tile(np.reshape(cpxphase, (len(cpxphase),1)), (1, n_trials))
    phaser = trial_phase_mat * cpxphase_mat
    phaser_sum = np.sum(phaser, axis=0)
    C_trial = np.angle(phaser_sum)
    coh_trial = np.abs(phaser_sum) / np.sum(np.abs(cpxphase))

    coh_high_max_ix = np.argmax(coh_trial)  # only select highest

    K0 = np.pi / 4 / bperp_range * trial_mult[coh_high_max_ix]
    C0 = C_trial[coh_high_max_ix]
    coh0 = coh_trial[coh_high_max_ix]

    # linearise and solve
    resphase = cpxphase * np.exp(-1j * (K0 * bperp))  # subtract approximate fit
    offset_phase = np.sum(resphase)
    resphase = np.angle(resphase * np.conj(offset_phase))  # subtract offset, take angle (unweighted)
    weighting = np.abs(cpxphase)
    a = np.reshape(weighting * bperp, (len(bperp), 1))
    b = weighting * resphase
    
    shape_a = np.shape(a)
    if len(shape_a) == 2:
        if shape_a[0] == shape_a[1]:
            m = solve(a, b)
        else:
            m = lstsq(a, b, rcond = None)
    else:
        m = lstsq(a, b, rcond = None)
    mopt = m[0][0]
    
    K0 = K0 + mopt
    phase_residual = cpxphase * np.exp(-1j * (K0 * bperp))
    mean_phase_residual = np.sum(phase_residual)
    C0 = np.angle(mean_phase_residual)  # static offset (due to noise of master + average noise of rest)
    coh0 = np.abs(mean_phase_residual) / np.sum(np.abs(phase_residual))

    return(K0, C0, coh0, phase_residual)

def hist(data):
    bins = np.arange(0.005, 1.01, 0.01)
    n = len(bins)
    out = np.zeros(n)
    for i in range(n):
        if i > 0:
            if i < n-1:
                up = (bins[i] + bins[i+1]) / 2
                dn = (bins[i-1] + bins[i]) / 2
                subdata = data[data < up]
                out[i] = len(subdata[subdata >= dn])
            else:
                dn = (bins[i-1] + bins[i]) / 2
                out[i] = len(data[data >= dn])
        else:
            up = (bins[i] + bins[i+1]) / 2
            out[i] = len(data[data < up])
    return(out)

def shiftdim(x, n = None, nargout = 2):
    outsel = slice(nargout) if nargout > 1 else 0
    x = np.asarray(x)
    s = x.shape
    m = next((i for i, v in enumerate(s) if v > 1), 0)
    if n is None:
        n = m
    if n > 0:
        n = n % x.ndim
    if n > 0:
        if n <= m:
            x = x.reshape(s[n:])
        else:
            x = x.transpose(np.roll(range(x.ndim), -n))
    elif n < 0:
            x = x.reshape((1,)*(-n)+x.shape)
            
    return(x, n)[outsel]

###
def filter2(k, a):
    return(convolve2d(a, np.rot90(k, 2), mode = 'same'))

def clap_filt_p(cph, alpha, beta, low_pass):
    cph[np.isnan(cph)] = 0
    gw = np.reshape(gaussian(7, 1.2), (7, 1))
    B = np.matmul(gw, gw.T)
    ph_fft = fft2(cph)
    H = np.abs(ph_fft)
    H = ifftshift(filter2(B, fftshift(H)))
    meanH = np.median(H)

    if meanH != 0:
        H = H / meanH
    
    H = H**alpha
    H = H - 1
    H[H < 0] = 0
    G = H * beta + low_pass
    ph_out = ifft2(ph_fft * G)
    return(ph_out)

def reest(inlist):
    ps_ij = inlist[0]
    ph_grid = inlist[1]
    opt = inlist[2]
    
    slc_osf = opt['slc_osf']
    clap_alpha = opt['clap_alpha']
    clap_beta = opt['clap_beta']
    low_pass = opt['low_pass']
    n_i = opt['n_i']
    n_j = opt['n_j']
    n_ifg = opt['n_ifg']
    n_win = opt['n_win']
    ph_filt = np.zeros((n_win, n_win, n_ifg), dtype = np.complex64)     

    i_min = ps_ij[0] - n_win // 2
    i_min = i_min if i_min > 1 else 1
    i_max = i_min + n_win - 1
    if i_max > n_i:
        i_min = i_min - i_max + n_i
        i_max = n_i
    
    j_min = ps_ij[1] - n_win // 2
    j_min = j_min if j_min > 1 else 1
    j_max = j_min + n_win - 1
    if j_max > n_j:
        j_min = j_min - j_max + n_j
        j_max = n_j
    
    if (j_min < 1) or (i_min < 1):
        out = np.zeros(n_ifg)
    else:
        ps_bit_i = ps_ij[0] - i_min + 1
        ps_bit_j = ps_ij[1] - j_min + 1
        
        ph_bit = ph_grid[i_min - 1 : i_max, j_min - 1 : j_max, :].copy()
        ph_bit[ps_bit_i - 1, ps_bit_j - 1, :] = 0

        ix_i = np.arange(ps_bit_i - slc_osf + 1, ps_bit_i + slc_osf)
        ix_i = ix_i[(ix_i > 0) & (ix_i <= i_max - i_min + 1)]
        
        ix_j = np.arange(ps_bit_j - slc_osf + 1, ps_bit_j + slc_osf)
        ix_j = ix_j[(ix_j > 0) & (ix_j <= j_max - j_min + 1)]
        
        ph_bit[ix_i - 1, ix_j - 1] = 0

        for ii in range(n_ifg):
            ph_filt[:, :, ii] = clap_filt_p(ph_bit[:, :, ii], clap_alpha, clap_beta, low_pass)
                         
        out = ph_filt[ps_bit_i - 1, ps_bit_j - 1, :].flatten()            
        ph_filt = ph_filt * 0
                
    return(out)

##########################
def ps_select(*args):
    
    begin_time = time.time()

    if len(args) == 3: # To debug
        path_to_task = args[0] + os.sep
        path_to_patch = path_to_task + 'PATCH_' + str(args[1]) + os.sep
    
    else: # To essential run
        path_to_task = os.sep.join(os.getcwd().split(os.path.sep)[:-1]) + os.sep
        path_to_patch = os.getcwd() + os.sep
    
    print()    
    print('********** Stage 3 *********')
    print('********** Select **********')
    print('')
    print('Work dir:', path_to_patch)

    psver = str(1) # str(loadmat(path_to_patch + 'psver.mat', squeeze_me = True)['psver'])

    runmode = 'single'        
    if len(args) > 0:
        if 'mpi' in args:
            runmode = 'mpi'
            
    op = loadmat(path_to_task + 'parms.mat', squeeze_me = True)
    
    datanames = ['ps', 'ph', 'bp', 'la', 'da', 'inc', 'pm']
    
    for fname in datanames:
        fnpath = path_to_patch + fname + psver + '.mat'
        if os.path.exists(fnpath):
            data = loadmat(fnpath, squeeze_me = True)
            for field in list(data):
                if field[:2] != '__':
                    op[field] = data[field]
                
    print('* Pixels loaded:', len(op['xy']))
    
    select_method = op['select_method']
    
    if select_method == 'PERCENT':
        max_percent_rand = op['percent_rand']
    else:
        max_density_rand = op['density_rand']
    
    low_coh_thresh = 31
    
    ph = op['ph']
    
    bperp = op['bperp']
    n_ifg = op['n_ifg']
    ifg_index =  np.setdiff1d(np.arange(n_ifg), op['drop_ifg_index'] - 1)
    
    # if small_baseline_flag != 'y':
    master_ix = np.sum(op['master_day'] > op['day'])
    no_master_ix = np.setdiff1d(np.arange(n_ifg), op['master_ix'] - 1)
    ifg_index = np.setdiff1d(ifg_index, op['master_ix'] - 1)
    
    for i in range(len(ifg_index)):
        if ifg_index[i] > master_ix:
            ifg_index[i]  = ifg_index[i] - 1
    
    ph = ph[:, no_master_ix]
    bperp = bperp[no_master_ix]
    n_ifg = len(no_master_ix)
    
    n_ps = op['n_ps']
    xy = op['xy'][:,1:]

    # lonlat = op['lonlat']
    # idxll = np.arange(n_ps)
    
    if 'D_A' in op:
        D_A = op['D_A']
    else:
        D_A = np.ones(1)
    
    if len(D_A) >= 10000:
        D_A_sort = np.sort(D_A)
        if len(D_A) >= 50000:
            bin_size = 10000
        else:
            bin_size = 2000
        damax = [0]
        for i in range(1, len(D_A) // bin_size):
            damax.append(D_A_sort[i * bin_size])
        damax.append(D_A_sort[-1])
    else:
        damax = np.asarray([0, 1])
        D_A = np.ones(len(op['coh_ps']))
    
    D_A_max = np.asarray(damax, dtype = np.float)
    
    if op['select_method'] != 'PERCENT':
        patch_area = (xy[:,0].max() - xy[:,0].min()) * (xy[:,1].max() - xy[:,1].min()) / 1000000
        max_percent_rand = max_density_rand * patch_area / (len(D_A_max) - 1)
    
    #print(D_A_max)
    min_coh  = np.zeros(len(D_A_max) - 1)
    D_A_mean = np.zeros(len(D_A_max) - 1)
    Nr_dist = op['Nr']
    coh_ps = op['coh_ps']
    
    for i in range(len(D_A_max) - 1):
        idx = np.argwhere((D_A > D_A_max[i]) & (D_A <= D_A_max[i+1])).flatten()
        coh_chunk = coh_ps[idx]
        D_A_mean[i] = np.mean(D_A[idx])
        coh_chunk = coh_chunk[coh_chunk != 0]
        Na = hist(coh_chunk)
        Nr = Nr_dist * np.sum(Na[:low_coh_thresh]) / np.sum(Nr_dist[:low_coh_thresh])
        Na[Na == 0] = 1
    
        if op['select_method'] == 'PERCENT':
            percent_rand = np.cumsum(np.flip(Nr)) / np.cumsum(np.flip(Na)) * 100
        else:
            percent_rand = np.cumsum(np.flip(Nr))
        
        percent_rand = np.flip(percent_rand)
        ok_ix = np.argwhere(percent_rand < max_percent_rand)
        
        if len(ok_ix) == 0:
            min_coh[i] = 1
        else:
            min_fit_ix = np.min(ok_ix) - 3
            if min_fit_ix <= 0:
                #print(min_coh, i)
                min_coh[i] = low_coh_thresh / 100
            else:
                max_fit_ix = np.min(ok_ix) + 2
                max_fit_ix = max_fit_ix if max_fit_ix < 100 else 100
                xp = percent_rand[min_fit_ix:max_fit_ix + 1]
                yp = np.arange(min_fit_ix + 1, max_fit_ix + 2) / 100
                p = np.polyfit(xp, yp, 3)
                min_coh[i] = np.polyval(p, max_percent_rand)

    nonnanix = np.invert(np.isnan(min_coh))
        
    if len(nonnanix == True) == 0:
        print('* No suitable data, using default threshold of 0.3')
        coh_thresh = 0.3
        coh_thresh_coeffs = []
    else:
        min_coh = min_coh[nonnanix]
        D_A_mean = D_A_mean[nonnanix]
        
        if len(min_coh) > 1:
            coh_thresh_coeffs = np.polyfit(D_A_mean, min_coh, 1)
            if coh_thresh_coeffs[0] > 0:
                coh_thresh = np.polyval(coh_thresh_coeffs, D_A)
            else:
                coh_thresh = np.polyval(coh_thresh_coeffs, 0.35)
                coh_thresh_coeffs = [];
        else:
            coh_thresh = min_coh
            coh_thresh_coeffs = []
    
    coh_thresh[coh_thresh < 0] = 0
    
    ix = np.argwhere(coh_ps > coh_thresh).flatten()

    n_ps = len(ix)
    
    print('* Pixels selected as ps initially:', n_ps)
    ###########################################################################
    if op['gamma_stdev_reject'] > 0:
        ph_res_cpx = np.exp(1j * op['ph_res'][:,ifg_index])
        coh_std = np.zeros(len(ix))
        np.random.seed()
        
        for i in range(len(ix)):
            count = len(ifg_index)
            bootdata = ph_res_cpx[ix[i], ifg_index]
            smpl = np.zeros(count)
            for t in range(100):
                randdata = np.random.choice(bootdata, count)
                smpl[t] = np.abs(np.sum(randdata)) / len(randdata)
            
            coh_std[i] = np.std(smpl)
    
        ix = ix[coh_std < op['gamma_stdev_reject']]
        n_ps = len(ix)
    
    if len(coh_thresh) > 1:
        coh_thresh = coh_thresh[ix]
    
    ph_patch2 = np.zeros((n_ps, n_ifg), dtype = complex)
    ph_res2 = np.zeros((n_ps, n_ifg))
    ph = ph[ix, :]
    grid_ij = np.asarray(op['grid_ij'], dtype = int)
    K_ps2 = np.zeros(n_ps)
    C_ps2 = np.zeros(n_ps)
    coh_ps2 = np.zeros(n_ps)

    opt = dict()
    opt['slc_osf'] = op['slc_osf']
    opt['clap_alpha'] = op['clap_alpha']
    opt['clap_beta'] = op['clap_beta']
    opt['low_pass'] = op['low_pass']
    opt['n_win'] = op['clap_win']
    opt['n_ifg'] = n_ifg
    opt['n_i'] = np.max(grid_ij[:, 0])
    opt['n_j'] = np.max(grid_ij[:, 1])
    
    re_lst = [[grid_ij[ix[i],:], op['ph_grid'], opt] for i in range(n_ps)]
    
    if runmode =='single':
        print('* Processing ps with reestimation ...')
        for i in range(n_ps):
            # print('* Processing ps with reestimation:', i + 1, 'from', n_ps, end = '\r', flush = True)
            ph_patch2[i,:] = reest(re_lst[i])

        print('')
    
    if runmode == 'mpi':
       
        cpus = os.cpu_count() // 2
        print('* Processing ps with reestimation: mpi using', cpus, 'threads')
        poolx = Pool(cpus)
        res = poolx.map(reest, re_lst)
        poolx.close()
        poolx.join()
        for i in range(n_ps):
            ph_patch2[i,:] = res[i]
            
    bperp_mat = op['bperp_mat'][ix,:]
    
    psdph = ph * np.conj(ph_patch2)
    psdph_abs = np.abs(psdph)
    psdph_abs[psdph_abs == 0] = 1
    psdph = psdph / psdph_abs
    psdph_lst = [[psdph[i][ifg_index], bperp_mat[i, ifg_index].T, op['n_trial_wraps']] for i in range(n_ps)]
    
    if runmode =='single':
        for i in range(n_ps):
            print('* Processing ps with topofit:', i + 1, 'from', n_ps, end = '\r', flush = True)
            [Kopt, Copt, cohopt, ph_residual] = ps_topofit_orig(psdph_lst[i])
            K_ps2[i] = Kopt
            C_ps2[i] = Copt
            coh_ps2[i] = cohopt
            ph_res2[i, ifg_index] = np.angle(ph_residual)
        print('')
    
    if runmode == 'mpi':
        cpus = os.cpu_count() // 2
        print('* Processing ps with topofit: mpi using', cpus, 'threads')
        poolx = Pool(cpus)
        res = poolx.map(ps_topofit_orig, psdph_lst)
        poolx.close()
        poolx.join()
        for i in range(n_ps):
            [Kopt, Copt, cohopt, ph_residual] = res[i]
            K_ps2[i] = Kopt
            C_ps2[i] = Copt
            coh_ps2[i] = cohopt
            ph_res2[i, ifg_index] = np.angle(ph_residual)
            
    op['coh_ps'][ix] = coh_ps2
    
    for i in range(len(D_A_max) - 1):
        idx = np.argwhere((D_A > D_A_max[i]) & (D_A <= D_A_max[i+1])).flatten()
        coh_chunk = op['coh_ps'][idx]
        # print(idx)
        # print(np.shape(D_A))
        D_A_mean[i] = np.mean(D_A[idx])
        coh_chunk = coh_chunk[coh_chunk != 0]
        Na = hist(coh_chunk)
        Nr = Nr_dist * np.sum(Na[:low_coh_thresh]) / np.sum(Nr_dist[:low_coh_thresh])
        Na[Na == 0] = 1
        if op['select_method'] == 'PERCENT':
            percent_rand = (np.cumsum(Nr[::-1]) / np.cumsum(Na[::-1]) * 100)[::-1]
        else:
            percent_rand = np.cumsum(Nr[::-1])[::-1]
        
        ok_ix = np.argwhere(percent_rand < max_percent_rand).flatten()
        
        if len(ok_ix) == 0:
            min_coh[i] = 1
        else:
            min_fit_ix = np.min(ok_ix) - 3
            if min_fit_ix <= 0:
                min_coh[i] = np.nan
            else:
                max_fit_ix = np.min(ok_ix) + 2
                max_fit_ix = max_fit_ix if max_fit_ix < 100 else 100
                xp = percent_rand[min_fit_ix:max_fit_ix + 1]
                yp = np.arange(min_fit_ix + 1, max_fit_ix + 2) / 100
                mu = np.mean(xp)
                st = np.std(xp, ddof = 1)
                p = np.polyfit((xp - mu) / st, yp, 3)
                min_coh[i] = np.polyval(p, (max_percent_rand - mu) / st)
                
    nonnanix = np.invert(np.isnan(min_coh))
    
    if len(nonnanix == True) == 0:
        coh_thresh = 0.3
        coh_thresh_coeffs = []
    else:
        min_coh = min_coh[nonnanix]
        D_A_mean = D_A_mean[nonnanix]

        if len(min_coh) > 1:
            # mu = np.mean(D_A_mean)
            # st = np.std(D_A_mean, ddof = 1)
            # coh_thresh_coeffs = np.polyfit((D_A_mean - mu) / st, min_coh, 1)
            coh_thresh_coeffs = np.polyfit(D_A_mean, min_coh, 1)
            
            if coh_thresh_coeffs[0] > 0:
                coh_thresh = np.polyval(coh_thresh_coeffs, D_A[ix])

            else:
                coh_thresh = np.polyval(coh_thresh_coeffs, 0.35)
                coh_thresh_coeffs = []
        else:
            coh_thresh = min_coh
            coh_thresh_coeffs = []
    
    coh_thresh[coh_thresh < 0] = 0
    
    bperp_range = np.max(bperp) - np.min(bperp)
    kps_tmp = np.abs(op['K_ps'][ix] - K_ps2)
    
    idx1 = coh_ps2 > coh_thresh
        
    idx2 = kps_tmp < 2 * np.pi / bperp_range # Allways true ?!
    keep_ix = idx1 * idx2
    ps_count = len(keep_ix[keep_ix == True])
    
    if ps_count == 0:
        print('* Pixels saved as ps:', 0)
    else:
        print('* Pixels saved as ps initially:', ps_count)

    savedict = {'ix':v2c(ix + 1),
                'keep_ix':v2c(keep_ix),
                'ph_patch2':ph_patch2,
                'ph_res2':ph_res2,
                'K_ps2':v2c(K_ps2),
                'C_ps2':v2c(C_ps2),
                'coh_ps2':v2c(coh_ps2),
                'coh_thresh':v2c(coh_thresh),
                'coh_thresh_coeffs':v2r(coh_thresh_coeffs),
                'clap_alpha':op['clap_alpha'],
                'clap_beta':op['clap_beta'],
                'n_win':op['clap_win'],
                'max_percent_rand':max_percent_rand,
                'gamma_stdev_reject':op['gamma_stdev_reject'],
                'small_baseline_flag':op['small_baseline_flag'],
                'ifg_index':v2r(ifg_index + 1)}
    
    savemat(path_to_patch + 'select' + psver + '.mat', savedict)
    
    print('Done at', int(time.time() - begin_time), 'sec')
    
if __name__ == "__main__":
    # For testing
    test_path = 'C:\\Users\\Ryzen\\Documents\\PYTHON\\stampsexport'
    ps_select(test_path, 1, 'mpi')