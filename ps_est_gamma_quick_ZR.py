#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:06:44 2020

@author: anyuser
"""

import os
import sys
import time

from multiprocessing import Pool

import numpy as np
from scipy.io import loadmat, savemat
from scipy.interpolate import splrep, splev, interp1d
from scipy.signal import convolve2d, lfilter
from scipy.signal.windows import gaussian
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from numpy.linalg import solve, lstsq
import clap_filt as cf

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

    ix = np.argwhere(cpxphase).flatten()

    if len(ix) > 0:
        cpxphase = cpxphase[ix]
        
        bperp = bperp.copy()[ix]
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
    else:
        K0 = 0
        C0 = 0
        coh0 = 0
        phase_residual = np.zeros(len(cpxphase))
        
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

def interpolate1d(vector, factor):

    x = np.arange(np.size(vector))
    y = vector
    f = interp1d(x, y)

    x_extended_by_factor = np.linspace(x[0], x[-1], np.size(x) * factor)
    y_new = np.zeros(np.size(x_extended_by_factor))

    i = 0
    for x in x_extended_by_factor:
        y_new[i] = f(x)
        i += 1

    return(y_new)

def interp_m(x):
    fc = [0,-0.00195897,-0.00376454,-0.0052573,-0.00629795,-0.00677751,-0.00662615,
          -0.00582005,-0.00438587,-0.00240256,6.82121e-13,0.0109082,0.0211769,0.0298946,
          0.0362235,0.0394578,0.0390785,0.0348015,0.0266164,0.0148138,-9.09495e-13,-0.0359897,
          -0.0712704,-0.102814,-0.127577,-0.142658,-0.145453,-0.133805,-0.106138,
          -0.0615693,9.09495e-13,0.107916,0.227948,0.355221,0.484381,0.609836,0.726002,
          0.827549,0.909648,0.968193,1,0.968193,0.909648,0.827549,0.726002,0.609836,0.484381,
          0.355221,0.227948,0.107916,9.09495e-13,-0.0615693,-0.106138,-0.133805,-0.145453,
          -0.142658,-0.127577,-0.102814,-0.0712704,-0.0359897,-9.09495e-13,0.0148138,0.0266164,
          0.0348015,0.0390785,0.0394578,0.0362235,0.0298946,0.0211769,0.0109082,6.82121e-13,
          -0.00240256,-0.00438587,-0.00582005,-0.00662615,-0.00677751,-0.00629795,-0.0052573,
          -0.00376454,-0.00195897,0]
    fca = np.array(fc)
    v = np.insert(x.copy(), 0, np.ones((5)))
    pp = np.zeros(len(v) * 10)
    pp[0::10] = v
    out = np.convolve(fca, pp, 'valid')
    out = np.hstack((out, np.zeros((40))))
    return(out)

def filter1(x):
    b = gaussian(7, 1.2)
    signal = np.insert(x.copy(), 0, np.ones(7))
    out = np.convolve(signal, b, mode = 'full') / sum(b)
    return(out[:-6])
    
def filter2(k, a):
    return(convolve2d(a, np.rot90(k, 2), mode = 'same'))

def clap_filt(phin, alpha = 0.5, beta = 0.1, n_win = 32, n_pad = 0, low_pass = []):
    n_win = int(n_win)
    n_pad = int(n_pad)
    if len(low_pass) == 0:
        low_pass = np.zeros(n_win + n_pad)

    ph_out = np.zeros_like(phin)
    [n_i,n_j] = np.shape(phin)
    
    n_inc = int(np.floor(n_win / 4))
    n_win_i = int(np.ceil(n_i / n_inc) - 3)
    n_win_j = int(np.ceil(n_j / n_inc) - 3)
    
    x = np.arange(0, n_win / 2)
    X, Y = np.meshgrid(x, x)
    X = X + Y
    wind_func = np.hstack((X, np.fliplr(X)))
    wind_func = np.vstack((wind_func, np.flipud(wind_func)))
    wind_func = wind_func + 1e-6 
    
    phin[np.isnan(phin)] = 0
    gw = np.reshape(gaussian(7, 1.2), (7, 1))
    B = np.matmul(gw, gw.T)
    n_win_ex = n_win + n_pad
    ph_bit = np.zeros((n_win_ex, n_win_ex), dtype = np.complex128)
    
    for ix1 in range(n_win_i):
        wf = wind_func.copy()
        i1 = ix1 * n_inc
        i2 = i1 + n_win
        if i2 > n_i:
            i_shift = i2 - n_i
            i2 = n_i
            i1 = n_i - n_win
            wf = np.vstack((np.zeros((i_shift,n_win)), wf[:n_win - i_shift,:]))
        
        for ix2 in range(n_win_j):
            wf2 = wf
            j1 = ix2 * n_inc
            j2 = j1 + n_win
            if j2 > n_j:
                j_shift = j2 - n_j
                j2 = n_j
                j1 = n_j - n_win
                wf2 = np.hstack((np.zeros((n_win,j_shift)), wf2[:,:n_win - j_shift]))
            
            ph_clip = phin[i1:i2, j1:j2].copy()
            ph_bit[:n_win, :n_win] = ph_clip.copy()
            ph_fft = fft2(ph_bit)
            H = np.abs(ph_fft)
            H = filter2(B, fftshift(H))
            H = ifftshift(H)
            meanH = np.median(H[:])
            if meanH != 0:
                H = H / meanH
            
            H = H**alpha
            H = H - 1
            H[H < 0] = 0
            G = H * beta + low_pass
            ph_filt = ifft2(ph_fft * G)
            ph_filt = ph_filt[:n_win, :n_win] * wf2
            
            if np.isnan(ph_filt[0,0]):
                print('* NaN in array in clap_filt.')
                sys.exit(0)
            
            ph_out[i1:i2, j1:j2] = ph_out[i1:i2, j1:j2] + ph_filt

    return(ph_out)

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
            x = x.reshape((1,)*(-n) + x.shape)
            
    return(x, n)[outsel]

##########################
def ps_est_gamma_quick(*args):
    
    begin_time = time.time()

    if len(args) == 3: # To debug
        path_to_task = args[0] + os.sep
        path_to_patch = path_to_task + 'PATCH_' + str(args[1]) + os.sep
    
    else: # To essential run
        path_to_task = os.sep.join(os.getcwd().split(os.path.sep)[:-1]) + os.sep
        path_to_patch = os.getcwd() + os.sep
    
    path_to_script = os.path.dirname(os.path.realpath(sys.argv[0])) + os.sep
    
    print()    
    print('********** Stage 2 *********')
    print('*** Estimate gamma quick ***')
    print('')
    print('Work dir:', path_to_patch)

    psver = str(1) # str(loadmat(path_to_patch + 'psver.mat', squeeze_me = True)['psver'])
    step_number = 1
    restart_flag = 0
    
    runmode = 'single'        
    if len(args) > 0:
        if 'mpi' in args:
            runmode = 'mpi'
                
    op = loadmat(path_to_task + 'parms.mat', squeeze_me = True)
    
    op['rho'] = 830000 # mean range - need only be approximately correct
    op['n_rand'] = 300000 # number of simulated random phase pixels

    datanames = ['ps', 'ph', 'bp', 'la', 'da', 'inc']
    
    for fname in datanames:
        fnpath = path_to_patch + fname + psver + '.mat'
        if os.path.exists(fnpath):
            data = loadmat(fnpath, squeeze_me = True)
            for field in list(data):
                if field[:2] != '__':
                    op[field] = data[field]

    print('* Pixels loaded:', len(op['xy']))
    
    low_coh_thresh = 31

    if 'D_A' in op:
        da = op['D_A']
    else:
        da = np.ones(op['n_ps'])

    inc_mean = 0
    if 'inc' in op:
        inc_mean = np.mean(op['inc'])
    else:
        if 'la' in op:
            inc_mean = np.mean(op['la']) + 0.052
        else:
            inc_mean = 21 * np.pi / 180
            
    master_ix = op['master_ix'] if 'master_ix' in op else 0
    
    freq0 = 1 / op['clap_low_pass_wavelength']
    n_win = op['clap_win']
    fgs = op['filter_grid_size']
    freq_i = np.arange(-(n_win)/fgs/n_win/2, (n_win-2)/fgs/n_win/2+(1/fgs/n_win*0.01), 1/fgs/n_win)
    butter_i = (1. / (1 + (freq_i / freq0) ** (2*5))).reshape((1,len(freq_i)))
    low_pass = np.dot(butter_i.reshape((len(freq_i),1)), butter_i)
    low_pass = np.fft.fftshift(low_pass)
    ########################################################################### 
    good_ix = np.ones(op['n_ps'], dtype = int)
    ph_zeros_idx = np.argwhere(np.abs(op['ph']) == 0)
    for i in range(len(ph_zeros_idx)):
        good_ix[ph_zeros_idx[i,0]] = 0
    
    master_ix = op['master_ix']
    ph = op['ph']
    ph = np.delete(ph, master_ix - 1, axis = 1)
    bperp = op['bperp']
    bperp = np.delete(bperp, master_ix - 1, axis = 0)
    n_ifg = op['n_ifg'] - 1
    n_ps = op['n_ps']
    xy = op['xy']
    
    A = np.abs(ph)
    A[A == 0] = 1
    ph = ph / A

    max_K = op['max_topo_err'] / (op['lambda'] * op['rho'] * np.sin(inc_mean) / 4 / np.pi)    
    bperp_range = np.max(bperp) - np.min(bperp)
    n_trial_wraps = (bperp_range * max_K / (2 * np.pi))
    
    print('* Initialising random distribution.')
    fnpath = "any" #path_to_task + 'random.npy'
    if os.path.exists(fnpath):
        print('* Load random matrix.')
        rand_ifg = np.load(fnpath)
    else:
        print('* Generate random matrix.')
        np.random.seed(seed = 2005)
        rand_ifg = np.exp(1j * 2 * np.pi * np.random.rand(op['n_rand'], n_ifg))
        # np.save(fnpath, rand_ifg)
        
    icount = np.arange(op['n_rand'])
    rand_lst = [[rand_ifg[i,:], bperp, n_trial_wraps] for i in icount]
    
    coh_rand = np.zeros(op['n_rand'])

    if runmode =='single':
        for i in range(op['n_rand']):
            print('* Processing random with topofit:', i + 1, 'from', op['n_rand'], end = '\r', flush = True)
            res = ps_topofit_orig(rand_lst[i])
            coh_rand[i] = res[2]
        print('')
    
    if runmode == 'mpi':
        cpus = os.cpu_count() // 2
        print('* Processing random with topofit: mpi using', cpus, 'threads')
        poolx = Pool(cpus)
        res = poolx.map(ps_topofit_orig, rand_lst)
        poolx.close()
        poolx.join()
        for i in range(op['n_rand']):
            coh_rand[i] = res[i][2]
        
    Nr = hist(coh_rand)
    i = len(Nr) - 1
    while Nr[i] == 0:
        i -= 1
    Nr_max_nz_ix = i
    
    K_ps = np.zeros(n_ps)
    C_ps = np.zeros(n_ps)
    coh_ps = np.zeros(n_ps)
    coh_ps_save = np.zeros(n_ps)
    N_opt = np.zeros(n_ps)
    ph_res = np.zeros((n_ps, n_ifg), dtype = float)
    ph_patch = np.zeros_like(ph, dtype = complex)
    
    grid_ij = np.zeros((n_ps, 2), dtype = int)
    for i in range(n_ps):
        if np.isnan(xy[i,1]) or np.isnan(xy[i,2]):
            xy[i,1] = 0 #xy[i-1,1] * 1.001
            xy[i,2] = 0 #xy[i-1,2] * 0.999
    
    min1 = np.min(xy[:,1])
    min2 = np.min(xy[:,2])
    
    x = np.ceil((xy[:,1] - min1 + 1e-6) / fgs)
    y = np.ceil((xy[:,2] - min2 + 1e-6) / fgs)
    
    y[y == y.max()] = y.max() - 1
    x[x == x.max()] = x.max() - 1
    y = y.astype(int)
    x = x.astype(int)
    grid_ij[:,0] = y
    grid_ij[:,1] = x
    n_i = np.max(y)
    n_j = np.max(x)
    
    weighting = 1 / da
    gamma_change_save = 0    
    loop_end_sw = 0
    i_loop = 1
    
    while loop_end_sw == 0:
        print('* Iteration:', i_loop)
        ph_grid = np.zeros((n_i, n_j, n_ifg), dtype = complex)
        ph_filt = np.zeros((n_i, n_j, n_ifg), dtype = complex)
        a = -1j * op['bperp_mat']
        b = np.repeat(np.reshape(K_ps, (n_ps, 1)), n_ifg, axis = 1)
        c = np.repeat(np.reshape(weighting, (n_ps, 1)), n_ifg, axis = 1)
        ph_weight = ph * np.exp(a * b) * c
                                        
        for i in range(n_ps):
            ph_grid[y[i]-1,x[i]-1,:] = ph_grid[y[i]-1,x[i]-1,:] + shiftdim(ph_weight[i,:], -1)[0].flatten()
    
        for i in range(n_ifg):
            ph_filt[:,:,i] = clap_filt(ph_grid[:,:,i], op['clap_alpha'], op['clap_beta'], n_win*0.75, n_win*0.25, low_pass)
                    
        for i in range(n_ps):
            ph_patch[i,:n_ifg] = np.squeeze(ph_filt[y[i]-1, x[i]-1, :])
        
        abs_ph_patch = np.abs(ph_patch)
        ph_patch[abs_ph_patch == 0] = 1
        ph_patch = ph_patch / abs_ph_patch
        
        if restart_flag < 2:
            
            psdph_lst = [[np.multiply(ph[i,:], np.conj(ph_patch[i,:])), op['bperp_mat'][i,:], n_trial_wraps] for i in range(n_ps)]

            if runmode == 'single':
                for i in range(n_ps):
                    print('* Processing ps with topofit:', i + 1, 'from', n_ps, end = '\r', flush = True)
                    res = ps_topofit_orig(psdph_lst[i])
                    K_ps[i] = res[0]
                    C_ps[i] = res[1]
                    coh_ps[i] = res[2]
                    ph_res[i,:] = np.angle(res[3])
                    N_opt[i] = 1
                print('')
            
            if runmode == 'mpi':
                cpus = os.cpu_count() // 2
                print('* Processing ps with topofit: mpi using', cpus, 'threads')
                poolx = Pool(cpus)
                res = poolx.map(ps_topofit_orig, psdph_lst)
                poolx.close()
                poolx.join()
                for i in range(n_ps):
                    K_ps[i] = res[i][0] if not np.isnan(res[i][0]) else 0
                    C_ps[i] = res[i][1] if not np.isnan(res[i][1]) else 0
                    coh_ps[i] = res[i][2]
                    ph_res[i,:] = np.angle(res[i][3])
                    N_opt[i] = 1
                    
            gamma_change_rms = np.sqrt(np.sum((coh_ps - coh_ps_save)**2) / n_ps)
            gamma_change_change = gamma_change_rms - gamma_change_save
    
            gamma_change_save = gamma_change_rms
            coh_ps_save = coh_ps
        
            if (np.abs(gamma_change_change) < op['gamma_change_convergence']) or (i_loop >= op['gamma_max_iterations']):
                loop_end_sw = 1
                
            else:
                i_loop = i_loop + 1
                # print('* Iteration', i_loop)
                if op['filter_weighting'] == 'P-square':
                    Na = hist(coh_ps)
                    Nr = Nr * np.sum(Na[:low_coh_thresh]) / np.sum(Nr[:low_coh_thresh])
                    Na[Na == 0] = 1
                    Prand = Nr / Na
                    Prand[:low_coh_thresh] = 1
                    Prand[Nr_max_nz_ix + 1:] = 0
                    Prand[Prand > 1] = 1
                    Prand = filter1(Prand)
                    Prand = Prand[7:]
                    Prand = interp_m(Prand)
                    Prand = Prand[0:-9]
                    idx = (np.round(coh_ps * 1000)).astype(int)
                    Prand_ps = Prand[idx]
                    weighting = (1 - Prand_ps) ** 2
                else:
                    print('* Not supported filter weighting:', op['filter_weighting'])
                    sys.exit(0)

        else:
            loop_end_sw = 1
    # end while loop_end_sw == 0:
        
    savedict = {'ph_patch':ph_patch,
                'K_ps':v2c(K_ps),
                'C_ps':v2c(C_ps),
                'coh_ps':v2c(coh_ps),
                'N_opt':v2c(N_opt),
                'ph_res':ph_res,
                'step_number':step_number,
                'ph_grid':ph_grid,
                'n_trial_wraps':n_trial_wraps,
                'grid_ij':grid_ij,
                'grid_size':fgs,
                'low_pass':low_pass,
                'i_loop':i_loop,
                'ph_weight':ph_weight,
                'Nr':v2r(Nr),
                'Nr_max_nz_ix':Nr_max_nz_ix,
                'coh_bins':v2r(np.arange(0.005, 1, 0.01)),
                'coh_ps_save':coh_ps_save,
                'gamma_change_save':gamma_change_save}

    savefile = path_to_patch + 'pm' + psver + '.mat'    

    savemat(savefile, savedict)    

    print('Done at', int(time.time() - begin_time), 'sec')
    
if __name__ == "__main__":
    # For testing
    test_path = 'C:\\Users\\Ryzen\\Documents\\PYTHON\\stampsexport_zry'
    ps_est_gamma_quick(test_path, 1, 'mpi')