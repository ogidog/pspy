#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:06:44 2020

@author: anyuser
"""

import numpy as np
from scipy.optimize import nnls

def ps_topofit(*args):
    
    data_in = args[0]    
    
    if data_in[4] == 1:
        cpxphase1 = data_in[0].flatten()
        bperp1 = data_in[1].flatten()
        n_trial_wraps = data_in[2]
        asym = data_in[3]
        pi = np.pi
    
        ix1 = np.argwhere(cpxphase1)
        n_ix = len(ix1)
    
        cpxphase1 = cpxphase1[ix1]
        bperp1 = bperp1[ix1]
        bperp_range = np.max(bperp1) - np.min(bperp1)
            
        dtrial = np.ceil(8 * n_trial_wraps)
        trial_mult = np.arange(-dtrial, dtrial + 1) + asym * 8 * n_trial_wraps
        n_trials = len(trial_mult)
        trial_mult = np.reshape(trial_mult, (1, n_trials))
        trial_phase = np.reshape(bperp1 / bperp_range * pi / 4, (n_ix, 1))
        
        trial_phase_mat = np.exp(-1j * np.matmul(trial_phase, trial_mult), dtype = np.complex64)
        
        cpxphase_mat = np.repeat(np.reshape(cpxphase1,(n_ix,1)), n_trials, axis = 1)
        phaser = np.multiply(trial_phase_mat, cpxphase_mat)
        phaser_sum = np.sum(phaser, axis = 0)
        
        C_trial = np.angle(phaser_sum)
        coh_trial = np.abs(phaser_sum) / np.sum(np.abs(cpxphase1), axis = 0) # abs changed
        coh_high_max_ix = np.argmax(coh_trial)
        
        # Linearise and solve
        K0 = pi / 4 / bperp_range * trial_mult.flatten()[coh_high_max_ix]
        C0 = C_trial[coh_high_max_ix]
        coh0 = coh_trial[coh_high_max_ix]
        resphase = np.multiply(cpxphase1, np.exp(-1j * (K0 * bperp1), dtype = np.complex64))
        offset_phase = np.sum(resphase, axis = 0)
        resphase = np.angle(resphase * np.conj(offset_phase))
        weighting = np.abs(cpxphase1)
        
        Aa = np.reshape(bperp1 * weighting, (n_ix,1))
        Bb = (resphase * weighting).flatten()
    
        mopt = nnls(Aa, Bb)
        K0 = K0 + mopt[0]
        
        phase_residual = np.multiply(cpxphase1, np.exp(-1j * (K0 * bperp1), dtype = np.complex64))
        mean_phase_residual = np.sum(phase_residual, axis = 0)
        C0 = np.angle(mean_phase_residual)
        coh0 = np.divide(np.abs(mean_phase_residual), np.sum(np.abs(phase_residual), axis = 0))
        
        out = [K0[0], C0[0], coh0[0], phase_residual[:,0]]
        
    else:
        out = [0, 0, 0, np.zeros_like(data_in[0], dtype = np.float32)]
    
    return(out)