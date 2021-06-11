import numpy as np
import numpy.linalg
from numpy.linalg import solve, lstsq

from utils import compare_objects, compare_complex_objects, not_supported_param, not_supported

def ps_topofit(cpxphase, bperp, n_trial_wraps, sym):
    
    asym = 0
    
    cpxphase = cpxphase.reshape(-1, 1)

    ix = cpxphase != 0  # if signal of one image is 0, dph set to 0
    ix = ix.flatten()
    cpxphase = cpxphase[ix]
    
    bperp = bperp[ix]
    bperp_range = np.max(bperp) - np.min(bperp)
    
    trial_mult = np.arange(-np.ceil(8 * n_trial_wraps), np.ceil(8 * n_trial_wraps) + 1) + asym * 8 * n_trial_wraps
    n_trials = len(trial_mult)
    trial_phase = bperp / bperp_range * np.pi / 4
    trial_phase_mat = np.exp(-1j * trial_phase * trial_mult)
    cpxphase_mat = np.tile(cpxphase, (1, n_trials))
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
    a = weighting.astype("float64") * bperp.astype("float64")
    b = weighting.astype("float64") * resphase.astype("float64")
    shape_a = np.shape(a)
    if shape_a[0] == shape_a[1]:
        mopt = solve(a, b)[0][0][0]
    else:
        mopt = lstsq(a, b, rcond = None)[0][0][0]
    K0 = K0 + mopt
    phase_residual = cpxphase * np.exp(-1j * (K0 * bperp))
    mean_phase_residual = np.sum(phase_residual)
    C0 = np.angle(mean_phase_residual)  # static offset (due to noise of master + average noise of rest)
    coh0 = np.abs(mean_phase_residual) / np.sum(np.abs(phase_residual))

    return(K0, C0, coh0, phase_residual)
