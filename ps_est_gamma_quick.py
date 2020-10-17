import numpy as np
import os
from matplotlib.pyplot import hist
import random
from numpy.fft import fftshift
from scipy.io import loadmat, savemat

from clap_filt import clap_filt
from getparm import get_parm_value as getparm
from utils import compare_objects, not_supported_param, not_supported, compare_complex_objects, compare_complex_objects2
from ps_topofit import ps_topofit


def ps_est_gamma_quick(*args):
    print("\nEstimating gamma for candidate pixels\n")

    if len(args) < 1:
        restart_flag = 0
    else:
        restart_flag = args[0]

    rho = 830000  # mean range - need only be approximately correct
    n_rand = 300000  # number of simulated random phase pixels

    grid_size = getparm("filter_grid_size")[0][0][0]
    filter_weighting = getparm("filter_weighting")[0][0]
    n_win = getparm("clap_win")[0][0][0]
    low_pass_wavelength = getparm("clap_low_pass_wavelength")[0][0][0]
    clap_alpha = getparm("clap_alpha")[0][0][0]
    clap_beta = getparm("clap_beta")[0][0][0]
    max_topo_err = getparm("max_topo_err")[0][0][0]
    lambda1 = getparm("lambda")[0][0][0]
    gamma_change_convergence = getparm("gamma_change_convergence")[0][0][0]
    gamma_max_iterations = getparm("gamma_max_iterations")[0][0][0]
    small_baseline_flag = getparm("small_baseline_flag")[0][0]

    if small_baseline_flag == "y":
        low_coh_thresh = 15  # equivalent to coh of 15/100
    else:
        low_coh_thresh = 31  # equivalent to coh of 31/100

    freq0 = 1 / low_pass_wavelength
    freq_i = np.arange(-(n_win) / grid_size / n_win / 2, (n_win - 1) / grid_size / n_win / 2,
                       1 / grid_size / n_win)
    butter_i = np.array(1 / (1 + (freq_i / freq0) ** (2 * 5)))
    low_pass = butter_i.reshape(-1, 1) * butter_i
    low_pass = fftshift(low_pass)

    psver = loadmat("psver.mat")["psver"][0][0]
    psname = "ps" + str(psver) + ".mat"
    phname = 'ph' + str(psver) + ".mat"
    bpname = 'bp' + str(psver) + ".mat"
    laname = 'la' + str(psver) + ".mat"
    incname = 'inc' + str(psver) + ".mat"
    pmname = 'pm' + str(psver) + ".mat"
    daname = 'da' + str(psver) + ".mat"

    ps = loadmat(psname)
    bp = loadmat(bpname)

    if os.path.exists(daname):
        da = loadmat(daname)
        D_A = da["D_A"]
        da.clear()
    else:
        D_A = np.ones((ps['n_ps'][0][0], 1))

    if os.path.exists(phname):
        phin = loadmat(phname)
        ph = phin["ph"]
        phin.clear()
    else:
        ph = ps["ph"]

    null_i = np.where(ph.T == 0)[1]
    null_i = np.unique(null_i)
    good_ix = np.ones((ps["n_ps"][0][0], 1))
    good_ix[null_i] = 0
    good_ix = good_ix.astype("bool")

    if small_baseline_flag == 'y':
        not_supported_param("small_baseline_flag", small_baseline_flag)
        # bperp=ps.bperp;
        # n_ifg=ps.n_ifg;
        # n_image=ps.n_image;
        # n_ps=ps.n_ps;
        # ifgday_ix=ps.ifgday_ix;
        # xy=ps.xy;
    else:
        ph = np.delete(ph, ps["master_ix"][0][0] - 1, axis=1)
        bperp = np.delete(ps["bperp"], ps["master_ix"][0][0] - 1, axis=0)
        n_ifg = ps["n_ifg"][0][0] - 1
        n_ps = ps["n_ps"][0][0]
        xy = ps["xy"]
    ps.clear()

    A = np.abs(ph)
    # A = A.astype("float32")
    A[A == 0] = 1  # avoid divide by zero
    ph = ph / A

    ### ===============================================
    ### The code below needs to be made sensor specific
    ### ===============================================
    if os.path.exists(incname):
        not_supported()
        # print('Found inc angle file \n')
        # inc = loadmat(incname)
        # inc_mean = np.mean(inc["inc"][inc["inc"] != 0])
        # inc.clear()
    else:
        if os.path.exists(laname):
            print('Found look angle file \n')
            la = loadmat(laname)
            inc_mean = np.mean(la["la"]) + 0.052  # incidence angle approx equals look angle + 3 deg
            la.clear()
        else:
            inc_mean = 21 * np.pi / 180  # guess the incidence angle
    max_K = max_topo_err / (lambda1 * rho * np.sin(inc_mean) / 4 / np.pi)
    ### ===============================================
    ### The code below needs to be made sensor specific
    ### ===============================================

    bperp_range = np.max(bperp) - np.min(bperp)
    n_trial_wraps = (bperp_range * max_K / (2 * np.pi))
    print('n_trial_wraps = {}'.format(n_trial_wraps))

    if restart_flag > 0:
        not_supported()
        # %disp(['Restarting: iteration #',num2str(i_loop),' step_number=',num2str(step_number)])
        # logit('Restarting previous run...')
        # load(pmname)
        # weighting_save=weighting;
        # if ~exist('gamma_change_save','var')
        #    gamma_change_save=1;
        # end
    else:
        print('Initialising random distribution...')

        if small_baseline_flag == "y":
            not_supported_param("small_baseline_flag", small_baseline_flag)
            # rand_image=2*pi*rand(n_rand,n_image);
            # rand_ifg=zeros(n_rand,n_ifg);
            # for i=1:n_ifg
            # rand_ifg(:, i)=rand_image(:, ifgday_ix(i, 2))-rand_image(:, ifgday_ix(i, 1));
            # end
            # clear rand_image
        else:
            # TODO: только для тестов, убрать!!!!
            rand_ifg = loadmat("..\\..\\rand_ifg.mat")["rand_ifg"]
            # TODO: раскоментить это
            # np.random.seed(seed=2005)  # determine distribution for random phase
            # rand_ifg = 2 * np.pi * np.random.rand(n_rand, n_ifg)

        # TODO: убрать if - это для быстрой отладки
        if os.path.exists("coh_rand.mat"):
            coh_rand = loadmat("coh_rand.mat")["coh_rand"][0]
        else:
            coh_rand = np.zeros(n_rand)
            for i in np.arange(n_rand - 1, -1, -1):
                K_r, C_r, coh_r, phase_residual = ps_topofit(np.exp(1j * rand_ifg[i, :]), bperp, n_trial_wraps, 'n')
                coh_rand[i] = coh_r

            # TODO: убрать, Это только для быстрой отладки!!!
            savemat("coh_rand.mat", {"coh_rand": coh_rand})

        rand_ifg = []
        coh_bins = np.arange(0.005, 1.01, 0.01)
        step = 0.01 / 2
        Nr = np.histogram(coh_rand, coh_bins - step)[0]  # distribution of random phase points
        Nr[0] += len(coh_rand[coh_rand < step])
        i = len(Nr) - 1
        while Nr[i] == 0:
            i = i - 1
        Nr_max_nz_ix = i

        step_number = 1
        K_ps = np.zeros((n_ps, 1))
        C_ps = np.zeros((n_ps, 1))
        coh_ps = np.zeros((n_ps, 1))
        coh_ps_save = np.zeros((n_ps, 1))
        N_opt = np.zeros((n_ps, 1))
        ph_res = np.zeros((n_ps, n_ifg)).astype("float32")
        ph_patch = np.zeros(np.shape(ph)).astype("complex")
        N_patch = np.zeros((n_ps, 1))

        xy = xy.astype("float32")
        grid_ij = np.array(np.ceil((xy[:, 2] - np.min(xy[:, 2]) + 1e-6) / grid_size).reshape(-1, 1), dtype="float32")
        grid_ij[grid_ij[:, 0] == np.max(grid_ij[:, 0]), 0] = np.max(grid_ij[:, 0]) - 1
        grid_ij = np.append(grid_ij, np.ceil((xy[:, 1] - np.min(xy[:, 1]) + 1e-6) / grid_size).reshape(-1, 1), axis=1)
        grid_ij[grid_ij[:, 1] == np.max(grid_ij[:, 1]), 1] = np.max(grid_ij[:, 1]) - 1
        i_loop = 1
        weighting = 1 / D_A
        weighting_save = weighting
        gamma_change_save = 0

    n_i = int(np.max(grid_ij[:, 0]))
    n_j = int(np.max(grid_ij[:, 1]))

    print('\n{} PS candidates to process'.format(n_ps))
    xy[:, 0] = np.array([*range(n_ps)]) + 1  # assumption that already sorted in ascending column 3 (y-axis) order
    loop_end_sw = 0
    n_high_save = 0

    while loop_end_sw == 0:
        # if step_number==1     % check in case restarting and step 1 already completed
        print('iteration {}'.format(i_loop))
        print('Calculating patch phases...')

        ph_grid = np.zeros((n_i, n_j, n_ifg)).astype("complex")
        ph_filt = np.copy(ph_grid)
        ph_weight = ph * np.exp(-1j * bp["bperp_mat"] * np.tile(K_ps, (1, n_ifg))) * np.tile(weighting, (1, n_ifg))

        grid_ij = grid_ij.astype("int")
        # TODO: убрать
        grid_ij[5084, 1] -= 1
        ######################
        for i in range(n_ps):
            # ph_grid(grid_ij(i,1),grid_ij(i,2),:)=ph_grid(grid_ij(i,1),grid_ij(i,2),:)+shiftdim(ph(i,:),-1)*weighting(i);
            ph_grid[grid_ij[i, 0] - 1, grid_ij[i, 1] - 1, :] = ph_grid[grid_ij[i, 0] - 1, grid_ij[i, 1] - 1,
                                                               :] + ph_weight[i, :]
        for i in range(n_ifg):
            ph_filt[:, :, i] = clap_filt(ph_grid[:, :, i], clap_alpha, clap_beta, n_win * 0.75, n_win * 0.25, low_pass)

        for i in range(n_ps):
            ph_patch[i, 0:n_ifg] = ph_filt[grid_ij[i, 0] - 1, grid_ij[i, 1] - 1, :]

        ph_filt = []
        ix = ph_patch != 0
        ph_patch[ix] = ph_patch[ix] / abs(ph_patch[ix])

        if restart_flag < 2:

            print('Estimating topo error...')
            step_number = 2

            for i in range(n_ps):
                psdph = ph[i, :] * np.conj(ph_patch[i, :])
                if np.sum(psdph == 0) == 0:
                    [Kopt, Copt, cohopt, ph_residual] = ps_topofit(psdph, bp["bperp_mat"][i, :].reshape(-1, 1),
                                                                   n_trial_wraps, 'n')
                    K_ps[i] = Kopt
                    C_ps[i] = Copt
                    coh_ps[i] = cohopt
                    N_opt[i] = len(Kopt) if type(Kopt) == np.ndarray else 1
                    ph_res[i, :] = np.angle(ph_residual).flatten()
                else:
                    K_ps[i] = np.nan
                    coh_ps[i] = 0
                if i % 100000 == 0 and i != 0:
                    print('{} PS processed'.format(i))

            step_number = 1

            diff = compare_complex_objects(ph_residual, 'ph_residual')
            pass

    return []
