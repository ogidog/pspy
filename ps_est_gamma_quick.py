import numpy as np
import os
from numpy.fft import fftshift
from scipy.io import loadmat

from getparm import get_parm_value as getparm
from utils import compare_objects, not_supported_param, not_supported


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
        #random.seed(a=2005,version=2)

        np.random.seed(2005)  # determine distribution for random phase

        if small_baseline_flag == "y":
            not_supported_param("small_baseline_flag", small_baseline_flag)
            # rand_image=2*pi*rand(n_rand,n_image);
            # rand_ifg=zeros(n_rand,n_ifg);
            # for i=1:n_ifg
            # rand_ifg(:, i)=rand_image(:, ifgday_ix(i, 2))-rand_image(:, ifgday_ix(i, 1));
            # end
            # clear rand_image
        else:
            rand_ifg = 2 * np.pi * np.random.rand(n_rand, n_ifg)

    # diff = compare_objects(good_ix, 'good_ix')

    return []
