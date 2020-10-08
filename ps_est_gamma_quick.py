import numpy as np
from getparm import get_parm_value as getparm


def ps_est_gamma_quick(*args):
    print("\nEstimating gamma for candidate pixels\n")

    if len(args) < 1:
        restart_flag = 0

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
    if (int(1 / grid_size / n_win) == 0 or int((n_win - 2) / grid_size / n_win / 2) == 0):
        freq0 = np.array([])
    else:
        freq_i = np.arange(int(-(n_win) / grid_size / n_win / 2), int((n_win - 2) / grid_size / n_win / 2),
                           int(1 / grid_size / n_win))

    return []
