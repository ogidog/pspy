import numpy as np


def clap_filt(*args):
    # CLAP_FILT Combined Low-pass Adaptive Phase filtering
    #   [ph_out]=CLAP_filt(ph,alpha,beta,n_win,n_pad,low_pass_fftshifted)
    #
    #   Andy Hooper, June 2006

    # ph,alpha,beta,n_win,n_pad,low_pass

    if len(args) < 2:
        alpha = 0.5

    if len(args) < 3:
        beta = 0.1

    if len(args) < 4:
        n_win = 32

    if len(args) < 5:
        n_pad = 0

    if len(args) < 6:
        low_pass = np.zeros(n_win + n_pad)

    ph = args[0]
    ph_out = np.zeros(np.shape(ph))
    [n_i, n_j] = np.shape(ph)

    n_win = args[3]
    n_inc = np.floor(n_win / 4)
    n_win_i = np.ceil(n_i / n_inc) - 3
    n_win_j = np.ceil(n_j / n_inc) - 3

    x = [*range(int(n_win / 2))]

    return ph_out
