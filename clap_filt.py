import numpy as np
from scipy import signal

from utils import compare_objects, not_supported_param, not_supported, compare_complex_objects


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
    [X, Y] = np.meshgrid(x, x)
    X = X + Y
    wind_func = np.concatenate((X, np.fliplr(X)), axis=1)
    wind_func = np.concatenate((wind_func, np.flipud(wind_func)), axis=0)
    wind_func = wind_func + 1e-6  # so doesn't go to zero in corners

    ph[np.isnan(ph)] = 0

    # For gaussian std=(N-1)/(2*alpha)
    N = 7
    alpha = 2.5
    std = ((N - 1) / (2 * alpha))
    B = signal.gaussian(7, std=std) * signal.gaussian(7, std=std).reshape(-1, 1)
    n_pad = args[4]
    n_win_ex = int(n_win + n_pad)
    ph_bit = np.zeros((n_win_ex, n_win_ex)).astype("complex")

    for ix1 in range(int(n_win_i)):
        wf = np.copy(wind_func)
        i1 = int((ix1) * n_inc + 1)
        i2 = int(i1 + n_win - 1)
        if i2 > n_i:
            i_shift = int(i2 - n_i)
            i2 = int(n_i)
            i1 = int(n_i - n_win + 1)
            wf = np.concatenate((np.zeros((i_shift, n_win)), wf[0:n_win - i_shift, :]))

        for ix2 in range(int(n_win_j)):
            wf2 = np.copy(wf)
            j1 = int((ix2) * n_inc + 1)
            j2 = int(j1 + n_win - 1)
            if j2 > n_j:
                j_shift = int(j2 - n_j)
                j2 = int(n_j)
                j1 = int(n_j - n_win + 1)
                wf2 = np.concatenate((np.zeros((n_win, j_shift)), wf2[:, 0:n_win - j_shift]))

            ph_bit[0:int(n_win), 0:int(n_win)] = ph[i1-1:i2, j1 - 1:j2]

            diff = compare_complex_objects(ph_bit, "ph_bit")
            pass

    return ph_out
