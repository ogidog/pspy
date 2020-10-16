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
    else:
        alpha = args[1]

    if len(args) < 3:
        beta = 0.1
    else:
        beta = args[2]

    if len(args) < 4:
        n_win = 32
    else:
        n_win = args[3]

    if len(args) < 5:
        n_pad = 0
    else:
        n_pad = args[4]

    if len(args) < 6:
        low_pass = np.zeros(int(n_win) + int(n_pad))
    else:
        low_pass = args[5]

    ph = args[0]
    ph_out = np.zeros(np.shape(ph)).astype("complex")
    [n_i, n_j] = np.shape(ph)

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

    # For gaussian std=(N-1)/(2*alpha1)
    N = 7
    alpha1 = 2.5
    std = ((N - 1) / (2 * alpha1))
    B = signal.gaussian(7, std=std) * signal.gaussian(7, std=std).reshape(-1, 1)
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
            wf = np.concatenate((np.zeros((int(i_shift), int(n_win))), wf[0:int(n_win) - int(i_shift), :]), axis=0)

        for ix2 in range(int(n_win_j)):
            wf2 = np.copy(wf)
            j1 = int((ix2) * n_inc + 1)
            j2 = int(j1 + n_win - 1)
            if j2 > n_j:
                j_shift = int(j2 - n_j)
                j2 = int(n_j)
                j1 = int(n_j - n_win + 1)
                wf2 = np.concatenate((np.zeros((int(n_win), int(j_shift))), wf2[:, 0:int(n_win) - int(j_shift)]),
                                     axis=1)

            ph_bit[0:int(n_win), 0:int(n_win)] = ph[i1 - 1:i2, j1 - 1:j2]
            ph_fft = np.fft.fft2(ph_bit)
            H = np.abs(ph_fft)
            H = np.fft.ifftshift(signal.convolve2d(np.fft.fftshift(H), B, mode="same"))
            meanH = np.median(H)
            if meanH != 0:
                H = H / meanH
            H = H ** alpha
            H = H - 1  # set all values under median to zero
            H[H < 0] = 0  # set all values under median to zero
            G = H * beta + low_pass
            ph_filt = np.fft.ifft2(ph_fft * G)
            ph_filt = ph_filt[0:int(n_win), 0:int(n_win)] * wf2
            if np.isnan(ph_filt[0, 0]):
                ph_filt[0, 0] = 0j

            ph_out[i1 - 1:i2, j1 - 1:j2] = ph_out[i1 - 1:i2, j1 - 1:j2] + ph_filt

    return ph_out
