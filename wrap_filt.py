import numpy as np
from scipy import signal


def wrap_filt(*args):
    # params: ph, n_win, alpha, n_pad, low_flag

    if len(args) < 4:
        n_win = args[1]
        n_pad = round(n_win * 0.25)
    else:
        n_pad = args[3]
        if len(n_pad) == 0:
            n_win = int(args[1])
            n_pad = int(round(n_win * 0.25))

    if len(args) < 5:
        low_flag = 'n'

    ph = args[0]
    [n_i, n_j] = np.array([len(ph), len(ph[0])]).astype('int')
    n_inc = np.floor(n_win / 2)
    n_win_i = int(np.ceil(n_i / n_inc) - 1)
    n_win_j = int(np.ceil(n_j / n_inc) - 1)

    ph_out = np.zeros((n_i, n_j))
    low_flag = args[4]
    if low_flag == 'y':
        ph_out_low = ph_out
    else:
        ph_out_low = []

    x = np.array([int(i + 1) for i in np.arange(0, n_win / 2)])
    X, Y = np.meshgrid(x, x)
    X = X + Y
    wind_func = np.concatenate((X, np.fliplr(X)), axis=1)
    wind_func = np.concatenate((wind_func, np.flipud(wind_func)), axis=0)

    ph[np.isnan(ph)] = 0
    # For gaussian std=(N-1)/(2*alpha)
    N = 7
    alpha = 2.5
    std = ((N - 1) / (2 * alpha))
    B = (signal.gaussian(N, std=std).reshape(-1, 1)) * signal.gaussian(N, std=std)
    ph_bit = np.zeros((n_win + n_pad, n_win + n_pad))
    N = n_win + n_pad
    alpha = 16
    std = ((N - 1) / (2 * alpha))
    L = np.fft.ifftshift((signal.gaussian(N, std=std).reshape(-1, 1)) * signal.gaussian(N, std=std))

    # TODO: может быть ошибка
    for ix1 in range(1, n_win_i + 1):
        wf = wind_func
        i1 = int((ix1 - 1) * n_inc + 1)
        i2 = int(i1 + n_win - 1)

        if i2 > n_i:
            # TODO: может быть ошибка
            i_shift = int(i2 - n_i)
            i2 = int(n_i)
            i1 = int(n_i - n_win + 1)
            wf = np.concatenate((np.zeros((i_shift, n_win)), wf[0:n_win - i_shift, :]), axis=0)

        for ix2 in range(1, n_win_j + 1):
            wf2 = wf
            j1 = int((ix2 - 1) * n_inc + 1)
            j2 = int(j1 + n_win - 1)

            if j2 > n_j:
                # TODO: может быть ошибка
                j_shift = int(j2 - n_j)
                j2 = int(n_j)
                j1 = int(n_j - n_win + 1)
                wf2 = np.concatenate((np.zeros((n_win, j_shift)), wf2[:, 0:n_win - j_shift]), axis=1)

            ph_bit = ph_bit.astype('complex')
            ph_bit[0:n_win, 0:n_win] = ph[i1 - 1:i2, j1 - 1:j2]

            print()

    ph_out = ''
    ph_out_low = ''
    return [ph_out, ph_out_low]
