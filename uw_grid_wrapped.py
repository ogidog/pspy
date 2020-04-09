import numpy as np


def uw_grid_wrapped(*args):
    # params: ph_in, xy_in, pix_size, prefilt_win, goldfilt_flag, lowfilt_flag, gold_alpha, ph_in_predef

    print('Resampling phase to grid...\n')

    if len(args) < 2:
        print('not enough arguments')

    if len(args) < 3:
        pix_size = 200

    if len(args) < 4:
        prefilt_win = 32

    if len(args) < 5:
        goldfilt_flag = 'y'

    if len(args) < 6:
        lowfilt_flag = 'y'

    if len(args) < 7:
        gold_alpha = 0.8

    if len(args) < 8:
        ph_uw_predef = []

    ph_in_predef = args[7]
    if len(ph_in_predef) == 0:
        predef_flag = 'n'
    else:
        predef_flag = 'y'

    ph_in = args[0]
    n_ps = len(ph_in)
    n_ifg = len(ph_in[0])

    print('   Number of interferograms  : {}'.format(n_ifg))
    print('   Number of points per ifg  : {}'.format(n_ps))

    if np.any(np.isreal(ph_in)) == True and sum(sum(ph_in == 0)) > 0:
        print('Some phase values are zero')

    xy_in = args[1]
    xy_in[:, 0] = np.array([i + 1 for i in range(0, n_ps)])

    pix_size = args[2]
    if pix_size == 0:
        grid_x_min = 1
        grid_y_min = 1
        n_i = int(np.amax(xy_in[:, 2]))
        n_j = int(np.amax(xy_in[:, 1]))
        grid_ij = np.concatenate((xy_in[:, 2].reshape(-1, 1), xy_in[:, 1].reshape(-1, 1)), axis=1)
    else:
        grid_x_min = np.amin(xy_in[:, 1])
        grid_y_min = np.amin(xy_in[:, 2])

        grid_ij = np.ceil((xy_in[:, 2] - grid_y_min + 0.001) / pix_size).reshape(-1, 1)
        grid_ij[grid_ij[:, 0] == np.max(grid_ij[:, 0]), 0] = np.max(grid_ij[:, 0]) - 1
        grid_ij = np.concatenate((grid_ij, np.ceil((xy_in[:, 1] - grid_x_min + 0.001) / pix_size).reshape(-1, 1)),
                                 axis=1)
        grid_ij[grid_ij[:, 1] == np.max(grid_ij[:, 1]), 1] = np.max(grid_ij[:, 1]) - 1

        n_i = int(np.amax(grid_ij[:, 0]))
        n_j = int(np.amax(grid_ij[:, 1]))

    ph_grid = np.zeros((n_i, n_j), dtype=int)
    if predef_flag == 'y':
        ph_grid_uw = np.zeros((n_i, n_j), dtype=int)
        N_grid_uw = np.zeros((n_i, n_j), dtype=int)

    prefilt_win = args[3]
    if min(len(ph_grid), len(ph_grid[0])) < prefilt_win:
        print('Minimum dimension of the resampled grid ({} pixels) is less than prefilter window size ({})'.format(
            min(len(ph_grid), len(ph_grid[0])), prefilt_win))

    for i1 in range(0, n_ifg):
        if np.any(np.isreal(ph_in)):
            ph_this = np.exp(complex(0.0, 1.0) * ph_in[:, i1])
        else:
            ph_this = ph_in[:, i1]

        print()
