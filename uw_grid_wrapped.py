import sys
import numpy as np
import functools

from scipy.io import savemat

from utils import not_supported_param, compare_objects
from wrap_filt import wrap_filt
from utils import *


def uw_grid_wrapped(*args):
    # params: ph_in, xy_in, pix_size, prefilt_win, goldfilt_flag, lowfilt_flag, gold_alpha, ph_in_predef

    print('Resampling phase to grid...\n')

    if len(args) < 2:
        print('not enough arguments')
        sys.exit(0)

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
        grid_ij = grid_ij.astype('int')
    else:
        grid_x_min = np.amin(xy_in[:, 1])
        grid_y_min = np.amin(xy_in[:, 2])

        grid_ij = np.ceil((xy_in[:, 2] - grid_y_min + 0.001) / pix_size).reshape(-1, 1)
        grid_ij[grid_ij[:, 0] == np.max(grid_ij[:, 0]), 0] = np.max(grid_ij[:, 0]) - 1
        grid_ij = np.concatenate((grid_ij, np.ceil((xy_in[:, 1] - grid_x_min + 0.001) / pix_size).reshape(-1, 1)),
                                 axis=1)
        grid_ij[grid_ij[:, 1] == np.max(grid_ij[:, 1]), 1] = np.max(grid_ij[:, 1]) - 1

        grid_ij = grid_ij.astype('int')

        n_i = int(np.amax(grid_ij[:, 0]))
        n_j = int(np.amax(grid_ij[:, 1]))

    ph_grid = np.zeros((n_i, n_j), dtype=int)
    if predef_flag == 'y':
        not_supported_param('predef_flag', 'y')
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

        if predef_flag == 'y':
            not_supported_param('predef_flag', 'y')
            # ph_this_uw = ph_in_predef[:, i1]
            # ph_grid_uw[:] = 0
            # N_grid_uw[:] = 0

        ph_grid[:] = 0

        if pix_size == 0:
            not_supported_param('pix_size', 0)
            # ph_grid((xy_in(:,2)-1)*n_i+xy_in(:,3))=ph_this;
            # if predef_flag=='y':
            #    ph_grid_uw((xy_in(:,2)-1)*n_i+xy_in(:,3))=ph_this_uw;
        else:
            ph_grid = ph_grid.astype('complex')
            for i in range(0, n_ps):
                ph_grid[grid_ij[i, 0] - 1, grid_ij[i, 1] - 1] = ph_grid[grid_ij[i, 0] - 1, grid_ij[i, 1] - 1] + ph_this[
                    i]

            if predef_flag == 'y':
                not_supported_param('predef_flag', 'y')
            #    for i=1:n_ps
            #        if ~isnan(ph_this_uw(i))
            #            ph_grid_uw(grid_ij(i,1),grid_ij(i,2))=ph_grid_uw(grid_ij(i,1),grid_ij(i,2))+ph_this_uw(i);
            #           N_grid_uw(grid_ij(i,1),grid_ij(i,2))=N_grid_uw(grid_ij(i,1),grid_ij(i,2))+1;
            #        end
            #    end
            #    ph_grid_uw=ph_grid_uw./N_grid_uw;
            #    %ph_grid_uw(ph_grid_uw==inf)=nan;

        if i1 == 0:
            nzix = ph_grid[:] != 0
            n_ps_grid = np.sum(nzix)
            ph = np.zeros((n_ps_grid, n_ifg), dtype='int')

            lowfilt_flag = args[5]
            if lowfilt_flag == 'y':
                not_supported_param('lowfilt_flag', 'y')
                # ph_lowpass = ph
            else:
                ph_lowpass = []

            if predef_flag == 'y':
                not_supported_param('predef_flag', 'y')
                # ph_uw_predef=zeros(n_ps_grid,n_ifg,'single');
            else:
                ph_uw_predef = []

        goldfilt_flag = args[4]
        if goldfilt_flag == 'y' or lowfilt_flag == 'y':
            gold_alpha = args[6]
            [ph_this_gold, ph_this_low] = wrap_filt(ph_grid, prefilt_win, gold_alpha, [], lowfilt_flag)

            if lowfilt_flag == 'y':
                not_supported_param('lowfilt_flag', 'y')
                ph_lowpass[:, i1] = ph_this_low(nzix)

        ph = ph.astype('complex')
        if goldfilt_flag == 'y':
            ph[:, i1] = np.transpose(ph_this_gold)[np.transpose(nzix)]
        else:
            ph[:, i1] = np.transpose(ph_grid)[np.transpose(nzix)]

        if predef_flag == 'y':
            not_supported_param('predef_flag', 'y')
            # ph_uw_predef(:,i1)=ph_grid_uw(nzix);
            # ix=~isnan(ph_uw_predef(:,i1));
            # ph_diff=angle(ph(ix,i1).*conj(exp(j*ph_uw_predef(ix,i1))));
            # ph_diff(abs(ph_diff)>1)=nan; % if difference is too large, can't be sure which way to round
            # ph_uw_predef(ix,i1)=ph_uw_predef(ix,i1)+ph_diff; % set so ph_uw is filtered ph + integer num cycles

    n_ps = n_ps_grid
    print('   Number of resampled points: {}'.format(n_ps))

    nz_i_tmp = [np.nonzero(ph_grid[:, j]) for j in range(0, len(ph_grid[0]))]
    nz_i = (functools.reduce(lambda a, b: np.concatenate((a, b), axis=1), nz_i_tmp) + 1).reshape(-1, 1)
    nz_j_tmp = [np.tile(j + 1, len(nz_i_tmp[j][0])) for j in range(0, len(nz_i_tmp))]
    nz_j = functools.reduce(lambda a, b: np.concatenate((a, b), axis=0), nz_j_tmp).reshape(-1, 1)
    if pix_size == 0:
        xy = xy_in
    else:
        xy = np.concatenate(((np.array([i for i in range(1, n_ps + 1)])).reshape(-1, 1), (nz_j - 0.5) * pix_size,
                             (nz_i - 0.5) * pix_size), axis=1).astype('int')

    ij = np.concatenate((nz_i, nz_j), axis=1)

    uw_grid = {
        'ph': ph,
        'ph_in': ph_in,
        'ph_lowpass': ph_lowpass,
        'ph_uw_predef': ph_uw_predef,
        'ph_in_predef': ph_in_predef,
        'xy': xy,
        'ij': ij,
        'nzix': nzix,
        'grid_x_min': grid_x_min,
        'grid_y_min': grid_y_min,
        'n_i': n_i,
        'n_j': n_j,
        'n_ifg': n_ifg,
        'n_ps': n_ps,
        'grid_ij': grid_ij,
        'pix_size': pix_size
    }
    savemat('uw_grid.mat', uw_grid)
    uw_grid.clear()
