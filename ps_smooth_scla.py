import sys
import numpy as np
from triangle import triangulate

from scipy.io import loadmat, savemat
from getparm import get_parm_value as getparm


def ps_smooth_scla(use_small_baselines):
    print('\nSmoothing spatially-correlated look angle error...')

    scn_wavelength = 100;
    small_baseline_flag = getparm('small_baseline_flag')

    psver = loadmat('psver.mat')['psver'][0][0]
    psname = 'ps' + str(psver)
    bpname = 'bp' + str(psver)

    if use_small_baselines == 0:
        sclaname = 'scla' + str(psver)
        sclasmoothname = 'scla_smooth' + str(psver)
    else:
        print("You set the param use_small_baselines={}, but not supported yet.".format(
            getparm('use_small_baselines')[0][0]))
        sys.exit(0)
        # sclaname=['scla_sb',num2str(psver)];
        # sclasmoothname=['scla_smooth_sb',num2str(psver)];

    ps = loadmat(psname + '.mat')
    scla = loadmat(sclaname + '.mat')
    K_ps_uw = scla['K_ps_uw']
    C_ps_uw = scla['C_ps_uw']
    ph_ramp = scla['ph_ramp']
    scla.clear()

    n_ps = ps['n_ps'][0][0]
    n_ifg = ps['n_ifg'][0][0]

    print('Number of points per ifg: {}'.format(n_ps))

    xy = ps['xy']
    tri = triangulate({'vertices': xy[:, 1:]}, opts='e')
    edgs = tri['edges']
    n_edge = len(edgs)
    print('Number of arcs per ifg: {}'.format(n_edge))

    Kneigh_min = np.full((n_ps, 1), np.float('inf'))
    Kneigh_max = np.full((n_ps, 1), np.float('-inf'))
    Cneigh_min = np.full((n_ps, 1), np.float('inf'))
    Cneigh_max = np.full((n_ps, 1), np.float('-inf'))

    for i in range(n_edge):
        ix = edgs[i, 0:2]
        Kneigh_min[ix] = np.amin(np.concatenate((Kneigh_min[ix], K_ps_uw[np.flip(ix)]), axis=1), axis=1).reshape(-1, 1)
        Kneigh_max[ix] = np.amax(np.concatenate((Kneigh_max[ix], K_ps_uw[np.flip(ix)]), axis=1), axis=1).reshape(-1, 1)
        Cneigh_min[ix] = np.amin(np.concatenate((Cneigh_min[ix], C_ps_uw[np.flip(ix)]), axis=1), axis=1).reshape(-1, 1)
        Cneigh_max[ix] = np.amax(np.concatenate((Cneigh_max[ix], C_ps_uw[np.flip(ix)]), axis=1), axis=1).reshape(-1, 1)
        if i % 100000 == 0:
            print('{} arcs processed'.format(i))

    ix1 = K_ps_uw > Kneigh_max;
    ix2 = K_ps_uw < Kneigh_min;
    K_ps_uw[ix1] = Kneigh_max[ix1]
    K_ps_uw[ix2] = Kneigh_min[ix2]

    print()
