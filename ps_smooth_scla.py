import os, sys, platform

from collections import defaultdict
from itertools import permutations

from scipy.io import loadmat, savemat
from scipy.spatial import Delaunay

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

    arch = platform.platform().split('-')[0].lower()
    if arch == 'windows':
        use_triangle = 'n'
    else:
        tripath = os.system('which triangle > /dev/null');
        if tripath == 0:
            use_triangle = 'y'
        else:
            use_triangle = 'n';

    if use_triangle == 'y':
        nodename = 'scla.1.node'
    else:
        xy = ps['xy']
        tri = Delaunay(xy[:, 1:])
        n_edge = 3 * tri.npoints - 3 - len(tri.convex_hull)

    print('Number of arcs per ifg: {}'.format(n_edge))

