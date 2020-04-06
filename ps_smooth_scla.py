import os, sys, platform
import numpy as np

4
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

    edgs = []
    xy = ps['xy']
    tri = Delaunay(xy[:, 1:])
    for vertices in tri.simplices:
        edgs.append([vertices[0], vertices[1]])
        edgs.append([vertices[1], vertices[2]])
        edgs.append([vertices[2], vertices[0]])

    for edge in edgs:
        if edgs.count([edgs[0][0], edgs[0][1]]) > 1 or edgs.count([edgs[0][1], edgs[0][0]]) > 1:
            print(edge)

    # result = list(filter(lambda el: (edgs.count([el[0], el[1]]) + edgs.count([el[1], el[0]]) == 1), edgs))
    n_edge = 3 * tri.npoints - 3 - len(tri.convex_hull)
    print('Number of arcs per ifg: {}'.format(n_edge))

    Kneigh_min = np.full((n_ps, 1), np.float('inf'))
    Kneigh_max = np.full((n_ps, 1), np.float('-inf'))
    Cneigh_min = np.full((n_ps, 1), np.float('inf'))
    Cneigh_max = np.full((n_ps, 1), np.float('-inf'))

    for i in range(n_edge):
        print()
