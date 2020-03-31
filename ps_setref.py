import numpy as np
from scipy.io import loadmat
from getparm import get_parm_value as getparm


def ps_setref(ps2={}):

    if ps2 == {}:
        psver = loadmat('psver.mat')['psver'][0][0]
        psname = 'ps' + str(psver)
        ps2 = loadmat(psname + '.mat')
    else:
        psver = loadmat('psver.mat')['psver'][0][0]
        psname = 'ps' + str(psver)
        ps_temp = loadmat(psname + '.mat')
        ps2['ll0'] = ps_temp['ll0']
        ps2['n_ps'] = len(ps2['lonlat'][0])

    [ref_lon, parmname] = getparm('ref_x');

    if parmname == 'ref_x':
        ps2_x = ps2['xy'][:, 1]
        ps2_y = ps2['xy'][:, 2]
        ref_x = getparm('ref_x')[0]
        ref_y = getparm('ref_y')[0]
        ref_ps = [i for i in range(len(ps2_x)) if
                  ps2_x[i] > ref_x[0] and ps2_x[i] < ref_x[1] and ps2_y[i] > ref_y[0] and ps2_y[i] < ref_y[1]]
    else:
        ref_lon = getparm('ref_lon')[0][0]
        ref_lat = getparm('ref_lat')[0][0]
        ref_centre_lonlat = getparm('ref_centre_lonlat')[0][0]
        ref_radius = getparm('ref_radius')[0][0][0]

    if ref_radius==np.float('-inf'):
        ref_ps = 0
    else:
        ref_ps=find(ps2.lonlat(:,1)>ref_lon(1)&ps2.lonlat(:,1)<ref_lon(2)&ps2.lonlat(:,2)>ref_lat(1)&ps2.lonlat(:,2)<ref_lat(2

    print()
