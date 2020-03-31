import numpy as np
from scipy.io import loadmat
from getparm import get_parm_value as getparm


def ps_setref(ps2={}):
    if ps2 == {}:
        nargin = 0
        psver = loadmat('psver.mat')['psver'][0][0]
        psname = 'ps' + str(psver)
        ps2 = loadmat(psname + '.mat')
    else:
        nargin = 1
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

    if ref_radius == np.float('-inf'):
        ref_ps = 0
    else:
        ps2_lon = ps2['lonlat'][:, 0]
        ps2_lat = ps2['lonlat'][:, 1]
        ref_ps = [i for i in range(len(ps2_lon)) if
                  ps2_lon[i] > ref_lon[0] and ps2_lon[i] < ref_lon[1] and ps2_lat[i] > ref_lat[0] and ps2_lat[i] <
                  ref_lat[1]]

        if ref_radius < np.float('inf'):
            print("You set the param ref_radius={}, but not supported yet.".format(getparm('ref_radius')[0][0]))
            # ref_xy = llh2local(ref_centre_lonlat.T, ps2['ll0'][0]) * 1000;
            # xy=llh2local(ps2.lonlat(ref_ps,:)',ps2.ll0)*1000;
            # dist_sq=(xy(1,:)-ref_xy(1)).^2+(xy(2,:)-ref_xy(2)).^2;
            # ref_ps=ref_ps(dist_sq<=ref_radius^2);

    if nargin == 1:
        if len(ref_ps) == 0:
            print('None of your external data points have a reference, all are set as reference. \n')
            ref_ps = np.range(1, ps2['n_ps'][0][0])

    if nargin < 1:
        if ref_ps == 0:
            print('No reference set')
        else:
            print('{} ref PS selected'.format(len(ref_ps)))

    return ref_ps
