import numpy as np
from scipy.io import loadmat
from getparm import get_parm_value as getparm


def ps_setref(ps2 = {}, *args):
    
    path_to = args[0]
    ops = args[1]
    
    if ps2 == {}:
        nargin = 0
        psver = str(loadmat(path_to + 'psver.mat', squeeze_me = True)['psver'])
        psname = path_to + 'ps' + psver + '.mat'
        ps2 = loadmat(psname)
    else:
        nargin = 1
        psver = str(loadmat(path_to + 'psver.mat', squeeze_me = True)['psver'])
        psname = path_to + 'ps' + psver + '.mat'
        ps_temp = loadmat(psname, squeeze_me = True)
        ps2['ll0'] = ps_temp['ll0']
        ps2['n_ps'] = len(ps2['lonlat'])

    ref_lon = ops['ref_lon']
    ref_lat = ops['ref_lat']
    ref_centre_lonlat = ops['ref_centre_lonlat']
    ref_radius = ops['ref_radius']

    if ref_radius == np.float('-inf'):
        ref_ps = 0
    else:
        ps2_lon = ps2['lonlat'][:, 0]
        ps2_lat = ps2['lonlat'][:, 1]
        ref_ps = [i for i in range(len(ps2_lon)) if ps2_lon[i] > ref_lon[0] and ps2_lon[i] < ref_lon[1] and ps2_lat[i] > ref_lat[0] and ps2_lat[i] < ref_lat[1]]

        if ref_radius < np.float('inf'):
            print('* You set the param ref_radius != inf, but not supported yet (ps_setref).')

    if nargin == 1:
        if len(ref_ps) == 0:
            print('* None of your external data points have a reference, all are set as reference (ps_setref).')
            ref_ps = np.range(1, ps2['n_ps'])

    if nargin < 1:
        if ref_ps == 0:
            print('* No reference set (ps_setref).')
        else:
            print(len(ref_ps), 'reference PS selected (ps_setref).')

    return(ref_ps)