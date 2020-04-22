import sys
import numpy as np

from utils import *
from uw_grid_wrapped import uw_grid_wrapped
from uw_interp import uw_interp
from uw_sb_unwrap_space_time import uw_sb_unwrap_space_time
from uw_stat_costs import uw_stat_costs


def uw_3d(*args):
    # params: ph, xy, day, ifgday_ix, bperp, options

    if len(args) < 4:
        print('not enough arguments')
        sys.exit(0)

    if len(args) < 5:
        bperp = []

    if len(args) < 6:
        options = {}

    ifgday_ix = args[3]
    if len(ifgday_ix) == 0:
        single_master_flag = 1
    else:
        single_master_flag = 0

    options = args[5]
    valid_options = ['la_flag', 'scf_flag', 'master_day', 'grid_size', 'prefilt_win', 'time_win', 'unwrap_method',
                     'goldfilt_flag', 'lowfilt_flag', 'gold_alpha', 'n_trial_wraps', 'temp', 'n_temp_wraps',
                     'max_bperp_for_temp_est', 'variance', 'ph_uw_predef']
    invalid_options = set(options.keys()).difference(valid_options)
    if len(invalid_options) > 0:
        for item in invalid_options:
            break
        print('{} is an invalid option'.format(item))

    if 'master_day' not in options.keys():
        options['master_day'] = 0

    if 'grid_size' not in options.keys():
        options['grid_size'] = 5

    if 'prefilt_win' not in options.keys():
        options['prefilt_win'] = 16

    if 'time_win' not in options.keys():
        options['time_win'] = 365

    if 'unwrap_method' not in options.keys():
        if single_master_flag == 1:
            options['unwrap_method'] = '3D'
        else:
            options['unwrap_method'] = '3D_FULL'

    if 'goldfilt_flag' not in options.keys():
        options['goldfilt_flag'] = 'n'

    if 'lowfilt_flag' not in options.keys():
        options['lowfilt_flag'] = 'n'

    if 'gold_alpha' not in options.keys():
        options['gold_alpha'] = 0.8

    if 'n_trial_wraps' not in options.keys():
        options['n_trial_wraps'] = 6

    if 'la_flag' not in options.keys():
        options['la_flag'] = 'y'

    if 'scf_flag' not in options.keys():
        options['scf_flag'] = 'y'

    ph = args[0]
    if 'temp' not in options.keys():
        options['temp'] = []
    else:
        if len(options['temp']) != len(ph[0]):
            print('options["temp"] must be M x 1 vector where M is no. of ifgs')

    if 'n_temp_wraps' not in options.keys():
        options['n_temp_wraps'] = 2

    if 'max_bperp_for_temp_est' not in options.keys():
        options['max_bperp_for_temp_est'] = 100

    if 'variance' not in options.keys():
        options['variance'] = []

    if 'ph_uw_predef' not in options.keys():
        options['ph_uw_predef'] = []

    xy = args[1]
    if len(xy[1]) == 2:
        xy[:, 1:2] = xy[:, 0:1]

    day = args[2]
    if len(day) == 1:
        day = day.reshape(-1, 1)

    if options['unwrap_method'] == '3D' or options['unwrap_method'] == '3D_NEW':
        if len(np.unique(ifgday_ix[:, 0])) == 1:
            options['unwrap_method'] = '3D_FULL'
        else:
            options['lowfilt_flag'] = 'y'

    # TODO: uncomment
    # uw_grid_wrapped(ph, xy, options['grid_size'], options['prefilt_win'], options['goldfilt_flag'],
    #                options['lowfilt_flag'],
    #                options['gold_alpha'], options['ph_uw_predef'])

    # TODO: uncomment
    # ph = []
    # uw_interp()

    # TODO: uncomment
    # bperp = args[4]
    # uw_sb_unwrap_space_time(day, ifgday_ix, options['unwrap_method'], options['time_win'], options['la_flag'], bperp,
    #                        options['n_trial_wraps'], options['prefilt_win'], options['scf_flag'], options['temp'],
    #                        options['n_temp_wraps'], options['max_bperp_for_temp_est'])
    uw_stat_costs(options['unwrap_method'], options['variance'])

    return np.array(['ph_uw', 'msd'])
