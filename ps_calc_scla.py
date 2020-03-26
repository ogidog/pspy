import sys
from inspect import signature
from scipy.io import loadmat


def ps_calc_scla(use_small_baselines, coest_mean_vel):
    print('');
    print('Estimating spatially-correlated look angle error...')

    sig = signature(ps_calc_scla)
    params = sig.parameters
    if len(params) < 1:
        use_small_baselines = 0;
    if len(params) < 2:
        coest_mean_vel = 0;

    # TODO: Implement getparm() function
    small_baseline_flag = 'n'  #getparm('small_baseline_flag', 1);
    drop_ifg_index = []  #getparm('drop_ifg_index', 1);
    scla_method = 'L2'  #getparm('scla_method', 1);
    scla_deramp = 'n'  #getparm('scla_deramp', 1);
    subtr_tropo = 'n'  #getparm('subtr_tropo', 1);
    tropo_method = 'a_l'  #getparm('tropo_method', 1);

    if use_small_baselines != 0:
        if small_baseline_flag != 'y':
            print('   Use small baselines requested but there are none')
            sys.exit()

    if use_small_baselines==0:
        # TODO: Implement getparm() function
        scla_drop_index= [] #getparm('scla_drop_index',1);
    else:
        # TODO: Implement getparm() function
        scla_drop_index= [] #getparm('sb_scla_drop_index',1);
        print('   Using small baseline interferograms\n')

    psver = loadmat('psver.mat')[0][0]

    sys.exit()
