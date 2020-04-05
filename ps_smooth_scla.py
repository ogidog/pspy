import sys
from scipy.io import loadmat, savemat
from getparm import get_parm_value as getparm


def ps_smooth_scla(use_small_baselines):
    print('\nSmoothing spatially-correlated look angle error...')

    scn_wavelength = 100;
    small_baseline_flag = getparm('small_baseline_flag')

    psver = loadmat('psver.mat')
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
