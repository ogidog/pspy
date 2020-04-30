import os

import numpy as np
from scipy.io import loadmat

from getparm import get_parm_value as getparm
from utils import compare_objects


def ps_merge_patches(*args):
    print('Merging patches...\n')

    if len(args) < 1:
        psver = 2

    small_baseline_flag = getparm('small_baseline_flag')[0][0]
    grid_size = getparm('merge_resample_size')[0][0][0]
    merge_stdev = getparm('merge_standard_dev')[0][0][0]
    phase_accuracy = 10 * np.pi / 180
    min_weight = 1 / np.power(merge_stdev, 2)
    np.random.seed(seed=1001)
    max_coh = (np.abs(sum(np.exp(complex(0, 1) * np.random.randn(1000, 1) * phase_accuracy))) / 1000)[0]

    psname = 'ps' + str(psver)
    phname = 'ph' + str(psver)
    rcname = 'rc' + str(psver)
    pmname = 'pm' + str(psver)
    phuwname = 'phuw' + str(psver)
    sclaname = 'scla' + str(psver)
    sclasbname = 'scla_sb' + str(psver)
    scnname = 'scn' + str(psver)
    bpname = 'bp' + str(psver)
    laname = 'la' + str(psver)
    incname = 'inc' + str(psver)
    hgtname = 'hgt' + str(psver)

    if os.path.exists('patch.list'):
        dirname = {'name': []}
        fid = open('patch.list', 'r')
        line = fid.readline().strip()
        while line:
            dirname['name'].append(line)
            line = fid.readline().strip()
        fid.close()
    else:
        dirname = {'name': np.array([dir for dir in os.listdir() if 'PATCH_' in dir])}

    n_patch = len(dirname['name'])
    remove_ix = []
    ij = np.zeros((0, 2))
    lonlat = np.zeros((0, 2))
    ph = np.zeros((0, 0))
    ph_rc = np.zeros((0, 0))
    ph_reref = np.zeros((0, 0))
    ph_uw = np.zeros((0, 0))
    ph_patch = np.zeros((0, 0))
    ph_res = np.zeros((0, 0))
    ph_scla = np.zeros((0, 0))
    ph_scla_sb = np.zeros((0, 0))
    ph_scn_master = np.zeros((0, 0))
    ph_scn_slave = np.zeros((0, 0))
    K_ps = np.zeros((0, 0))
    C_ps = np.zeros((0, 0))
    coh_ps = np.zeros((0, 0))
    K_ps_uw = np.zeros((0, 0))
    K_ps_uw_sb = np.zeros((0, 0))
    C_ps_uw = np.zeros((0, 0))
    C_ps_uw_sb = np.zeros((0, 0))
    bperp_mat = np.zeros((0, 0))
    la = np.zeros((0, 0))
    inc = np.zeros((0, 0))
    hgt = np.zeros((0, 0))
    amp = np.zeros((0, 0))

    for i in range(0, n_patch):
        if dirname['name'][i]:
            print('   Processing {}\n'.format(dirname['name'][i]))
            os.chdir('.' + os.path.sep + dirname['name'][i])
            ps = loadmat(psname + '.mat')
            n_ifg = ps['n_ifg'][0][0]
            if 'n_image' in ps.keys():
                n_image = ps['n_image'][0][0]
            else:
                n_image = ps['n_ifg'][0][0]

            patch = {'ij': []}
            fid = open('patch_noover.in', 'r')
            line = fid.readline().strip()
            while line:
                patch['ij'].append(int(line))
                line = fid.readline().strip()
            fid.close()
            patch['ij'] = np.array(patch['ij'])
            ix = ((ps['ij'][:, 1] >= patch['ij'][2] - 1) & (ps['ij'][:, 1] <= patch['ij'][3] - 1) & (
                    ps['ij'][:, 2] >= patch['ij'][0] - 1) & (ps['ij'][:, 2] <= patch['ij'][1] - 1))
            if sum(ix) == 0:
                ix_no_ps = 1
            else:
                ix_no_ps = 0

            if grid_size == 0:
                C, IA, IB = np.intersect1d(ps['ij'][ix, 1:3], ij, return_indices=True)
                remove_ix=[remove_ix,IB]
                #diff = compare_objects(ix, 'ix')
                print('ggg')
