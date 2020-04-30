import os

import numpy as np
from getparm import get_parm_value as getparm


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
        line = fid.readline()
        while line:
            dirname['name'].append(line)
            line = fid.readline()
        fid.close()
    else:
        dirname = {'name': np.array([dir for dir in os.listdir() if 'PATCH_' in dir])}

    n_patch = len(dirname['name'])
    remove_ix=[]
    ij=np.zeros((0,2))
    lonlat=np.zeros((0,2))
    ph=np.zeros((0,0))
    ph_rc=zeros(0,0);
    ph_reref=zeros(0,0);
    ph_uw=zeros(0,0);
    ph_patch=zeros(0,0);
    ph_res=zeros(0,0);
    ph_scla=zeros(0,0,'single');
    ph_scla_sb=zeros(0,0,'single');
    ph_scn_master=zeros(0,0);
    ph_scn_slave=zeros(0,0);
    K_ps=zeros(0,0);
    C_ps=zeros(0,0);
    coh_ps=zeros(0,0);
    K_ps_uw=zeros(0,0,'single');
    K_ps_uw_sb=zeros(0,0,'single');
    C_ps_uw=zeros(0,0,'single');
    C_ps_uw_sb=zeros(0,0,'single');
    bperp_mat=zeros(0,0,'single');
    la=zeros(0,0);
    inc=zeros(0,0);
    hgt=zeros(0,0);
    amp=zeros(0,0,'single');
