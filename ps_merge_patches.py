import os, time

import numpy as np
from scipy.io import loadmat, savemat

from getparm import get_parm_value as getparm
from llh2local import llh2local
from utils import compare_objects, compare_mat_with_number_values, not_supported_value, not_supported, not_supported_param

def intersect(A, B):
    if len(B) > 0:
        A = A[..., 0] + 1j * A[..., 1]
        B = B[..., 0] + 1j * B[..., 1]
        C = np.intersect1d(A, B, return_indices = True)
        return(np.dstack((C[0].real, C[0].imag))[0].astype('int'), C[1], C[2])
    else:
        return([], [], [])

def ps_merge_patches(*args):
    
    begin_time = time.time()
    
    if len(args) == 1: # To debug
        path_to_task = args[0] + os.sep
    
    else: # To essential run
        path_to_task = ''
    
    print()    
    print('********** Stage 5 *********')
    print('****** Merge patches *******')
    print('')
    print('Work dir:', os.getcwd() + os.sep)

    psver = str(2) # str(loadmat(path_to_patch + 'psver.mat', squeeze_me = True)['psver'])

    op = loadmat(path_to_task + 'parms.mat', squeeze_me = True)

    grid_size = op['merge_resample_size']
    merge_stdev = op['merge_standard_dev']
    small_baseline_flag = op['small_baseline_flag']
    
    phase_accuracy = 10 * np.pi / 180
    min_weight = 1 / merge_stdev ** 2
    np.random.seed(seed = 1001)
    max_coh = (np.abs(sum(np.exp(complex(0, 1) * np.random.randn(1000, 1) * phase_accuracy))) / 1000)[0]

    psname = 'ps' + psver
    phname = 'ph' + psver
    rcname = 'rc' + psver
    pmname = 'pm' + psver
    phuwname = 'phuw' + psver
    sclaname = 'scla' + psver
    sclasbname = 'scla_sb' + psver
    scnname = 'scn' + psver
    bpname = 'bp' + psver
    laname = 'la' + psver
    incname = 'inc' + psver
    hgtname = 'hgt' + psver

    if os.path.exists(path_to_task + 'patch.list'):
        dirname = []
        fid = open(path_to_task + 'patch.list', 'r')
        line = fid.readline().strip()
        while line:
            dirname.append(line)
            line = fid.readline().strip()
        fid.close()
    else:
        dirlist = os.listdir()
        dirname = [item for item, item in dirlist if 'PATCH_' in item]

    n_patch = len(dirname)
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

    for i in range(n_patch):

        print('* Processing directory.', dirname[i])
        os.chdir(path_to_task + dirname[i])

        ps = loadmat(psname + '.mat')
                    
        n_ifg = ps['n_ifg']
        
        if 'n_image' in ps.keys():
            n_image = ps['n_image']
        else:
            n_image = ps['n_ifg']

        patch = {'ij': []}
        fid = open('patch_noover.in', 'r')
        line = fid.readline().strip()
        
        while line:
            patch['ij'].append(int(line))
            line = fid.readline().strip()
        fid.close()
        
        patch['ij'] = np.array(patch['ij'])
        
        ij1 = ps['ij']

        ix = ((ij1[:, 1] >= patch['ij'][2] - 1) & (ij1[:, 1] <= patch['ij'][3] - 1) & (ij1[:, 2] >= patch['ij'][0] - 1) & (ij1[:, 2] <= patch['ij'][1] - 1))
        
        if sum(ix) == 0:
            ix_no_ps = 1
        else:
            ix_no_ps = 0

        if grid_size == 0:
            C, IA, IB = intersect(ps['ij'][ix, 1:3], ij)
            remove_ix = np.concatenate((remove_ix, IB), axis = 0)
            C, IA, IB = intersect(ps['ij'][:, 1:3], ij)
            ix_ex = np.ones(int(ps['n_ps'])).astype('bool')
            ix_ex[IA] = 0
            ix[ix_ex] = 1
        else:
            not_supported_value('grid_size', grid_size)

        if grid_size == 0:
            ij = np.vstack((ij, ps['ij'][ix, 1:]))
            lonlat = np.vstack((lonlat, ps['lonlat'][ix, :]))
        else:
            not_supported_value('grid_size', grid_size)

        if os.path.exists(phname + '.mat'):
            phin = loadmat(phname)
            ph_w = phin['ph']
            phin.clear()
        else:
            ph_w = ps['ph']

        if 'ph_w' in locals() or 'ph_w' in globals():
            if grid_size == 0:
                if len(ph) == 0:
                    ph = ph.reshape(0, np.shape(ph_w)[1])
                ph = np.concatenate((ph, ph_w[ix, :]), axis = 0)
            else:
                not_supported_value('grid_size', grid_size)

            ph_w = []

        rc = loadmat(rcname + '.mat', squeeze_me = True)
        
        if grid_size == 0:
            if len(ph_rc) == 0:
                ph_rc = ph_rc.reshape(0, np.shape(rc['ph_rc'])[1])
            ph_rc = np.concatenate((ph_rc, rc['ph_rc'][ix, :]), axis = 0)
            
            if small_baseline_flag != 'y':
                if len(ph_reref) == 0:
                    ph_reref = ph_reref.reshape(0, np.shape(rc['ph_reref'])[1])
                ph_reref = np.concatenate((ph_reref, rc['ph_reref'][ix, :]), axis = 0)
        else:
            not_supported_value('grid_size', grid_size)

        rc.clear()

        pm = loadmat(pmname + '.mat', squeeze_me = True)
        if grid_size == 0:
            
            if len(ph_patch) == 0:
                ph_patch = ph_patch.reshape(0, np.shape(pm['ph_patch'])[1])
            ph_patch = np.concatenate((ph_patch, pm['ph_patch'][ix, :]), axis = 0)
            
            if 'ph_res' in pm.keys():
                if len(ph_res) == 0:
                    ph_res = ph_res.reshape(0, np.shape(pm['ph_res'])[1])
                ph_res = np.concatenate((ph_res, pm['ph_res'][ix, :]), axis = 0)
            
            if 'K_ps' in pm.keys():
                tmp = pm['K_ps'][ix]
                if len(K_ps) == 0:
                    K_ps = np.reshape(tmp, (len(tmp), 1))
                else:
                    K_ps = np.vstack((K_ps, np.reshape(tmp, (len(tmp), 1))))
            
            if 'C_ps' in pm.keys():
                tmp = pm['C_ps'][ix]
                if len(C_ps) == 0:
                    C_ps = np.reshape(tmp, (len(tmp), 1))
                else:
                    C_ps = np.vstack((C_ps, np.reshape(tmp, (len(tmp), 1))))
            
            if 'coh_ps' in pm.keys():
                tmp = pm['coh_ps'][ix]
                if len(coh_ps) == 0:
                    coh_ps = np.reshape(tmp, (len(tmp), 1))
                else:
                    coh_ps = np.vstack((coh_ps, np.reshape(tmp, (len(tmp), 1))))
        else:
            not_supported_value('grid_size', grid_size)

        pm.clear()

        bp = loadmat(bpname + '.mat', squeeze_me = True)
        if grid_size == 0:
            if len(bperp_mat) == 0:
                bperp_mat = bperp_mat.reshape(0, np.shape(bp['bperp_mat'])[1])
            bperp_mat = np.concatenate((bperp_mat, bp['bperp_mat'][ix, :]), axis=0)
        else:
            not_supported_value('grid_size', grid_size)

        bp.clear()

        if os.path.exists(laname + '.mat'):
            lain = loadmat(laname + '.mat', squeeze_me = True)
            if grid_size == 0:
                tmp = lain['la'][ix]
                if len(la) == 0:
                    la = np.reshape(tmp, (len(tmp), 1))
                else:
                    la = np.vstack((la, np.reshape(tmp, (len(tmp), 1))))
            else:
                not_supported_value('grid_size', grid_size)

            lain.clear()

        incin = {}
        if os.path.exists(incname + '.mat'):
            not_supported()
            incin = loadmat(incname + '.mat', squeeze_me = True)

        if os.path.exists(hgtname + '.mat'):
            hgtin = loadmat(hgtname + '.mat', squeeze_me = True)
            if grid_size == 0:
                tmp = hgtin['hgt'][ix]
                if len(hgt) == 0:
                    hgt = np.reshape(tmp, (len(tmp), 1))
                else:
                    hgt = np.vstack((hgt, np.reshape(tmp, (len(tmp), 1))))
            else:
                not_supported_value('grid_size', grid_size)
        else:
            hgtin = {}
                
        if grid_size == 0:
            if os.path.exists(phuwname + '.mat'):
                not_supported()

            else:
                zeros = np.zeros((np.sum(ix), int(n_image)))
                if len(ph_uw) == 0:
                    ph_uw = ph_uw.reshape(0, np.shape(zeros)[1])
                ph_uw = np.concatenate((ph_uw, zeros), axis = 0)

            if os.path.exists(sclaname + '.mat'):
                not_supported()
                scla = loadmat(sclaname + '.mat', squeeze_me = True)

            if small_baseline_flag == 'y':
                not_supported_param('small_baseline_flag', 'y')

            if os.path.exists(scnname + '.mat'):
                not_supported()

        os.chdir('..')

    ps_new = ps
    n_ps_orig = len(ij)
    keep_ix = ix_ex = np.ones((n_ps_orig, 1)).astype('bool').flatten()
    keep_ix[remove_ix.astype('int')] = 0
    lonlat_save = lonlat
    coh_ps_weed = coh_ps[keep_ix]
    lonlat = lonlat[keep_ix, :]

    dummy, I = np.unique(lonlat, return_index = True, axis = 0)
    dups = np.setxor1d(I, np.array([*range(len(lonlat))]))
    keep_ix_num = np.nonzero(keep_ix)[0]

    for i in range(len(dups)):
        print('* Found duplicates via lon/lat!')

    if len(dups) > 0:
        lonlat = lonlat_save[keep_ix, :]
        print('* Dropped pixels via duplicate lon/lat:', len(dups))

    lonlat_save = []

    ll0 = (np.max(lonlat, axis = 0) + np.min(lonlat, axis = 0)) / 2
    xy = llh2local(lonlat.T, ll0) * 1000
    xy = xy.T
    sort_x = xy[np.argsort(xy[:, 0], kind = 'stable')]
    sort_y = xy[np.argsort(xy[:, 1], kind = 'stable')]

    n_pc = int(np.round(len(xy) * 0.001))
    bl = np.mean(sort_x[0:n_pc, :], axis = 0)
    tr = np.mean(sort_x[len(sort_x) - n_pc - 1:, :], axis = 0)
    br = np.mean(sort_y[0:n_pc, :], axis = 0)
    tl = np.mean(sort_y[len(sort_y) - n_pc - 1:, :], axis = 0)

    try:
        heading = getparm('heading')[0][0][0]
    except:
        heading = 0
    theta = (180 - heading) * np.pi / 180
    if theta > np.pi:
        theta = theta - 2 * np.pi

    rotm = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], dtype = np.float64)
    
    xy = xy.T
    xynew = np.dot(rotm, xy)

    if max(xynew[0, :]) - min(xynew[0, :]) < max(xy[0, :]) - min(xy[0, :]) and max(xynew[1, :]) - min(xynew[1, :]) < max(xy[1, :]) - min(xy[1, :]):
        xy = xynew
        print('* Rotating xy by {}  degrees.'.format(str(theta * 180 / np.pi)))

    xynew = []
    xy = xy.astype('float32').T
    xy_list = xy.tolist()
    xy_sort = np.array(sorted(xy_list, key = lambda t: (t[1], t[0])))
    sort_ix = np.array(sorted(range(len(xy_list)), key = lambda s: (xy_list[s][1], xy_list[s][0])))

    xy = xy[sort_ix, :]
    xy = np.concatenate((np.array([*range(len(xy))]).reshape(-1, 1) + 1, xy), axis = 1)
    xy[:, 1:3] = np.round(xy[:, 1:3] * 1000) / 1000
    lonlat = lonlat[sort_ix, :]

    all_ix = np.array([*range(len(ij))]).reshape(-1, 1)
    keep_ix = all_ix[keep_ix]
    sort_ix = keep_ix[sort_ix]

    n_ps = len(sort_ix)
    print('* Writing merged dataset (contains {} pixels)'.format(str(n_ps)))

    ij = ij[sort_ix.flatten(), :]

    ph_rc = ph_rc[sort_ix.flatten(), :]
    ph_rc[ph_rc != 0] = np.divide(ph_rc[ph_rc != 0], np.abs(ph_rc[ph_rc != 0]))
    if small_baseline_flag != 'y':
        ph_reref = ph_reref[sort_ix.flatten(), :]

    rc['ph_rc'] = ph_rc
    rc['ph_reref'] = ph_reref
    savemat(rcname + '.mat', rc)
    ph_rc = []
    ph_reref = []

    if len(ph_uw) == n_ps_orig:
        ph_uw = ph_uw[sort_ix.flatten(), :]
        savemat(phuwname + '.mat', {'ph_uw': ph_uw})
    ph_uw = []

    ph_patch = ph_patch[sort_ix.flatten(), :]
    if len(ph_res) == n_ps_orig:
        ph_res = ph_res[sort_ix.flatten(), :]
    else:
        ph_res = []
    if len(K_ps) == n_ps_orig:
        K_ps = K_ps[sort_ix.flatten(), :]
    else:
        K_ps = []
    if len(C_ps) == n_ps_orig:
        C_ps = C_ps[sort_ix.flatten(), :]
    else:
        C_ps = []
    if len(coh_ps) == n_ps_orig:
        coh_ps = coh_ps[sort_ix.flatten(), :]
    else:
        coh_ps = []
    
    pm['ph_patch'] = ph_patch
    pm['ph_res'] = ph_res
    pm['K_ps'] = K_ps
    pm['C_ps'] = C_ps
    pm['coh_ps'] = coh_ps
    savemat(pmname + '.mat', pm)
    ph_patch = []
    ph_res = []
    K_ps = []
    C_ps = []
    coh_ps = []

    if len(ph_scla) == n_ps:
        ph_scla = ph_scla[sort_ix.flatten(), :]
        K_ps_uw = K_ps_uw[sort_ix.flatten(), :]
        C_ps_uw = C_ps_uw[sort_ix.flatten(), :]
        scla['ph_scla'] = ph_scla
        scla['K_ps_uw'] = K_ps_uw
        scla['C_ps_uw'] = C_ps_uw
        savemat(sclaname + '.mat', scla)
    
    ph_scla = []
    K_ps_uw = []
    C_ps_uw = []

    if small_baseline_flag == 'y' and len(ph_scla_sb) == n_ps:
        not_supported_param('small_baseline_flag', 'y')

    if len(ph_scn_slave) == n_ps:
        not_supported()

    if len(ph) == n_ps_orig:
        ph = ph[sort_ix.flatten(), :]
    else:
        ph = []
    phin['ph'] = ph
    savemat(phname + '.mat', phin)

    if len(la) == n_ps_orig:
        la = la[sort_ix.flatten(), :]
    else:
        la = []
    lain['la'] = la
    savemat(laname + '.mat', lain)

    if len(inc) == n_ps_orig:
        inc = inc[sort_ix.flatten(), :]
    else:
        inc = []
    incin['inc'] = inc
    savemat(incname + '.mat', incin)

    if len(hgt) == n_ps_orig:
        hgt = hgt[sort_ix.flatten(), :]
    else:
        hgt = []
    hgtin['hgt'] = hgt
    savemat(hgtname + '.mat', hgtin)

    bperp_mat = bperp_mat[sort_ix.flatten(), :]
    bp['bperp_mat'] = bperp_mat
    savemat(bpname + '.mat', bp)
    bperp_mat = []

    ps_new['n_ps'] = n_ps
    ps_new['ij'] = np.concatenate(((np.array([*range(n_ps)]) + 1).reshape(-1, 1), ij), axis=1)
    ps_new['xy'] = xy
    ps_new['lonlat'] = lonlat
    savemat(psname + '.mat', ps_new)
    ps_new.clear()

    savemat('psver.mat', {'psver': psver})

    if os.path.exists('mean_amp.flt'):
        os.remove('mean_amp.flt')

    if os.path.exists('amp_mean.mat'):
        os.remove('amp_mean.mat')

    print('Done at', int(time.time() - begin_time), 'sec')
    
if __name__ == "__main__":
    # For testing
    test_path = 'C:\\Users\\Ryzen\\Documents\\PYTHON\\stampsexport'
    ps_merge_patches(test_path)        