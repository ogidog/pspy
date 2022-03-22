#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:06:44 2020

@author: anyuser
"""

import os
import time
import numpy as np
from scipy.io import loadmat, savemat
from scipy.spatial import Delaunay
from matplotlib.tri import Triangulation

def tr(matrix):
    out = np.transpose(matrix)
    return(out)

def v2c(v):
    m = len(v)
    out = np.reshape(v, (m, 1))
    return(out)

def v2r(v):
    m = len(v)
    out = np.reshape(v, (1, m))
    return(out)

def lscov_m(A, B, w = None):

    if w is None:
        Aw = A.copy()
        Bw = np.transpose(B.copy())
    else:
        W = np.sqrt(np.diag(np.array(w).flatten()))
        Aw = np.dot(W, A)
        Bw = np.dot(B.T, W)

    x, residuals, rank, s = np.linalg.lstsq(Aw, Bw.T, rcond = 1e-10)
    return(x)

def lscov_p(A, B, w):
    W = np.diag(w)
    solve = (((np.linalg.inv((A.T.dot(np.linalg.inv(W)).dot(A)))).dot(A.T)).dot(np.linalg.inv(W))).dot(B)
    return(solve)

def swapcols(a):
    out = np.hstack((v2c(a[:,1]), v2c(a[:,0])))
    return(out)

###
def ps_weed(*args):
    
    begin_time = time.time()
    
    if len(args) == 2: # To debug
        path_to_task = args[0] + os.sep
        path_to_patch = path_to_task + 'PATCH_' + str(args[1]) + os.sep
    
    else: # To essential run
        path_to_task = os.sep.join(os.getcwd().split(os.path.sep)[:-1]) + os.sep
        path_to_patch = os.getcwd() + os.sep
    
    print()    
    print('********** Stage 4 *********')
    print('*********** Weed ***********')
    print('')
    print('Work dir:', path_to_patch)

    psver = str(1) # str(loadmat(path_to_patch + 'psver.mat', squeeze_me = True)['psver'])
    
    op = loadmat(path_to_task + 'parms.mat', squeeze_me = True)
        
    if op['weed_neighbours'] == 'y':
        no_weed_adjacent = 0
    else:
        no_weed_adjacent = 1
    
    if op['weed_standard_dev'] >= np.pi and op['weed_max_noise'] >= np.pi:
        no_weed_noisy = 1
    else:
        no_weed_noisy = 0
    
    time_win = op['weed_time_win']
    weed_standard_dev = op['weed_standard_dev']
    weed_max_noise = op['weed_max_noise']
    # weed_zero_elevation = op['weed_zero_elevation']
    # weed_neighbours = op['weed_neighbours']
    drop_ifg_index = op['drop_ifg_index']

    hgtname = 'hgt' + psver + '.mat'
    
    fnpath = path_to_patch + 'ps' + psver + '.mat'
    if os.path.exists(fnpath):
        ps = loadmat(fnpath, squeeze_me = True)
    else:
        print('* No ps file')
        
    fnpath = path_to_patch + 'select' + psver + '.mat'
    if os.path.exists(fnpath):
        sl = loadmat(fnpath, squeeze_me = True)
    else:
        print('* No sl file')

    fnpath = path_to_patch + 'ph' + psver + '.mat'
    if os.path.exists(fnpath):
        ph = loadmat(fnpath, squeeze_me = True)['ph']
    else:
        ph = ps['ph']

    fnpath = path_to_patch + 'pm' + psver + '.mat'
    if os.path.exists(fnpath):
        pm = loadmat(fnpath, squeeze_me = True)
    else:
        print('* No pm file')
        
    n_ifg = ps['n_ifg']
    ifg_index =  np.setdiff1d(np.arange(n_ifg), drop_ifg_index - 1)        

    day = ps['day']
    bperp = ps['bperp']
    # master_day = ps['master_day']
    
    ix = sl['ix'] - 1
    if 'keep_ix' in sl:
        keep_ix = sl['keep_ix'].astype(bool)
    else:
        keep_ix = np.ones(len(ix), dtype = bool)
    
    ix2 = ix[keep_ix]
    K_ps2 = sl['K_ps2'][keep_ix]
    C_ps2 = sl['C_ps2'][keep_ix]
    coh_ps2 = sl['coh_ps2'][keep_ix]
    
    ij2 = ps['ij'][ix2,:]
    xy2 = ps['xy'][ix2,:]
    lonlat2 = ps['lonlat'][ix2,:]    
    ph2 = ph[ix2,:]
    ph_patch2 = pm['ph_patch'][ix2,:] #########################################
    
    if 'ph_res2' in sl:
        ph_res2 = sl['ph_res2'][keep_ix,:]
    else:
        ph_res2 = []
    
    n_ps_other = 0

    if os.path.isfile(path_to_patch + hgtname):
        ht = loadmat(path_to_patch + hgtname, squeeze_me = True)
        hgt = ht['hgt'][ix2]

    n_ps_low_D_A = len(ix2)
    n_ps = n_ps_low_D_A + n_ps_other
    ix_weed = np.asarray(np.ones(n_ps), dtype = np.bool)
    # print(n_ps_low_D_A,' low D_A ps, ', n_ps_other,' high D_A ps')
    
    old_ps = len(ix_weed[ix_weed == True])
    print('* Pixels loaded:', old_ps)

    if no_weed_adjacent == 0: # Not tested condition!!!!!!!!!!!!!!!!!!!!
        # print('INITIALISE NEIGHBOUR MATRIX')
                
        ij_shift = ij2[:,1:2] + np.tile(np.asarray([2,2]) - np.min(ij2[:,1:2]), (n_ps, 1))
        neigh_ix = np.zeros((np.max(ij_shift[:,0]) + 1, np.max(ij_shift[:,1]) + 1))
        miss_middle = np.ones((3, 3))
        miss_middle[1, 1] = 0
        
        for i in range(n_ps):
            neigh_this = neigh_ix[ij_shift[i,0] - 1 : ij_shift[i,0] + 1, ij_shift[i,1] - 1 : ij_shift[i,1] + 1]
            neigh_this[neigh_this == 0 & miss_middle] = i
            neigh_ix[ij_shift[i,0] - 1 : ij_shift[i,0] + 1, ij_shift[i,1] - 1 : ij_shift[i,1] + 1] = neigh_this
        
        # print('FIND NEIGHBOURS')
    
        neigh_ps = [[] for i in range(n_ps)]
        for i in range(n_ps):
            my_neigh_ix = neigh_ix[ij_shift[i, 0], ij_shift[i,1]]
            if my_neigh_ix != 0:
                neigh_ps[my_neigh_ix] = [neigh_ps[my_neigh_ix], i]
        
        for i in range(n_ps):
            if len(neigh_ps[i]) != 0:
                same_ps = [i]
                i2 = 1
                while i2 <= len(same_ps):
                    ps_i = same_ps[i2]
                    same_ps.append(neigh_ps[ps_i])
                    neigh_ps[ps_i] = []
                    i2 = i2 + 1
                
                same_ps = np.unique(np.assarray(same_ps))
                high_coh = np.argmax(coh_ps2[same_ps])
                low_coh_ix = np.zeros_like(same_ps) + 1
                low_coh_ix[high_coh] = 0
                ix_weed[same_ps[low_coh_ix]] = 0

        print('* Pixels kept after dropping adjacent items:', len(ix_weed == True))
    # Not tested condition!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    # Output how many PS are left after weeding zero elevations out
    if 'weed_zero_elevation' == 'y' and 'hgt' in globals(): 
        sea_ix = np.asarray(np.where(hgt < 1e-6)).flatten()
        ix_weed[sea_ix] = 0
        print('* Pixels kept after weeding zero elevation:', len(ix_weed == True))
    
    xy_weed = xy2[ix_weed, :]
    # Update PS inofmration
    n_ps = len(xy_weed)
    
    # Remove dupplicated points
    # Some non-adjacent pixels are allocated the same lon/lat by DORIS.
    # If duplicates occur, the pixel with the highest coherence is kept.
    ix_weed_num = np.asarray(np.nonzero(ix_weed)).flatten()
    vals, inds = np.unique(xy_weed[:,1:], axis = 0, return_index = True)
    dups = np.setxor1d(inds, np.arange(np.sum(ix_weed)))     # pixels with duplicate lon/lat
    # Not tested condition !!!
    if len(dups) > 0:
        for i in dups:
            dups_ix_weed = [i]
            for j in range(len(xy_weed)):
                if xy_weed[j,1] == xy_weed[i,1] and xy_weed[j,2] == xy_weed[i,2]:
                    dups_ix_weed.append(j)
            dups_ix = ix_weed_num[dups_ix_weed]
            max_ind = np.argmax(coh_ps2[dups_ix])
            for j in range(len(dups_ix)):
                if j != max_ind:
                    ix_weed[dups_ix[j]] = 0       # drop dups with lowest coh
    
    if len(dups) > 0:
        xy_weed = xy2[ix_weed,:]
        print('* Pixels dropped via duplicate lon/lat:', str(len(dups)))
    # Not tested condition !!!
    
    # Update PS inofmration
    n_ps = len(ix_weed == 1)
    ix_weed2 = np.asarray(np.ones(n_ps), dtype = np.bool)
        
    # Weedign noisy pixels
    ps_std = np.zeros(n_ps)
    ps_max = np.zeros(n_ps)
    
    # include a check to see if any points are actually left
    if n_ps != 0:
        if no_weed_noisy == 0:
           
            # Code for external triangulation are dropped !!!
            x1 = xy_weed[:,1]
            y1 = xy_weed[:,2]
            xy1 = xy_weed[:,1:]
            tri = Triangulation(x1, y1, triangles = Delaunay(xy1).simplices)
            # tri = Triangulation(x, y, triangles = Delaunay(np.hstack((y, x))).simplices)
            edgs_py = swapcols(tri.edges)

            edgs = edgs_py[np.lexsort(np.fliplr(edgs_py).T)]
            
            n_edge = len(edgs)
            print('* Found arcs:', n_edge)
            
            ph_weed = ph2[ix_weed,:] * np.exp(-1j * np.matmul(v2c(K_ps2[ix_weed]), v2r(bperp))) # subtract range error 
            ph_weed_abs = np.abs(ph_weed)
            ph_weed_abs[ph_weed_abs == 0] = 1
            ph_weed = ph_weed / ph_weed_abs
            ph_weed[:, ps['master_ix'] - 1] = np.exp(1j * (C_ps2[ix_weed]))  # add master noise if small_baseline_flag == 'y'
            
            edge_std = np.zeros(n_edge)
            edge_max = np.zeros(n_edge)
            
            dph_space = ph_weed[edgs[:,1],:] * np.conj(ph_weed[edgs[:,0],:])
            dph_space = dph_space[:,ifg_index]
            n_use = len(ifg_index)
            
            # print('Estimating noise for all arcs...')
            dph_smooth = np.zeros((n_edge, n_use), dtype = np.complex128)
            dph_smooth2 = np.zeros((n_edge, n_use), dtype = np.complex128)
            
            for i1 in range(n_use):
                time_diff = day[ifg_index[i1]] - day[ifg_index]
                weight_factor = np.exp(-(time_diff**2) / 2 / time_win**2)
                weight_factor = weight_factor / np.sum(weight_factor)
                
                dph_mean = np.sum(dph_space * np.tile(v2r(weight_factor), (n_edge, 1)), axis = 1)
                dph_mean_adj = np.angle(dph_space * np.tile(v2c(np.conj(dph_mean)), (1, n_use))) # subtract weighted mean
                
                G = np.hstack((np.ones((n_use, 1)), v2c(time_diff)))
                ### ls_cov
                m = lscov_m(G, tr(dph_mean_adj), weight_factor.flatten()) # weighted least-sq to find best-fit local line
                
                dph_mean_adj = np.angle(np.exp(1j * (dph_mean_adj - tr(np.matmul(G, m))))) # subtract first estimate
                ### ls_cov
                m2 = lscov_m(G, tr(dph_mean_adj), weight_factor.flatten()) # weighted least-sq to find best-fit local line
                
                dph_smooth[:,i1] = dph_mean * np.exp(1j * (tr(m[0,:]) + tr(m2[0,:]))) # add back weighted mean
                weight_factor[i1] = 0 # leave out point itself
                dph_smooth2[:,i1] = np.sum(dph_space * np.tile(weight_factor, (n_edge, 1)), axis = 1)

            dph_noise = np.angle(dph_space * np.conj(dph_smooth))
            dph_noise2 = np.angle(dph_space * np.conj(dph_smooth2))
            ifg_var = np.var(dph_noise2, axis = 0)
            ### ls_cov
            K = lscov_m(v2c(bperp[ifg_index]), tr(dph_noise), v2r(1/ifg_var).flatten()) # estimate arc dem error
            
            dph_noise = dph_noise - np.matmul(tr(K), v2r(bperp[ifg_index]))
            edge_std = np.std(dph_noise, ddof = 1, axis = 1)
            edge_max = np.max(np.abs(dph_noise), axis = 1)
                    
            # print('Estimating max noise for all pixels...')
            ps_std = np.zeros(n_ps) + np.Inf
            ps_max = np.zeros(n_ps) + np.Inf

            for i in range(n_edge):
                a1 = v2c(ps_std[edgs[i,:]])
                a2 = np.asarray([[edge_std[i]], [edge_std[i]]])
                tmp = np.hstack((a1, a2))
                ps_std[edgs[i,:]] = np.min(tmp, axis = 1)
                a1 = v2c(ps_max[edgs[i,:]])
                a2 = np.asarray([[edge_max[i]], [edge_max[i]]])
                tmp = np.hstack((a1, a2))
                ps_max[edgs[i,:]] = np.min(tmp, axis = 1)
                
            ix_weed2 = np.asarray(np.zeros(n_ps), dtype = np.bool)
            for i in range(n_ps):
                if ps_std[i] < weed_standard_dev and ps_max[i] < weed_max_noise:
                    ix_weed2[i] = True
            
            ix_weed[ix_weed] = ix_weed2
            n_ps = len(ix_weed2[ix_weed2 == True])
            
            print('* Pixels kept after dropping noisy items:', n_ps)

    # Keep information about number of PS left.
    if os.path.exists(path_to_patch + 'no_ps_info.mat'):
        stamps_step_no_ps = np.zeros((5, 1)) # Keep for the first 5 steps only
    else:
        loadmat(path_to_patch + 'no_ps_info.mat');
        # Reset as we are currently re-processing
        stamps_step_no_ps[3:] = 0
    
    if n_ps == 0:
        print('* No ps points left. Updating the stamps log for this.')
        # Update the flag indicating no ps left in step 3
        stamps_step_no_ps[3] = 1
    
    savemat(path_to_patch + 'no_ps_info.mat', {'stamps_step_no_ps':stamps_step_no_ps})

    ###########################################################################
    # Saving the results
    ###########################################################################
    
    savedict = {'ix_weed':v2c(ix_weed),
                'ix_weed2':v2c(ix_weed2),
                'ps_std':v2c(ps_std),
                'ps_max':v2c(ps_max),
                'ifg_index':v2r(ifg_index + 1)}  
    
    savefile = path_to_patch + 'weed' + psver + '.mat'

    savemat(savefile, savedict)
    
    # Save weeded version of files with psver = 2
    
    psver = str(2)
        
    coh_ps = coh_ps2[ix_weed]
    K_ps = K_ps2[ix_weed]
    C_ps = C_ps2[ix_weed]
    ph_patch = ph_patch2[ix_weed,:]
    
    if len(ph_res2) > 0:
        ph_res = ph_res2[ix_weed,:]
    else:
        ph_res = ph_res2
    
    savedict = {'ph_patch':ph_patch,
                'ph_res':ph_res,
                'coh_ps':v2c(coh_ps),
                'K_ps':v2c(K_ps),
                'C_ps':v2c(C_ps)}  
    
    savefile = path_to_patch + 'pm' + psver + '.mat'

    savemat(savefile, savedict)
        
    ph2 = ph2[ix_weed,:]
    ph = ph2

    savedict = {'ph':ph}  
    
    savefile = path_to_patch + 'ph' + psver + '.mat'

    savemat(savefile, savedict)    
        
    ps['xy'] = xy2[ix_weed,:]
    ps['ij'] = ij2[ix_weed,:]
    ps['lonlat'] = lonlat2[ix_weed,:]
    ps['n_ps'] = len(ph2)
    ps['bperp'] = v2c(ps['bperp'])
    ps['day'] = v2c(ps['day'])
    ps.pop('sort_ix', None)
    
    savefile = path_to_patch + 'ps' + psver + '.mat'

    savemat(savefile, ps) 

    savefile = path_to_patch + 'hgt' + psver + '.mat'    
    if os.path.exists(savefile):
        hgt = v2c(hgt[ix_weed])
        savefile = path_to_patch + 'hgt' + psver + '.mat'
        savedict = {'hgt':hgt}

        savemat(savefile, savedict) 
    
    if os.path.exists(path_to_patch + 'la1.mat'):
        la = loadmat(path_to_patch + 'la1.mat')
        la = la['la'][ix2][ix_weed]

        savefile = path_to_patch + 'la2.mat'    
        savedict = {'la':la}

        savemat(savefile, savedict) 
    
    if os.path.exists(path_to_patch + 'inc1.mat'):
        inc = loadmat(path_to_patch + 'inc1.mat')
        inc = [inc.inc[ix2]]

        savefile = path_to_patch + 'inc2.mat'    
        savedict = {'inc':inc[ix_weed]}

        savemat(savefile, savedict)         
    
    if os.path.exists(path_to_patch + 'bp1.mat'):
        bp = loadmat(path_to_patch + 'bp1.mat')
        bperp_mat = bp['bperp_mat'][ix2,:][ix_weed,:]

        savefile = path_to_patch + 'bp2.mat'    
        savedict = {'bperp_mat':bperp_mat}

        savemat(savefile, savedict) 
    
    print('Done at', int(time.time() - begin_time), 'sec')

if __name__ == "__main__":
    # For testing
    test_path = 'C:\\Users\\anyuser\\Documents\\PYTHON\\stampsexport'
    ps_weed(test_path, 1)