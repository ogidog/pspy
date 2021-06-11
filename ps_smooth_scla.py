import os, time
import numpy as np

from scipy.io import loadmat, savemat

from scipy.spatial import Delaunay
from matplotlib.tri import Triangulation

def ps_smooth_scla(*args):

    begin_time = time.time()
    
    if len(args) == 1: # To debug
        path_to_task = args[0] + os.sep
    
    else: # To essential run
        path_to_task = ''
    
    print()    
    print('******** Stage 7-2 *********')
    print('****** Smoothing scla ******')
    print('')
    print('Work dir:', os.getcwd() + os.sep)

    psver = str(2) # str(loadmat(path_to_patch + 'psver.mat', squeeze_me = True)['psver'])

    op = loadmat(path_to_task + 'parms.mat', squeeze_me = True)
    
    print('* Smoothing spatially-correlated look angle error.')
    
    psname = path_to_task + 'ps' + psver + '.mat'
    bpname = path_to_task + 'bp' + psver + '.mat'
    sclaname = path_to_task + 'scla' + psver + '.mat'
    sclasmoothname = path_to_task + 'scla_smooth' + psver + '.mat'

    ps = loadmat(psname, squeeze_me = True)
    scla = loadmat(sclaname, squeeze_me = True)

    K_ps_uw = scla['K_ps_uw']
    C_ps_uw = scla['C_ps_uw']
    ph_ramp = scla['ph_ramp']
    n_ps = ps['n_ps']

    print('* Number of points per ifg:', n_ps)

    xy = ps['xy']
    
    tri = Triangulation(xy[:, 1], xy[:, 2], triangles = Delaunay(xy[:, 1:]).simplices)
    edgs = tri.edges
    n_edge = len(edgs)
    
    print('* Number of arcs per ifg:', n_edge)

    Kneigh_min = np.zeros(n_ps) + np.float('inf')
    Kneigh_max = np.zeros(n_ps) + np.float('-inf')
    Cneigh_min = np.zeros(n_ps) + np.float('inf')
    Cneigh_max = np.zeros(n_ps) + np.float('-inf')

    for i in range(n_edge):
        ix = edgs[i, 0:2]

        Kneigh_min[ix] = np.min(np.hstack((np.reshape(Kneigh_min[ix], (2,1)), np.reshape(K_ps_uw[np.flip(ix)], (2,1)))), axis = 1)
        
        Kneigh_max[ix] = np.max(np.hstack((np.reshape(Kneigh_max[ix], (2,1)), np.reshape(K_ps_uw[np.flip(ix)], (2,1)))), axis = 1)
        
        Cneigh_min[ix] = np.min(np.hstack((np.reshape(Cneigh_min[ix], (2,1)), np.reshape(C_ps_uw[np.flip(ix)], (2,1)))), axis = 1)
        
        Cneigh_max[ix] = np.max(np.hstack((np.reshape(Cneigh_max[ix], (2,1)), np.reshape(C_ps_uw[np.flip(ix)], (2,1)))), axis = 1)

    ix1 = K_ps_uw > Kneigh_max
    ix2 = K_ps_uw < Kneigh_min
    K_ps_uw[ix1] = Kneigh_max[ix1]
    K_ps_uw[ix2] = Kneigh_min[ix2]

    ix1 = C_ps_uw > Cneigh_max
    ix2 = C_ps_uw < Cneigh_min
    C_ps_uw[ix1] = Cneigh_max[ix1]
    C_ps_uw[ix2] = Cneigh_min[ix2]

    bp = loadmat(bpname, squeeze_me = True)

    bperp_mat = np.concatenate((np.concatenate((bp['bperp_mat'][:, 0:ps['master_ix'] - 1], np.zeros(ps['n_ps']).reshape(-1, 1)), axis=1), bp['bperp_mat'][:, ps['master_ix'] - 1:]), axis=1)

    ph_scla = np.tile(np.reshape(K_ps_uw, (len(K_ps_uw), 1)), (1, len(bperp_mat[0]))) * bperp_mat

    sclasmooth = {
        'K_ps_uw': K_ps_uw,
        'C_ps_uw': C_ps_uw,
        'ph_scla': ph_scla,
        'ph_ramp': ph_ramp}
        
    savemat(sclasmoothname, sclasmooth, oned_as = 'column')
    
    print('Done at', int(time.time() - begin_time), 'sec')
    
if __name__ == "__main__":
    # For testing
    test_path = 'C:\\Users\\Ryzen\\Documents\\PYTHON\\stampsexport'
    ps_smooth_scla(test_path)