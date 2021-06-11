import os
import time
import numpy as np

from scipy.io import loadmat, savemat

def ps_correct_phase(*args):
    
    begin_time = time.time()
       
    if len(args) == 1: # To debug
        path_to_task = args[0] + os.sep
        path_to_patch = path_to_task + 'PATCH_1' + os.sep
    
    else: # To essential run
        path_to_task = os.sep.join(os.getcwd().split(os.path.sep)[:-1]) + os.sep
        path_to_patch = os.getcwd() + os.sep
    
    print()    
    print('********** Stage 5 *********')
    print('******* Correct phase ******')
    print('')
    print('Work dir:', path_to_patch)

    psver = str(2) # str(loadmat(path_to_patch + 'psver.mat', squeeze_me = True)['psver'])
    
    # op = loadmat(path_to_task + 'parms.mat', squeeze_me = True)
    
    psname = path_to_patch + 'ps' + psver + '.mat'
    phname = path_to_patch + 'ph' + psver + '.mat'
    pmname = path_to_patch + 'pm' + psver + '.mat'
    bpname = path_to_patch + 'bp' + psver + '.mat'

    ps = loadmat(psname, squeeze_me = True)
    pm = loadmat(pmname, squeeze_me = True)
    bp = loadmat(bpname, squeeze_me = True)

    if os.path.exists(phname):
        phin = loadmat(phname, squeeze_me = True)
        ph = phin['ph']
        phin = {}
    else:
        ph = ps['ph']

    K_ps = pm['K_ps']
    K_ps1 = np.reshape(K_ps, (len(K_ps), 1))
    C_ps = pm['C_ps']
    C_ps1 = np.reshape(C_ps, (len(C_ps), 1))
    master_ix = (sum(ps['master_day'] > ps['day']) + 1) - 1
    
    bperp_mat = np.concatenate((bp['bperp_mat'][:, 0:ps['master_ix'] - 1], np.zeros((ps['n_ps'], 1)), bp['bperp_mat'][:, ps['master_ix'] - 1:]), axis = 1)
    
    a = np.add(np.multiply(np.tile(K_ps1, (1, ps['n_ifg'])), bperp_mat), np.tile(C_ps1, (1, ps['n_ifg'])))
    
    ph_rc = np.multiply(ph, (np.exp(complex(0, -1) * a)))
    
    ph_reref = np.concatenate((pm['ph_patch'][:, 0:master_ix], np.ones((ps['n_ps'], 1)), pm['ph_patch'][:, master_ix:]), axis = 1)
    
    rcname = path_to_patch + 'rc' + psver + '.mat'
    
    rc2 = {'ph_rc': ph_rc, 'ph_reref': ph_reref}
        
    savemat(rcname, rc2, oned_as = 'column')
    
    print('Done at', int(time.time() - begin_time), 'sec')
    
if __name__ == "__main__":
    # For testing
    test_path = 'C:\\Users\\Ryzen\\Documents\\PYTHON\\stampsexport'
    ps_correct_phase(test_path)