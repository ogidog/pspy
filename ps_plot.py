# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 18:43:52 2021

@author: anyuser
"""

import os, sys, time
import numpy as np
from scipy.io import loadmat, savemat
from ps_setref import ps_setref
from getparm import get_parm_value as getparm
from mat2plot import mat2plot

def env_oscilator_corr(envisat_flag, forced_sm_flag, path_to_task, op):

    if len(envisat_flag) == 0:
      
        platform = op['platform']
        
        if len(platform) == 0:
            if os.path.exists(path_to_task + 'master.res'):
                master_file = path_to_task + 'master.res'
            else:
                master_file = ''
            
            if len(master_file) > 0:
                text_string = open(master_file).read()
                
                if 'ASAR' in text_string:
                    platform = 'ENVISAT'
                
                else:
                    print('* Could not check if this is Envisat.')
              
            if 'ENVISAT' == platform:
                envisat_flag = 'y'
                print('* This is Envisat, oscilator drift is being removed.')
            else:
                 envisat_flag = 'n'
    
    ps = loadmat(path_to_task + 'ps2.mat', squeeze_me = True)
            
    if envisat_flag == 'y':
     

        lambda1 = getparm('lambda')
        envisat_resolution = 7.8
        oscilator_drift_corr_year = 3.87e-7
        oscilatior_corr_velocity = (envisat_resolution * ps['ij'][:,2]) * oscilator_drift_corr_year * 1000
        n_ifg = ps['n_image']
        delta_year = (ps['day'] - ps['master_day']) / 365.25
        oscilatior_corr_ifgs = -4 * np.pi / lambda1 * np.tile(oscilatior_corr_velocity, (1, n_ifg)) / 1000 * np.tile(delta_year.T, (ps['n_ps'], 1))
    
    else:
        
        n_ifg = ps['n_image']
        oscilatior_corr_ifgs = np.zeros((ps['n_ps'], n_ifg))
        oscilatior_corr_velocity = np.zeros((ps['n_ps'], 1))
    
    return(oscilatior_corr_ifgs, oscilatior_corr_velocity)

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

    x, residuals, rank, s = np.linalg.lstsq(Aw, Bw.T, rcond = None)
    return(x)

def ps_deramp(ps, ph_all):
    print('* Deramping computed on the fly')
    xy = ps['xy']
    n_ps = ps['n_ps']
    n_ifg = ps['n_ifg']
    
    if os.path.exists('deramp_degree.mat'):
        print('* Found <deramp_degree.mat> file will use that value to deramp')
    else:
        degree = 1
    
    if n_ifg != np.shape(ph_all)[1]:
        n_ifg = np.shape(ph_all)[1]
    
    if degree == 1:
        A = np.hstack((xy[:,1:] / 1000, np.ones((n_ps, 1))))
        
    if degree == 1.5:
        A = np.hstack((xy[:,1:] / 1000, xy[:,1] * xy[:,2] / 1000**2, np.ones((n_ps, 1))))
        
    if degree == 2:
        A = np.hstack((xy[:,1:]**2 / 1000**2, xy[:,1] * xy[:,2] / 1000**2, np.ones((n_ps, 1))))
        
    if degree == 3:
        A = np.hstack(((xy[:,1:]/1000)**3,  (xy[:,1]/1000)**2 * xy[:,2]/1000, (xy[:,2]/1000)**2 * xy[:,1]/1000, (xy[:,1:]/1000)**2, xy[:,1]/1000 * xy[:,2]/1000, np.ones((n_ps, 1))))
    
    out = np.zeros_like(ph_all)
    ph_ramp = np.zeros_like(ph_all)
    
    for k in range(n_ifg):
        ixzero = ph_all[:,k] == 0
        ixnnz = ph_all[:,k] != 0
        if n_ps - np.sum(ixzero) > 5:
            coeff = lscov_m(A[ixnnz,:], ph_all[ixnnz, k])
            ph_ramp[:,k] = np.matmul(A, coeff)
            out[:,k] = ph_all[:,k] - ph_ramp[:,k]
        else:
           print('* Ifg', k, 'is not deramped') 
      
    return(out)

def ps_plot(value_type):

    begin_time = time.time()
    
    fn_conv = {'v'    : 'Mean LOS velocity (MLV) in mm/yr',
               'v-d'  : 'Mean LOS velocity (MLV) in mm/yr minus smoothed dem error',
               'v-o'  : 'Mean LOS velocity (MLV) in mm/yr minus orbital ramps',
               'v-do' : 'Mean LOS velocity (MLV) in mm/yr minus smoothed dem error and orbital ramps'
               }
    
    if value_type in list(fn_conv):
        print()
    else:
        print('Unknown option. Exit')
    
    path_to_task = os.getcwd() + os.sep
        
    print()    
    print('********* Plotting **********')
    print('***** and saving data *******')
    print('')
    print('Work dir:', path_to_task)
    print('Option <ts> always in use')
    
    op = loadmat(path_to_task + 'parms.mat', squeeze_me = True)
    psver = str(2)
    
    psname = path_to_task + 'ps' + psver + '.mat'
    sclaname = path_to_task + 'scla' + psver + '.mat'
    sclasmoothname = path_to_task + 'scla_smooth' + psver + '.mat'
    phuwname = path_to_task + 'phuw' + psver + '.mat'
    ifgstdname = path_to_task + 'ifgstd' + psver + '.mat'
    
    ps = loadmat(psname, squeeze_me = True)
    uw = loadmat(phuwname, squeeze_me = True)    
    scla = loadmat(sclaname, squeeze_me = True)
    sclasmooth = loadmat(sclasmoothname, squeeze_me = True)
    
    day = ps['day']
    master_day = ps['master_day']
    # xy = ps['xy']
    # lonlat = ps['lonlat']
    n_ps = ps['n_ps']
    # n_ifg = ps['n_ifg']
    master_ix = np.sum(day[day < master_day]) + 1
    ref_ps = 0
    drop_ifg_index = op['drop_ifg_index']
    # small_baseline_flag = op['small_baseline_flag']
    # scla_deramp = op['scla_deramp']
###############################################################################
    ref_ifg = 0
    ifg_list = []
    # ref_radius_data = 1000
    # n_x = 0
    # n_y = 0
    # bands = []
    unwrap_ifg_index = np.setdiff1d(np.arange(ps['n_ifg']), drop_ifg_index)
    # units = 'rad'
    forced_sm_flag = 1
    ph_unw_eni_osci, v_envi_osci = env_oscilator_corr([], forced_sm_flag, path_to_task, op)
   
    # Code used for general <v> option
    ph_uw = uw['ph_uw']
    ph_uw = ph_uw - ph_unw_eni_osci
    # Code used for general <v> option
    
    if value_type == 'v-d':
        if 'C_ps_uw' in list(scla):
            ph_uw = ph_uw - scla['ph_scla'] - np.tile(v2c(scla['C_ps_uw']), (1, np.shape(ph_uw)[1]))
        else:
             ph_uw = ph_uw - scla['ph_scla']
    
    if value_type == 'v-o':
        ph_uw = ps_deramp(ps, ph_uw)
    
    if value_type == 'v-do':
        if 'C_ps_uw' in list(scla):
            ph_uw = ph_uw - scla['ph_scla'] - np.tile(v2c(scla['C_ps_uw']), (1, np.shape(ph_uw)[1]))
        else:
             ph_uw = ph_uw - scla['ph_scla']
        
        ph_uw = ps_deramp(ps, ph_uw)
        
    ### <ts> option block
    ph_uw_ts = ph_uw.copy()
    if 'C_ps_uw' in list(scla):
        ph_uw_ts = ph_uw_ts - np.tile(v2c(scla['C_ps_uw']), (1, np.shape(ph_uw_ts)[1]))    
    ### <ts>
    
    ph_all = np.zeros(n_ps)
    ref_ps = np.asarray(ps_setref({}, path_to_task, op), dtype = int)
    
    unwrap_ifg_index = np.setdiff1d(unwrap_ifg_index, ps['master_ix'] - 1)
    
    if len(ifg_list) > 0:
        unwrap_ifg_index = np.intersect1d(unwrap_ifg_index, ifg_list)
        ifg_list = []
    
    ph_uw = ph_uw[:, unwrap_ifg_index]
    day = day[unwrap_ifg_index]
    
    ph_uw = ph_uw - np.tile(np.nanmean(ph_uw[ref_ps, :], axis = 0), (n_ps, 1))
    
    ### ts
    ph_uw_ts = ph_uw_ts[:, unwrap_ifg_index]
    ph_uw_ts = ph_uw_ts - np.tile(np.nanmean(ph_uw_ts[ref_ps, :], axis = 0), (n_ps, 1))
    ### ts    
    
    if os.path.exists(ifgstdname) == False:
        sm_cov = np.eye(len(unwrap_ifg_index))

    else:
        ifgstd = loadmat(ifgstdname, squeeze_me = True)
        if 'ifg_std' in list(ifgstd):
            
            ifgvar = (ifgstd['ifg_std'] * np.pi / 181) ** 2
            sm_cov = ifgvar[unwrap_ifg_index]
        else:
            sm_cov = np.ones(len(unwrap_ifg_index))
      
    G = np.hstack((np.ones((len(day), 1)), v2c(day - master_day)))
    lambda1 = op['lambda']
    
    m = lscov_m(G, ph_uw.T, sm_cov)
    ph_all = -m[1,:].T * 365.25 / 4 / np.pi * lambda1 * 1000

    if len(ifg_list) == 0:
        ph_shape = np.shape(ph_all)
        if len(ph_shape) == 1:
            ifg_list = np.arange(0)
            ph_disp = ph_all
        else:
            ifg_list = np.arange(ph_shape[1])
            ph_disp = ph_all[:, ifg_list]
    
    reals_ix = np.isreal(ph_all)
    
    if len(reals_ix == True) == len(reals_ix):
        
        if ref_ifg != 0:
        
            if ref_ifg == -1:
                ph_disp = ph_disp - np.hstack((ph_disp[:,1], ph_disp[:,1:-1]))
            
            else:
                ph_disp = ph_disp - np.tile(ph_all[:,ref_ifg], (1, np.shape(ph_disp)[1]))
            
        else:
            ref_ifg = ps['master_ix'] - 1
    
        if len(ref_ps) > 0:
            if len(ifg_list) == 0:
                mean_ph = np.mean(ph_disp)
                ph_disp = ph_disp - mean_ph
                
            else:
                ref_ph = ph_disp[ref_ps,:]
                mean_ph = np.zeros(np.shape(ph_disp)[1])
                
                for i in range(np.shape(ph_disp)[1]):
                    mean_ph[i] = np.mean(ref_ph[np.invert(np.isnan(ref_ph[:,i])), i])
                    
                    if np.isnan(mean_ph[i]):
                        mean_ph[i] = 0
                        print('* Interferogram (', str(i), ') does not have a reference area.')

                ph_disp = ph_disp - np.tile(mean_ph, (n_ps, 1))

        # phsort = np.sort(ph_disp[np.invert(np.isnan(ph_disp))])
        
    else: # if isreal
        if ref_ifg == 0:
            ref_ifg = master_ix
            
        if ref_ifg == -1:
            ph_disp = ph_disp * np.conj(np.hstack((ph_disp[:, 1], ph_disp[:,1:-1])))
        
        if ref_ps != 0:
            ph_disp = ph_disp / abs(ph_disp)
            ref_ph = ph_disp[ref_ps,:]
            mean_ph = np.zeros(np.shape(ph_disp, 1))
            
            for i in range(np.shape(ph_disp)[1]):
                mean_ph[i] = sum(ref_ph[np.invert(np.isnan(ref_ph[:,i])), i])
    
            ph_disp = ph_disp * np.conj(np.tile(mean_ph, (n_ps, 1)))

    try:
        savemat(path_to_task + 'mean_v.mat', {'m':m}, oned_as = 'column')
    except:
        print('* Read access only, velocities not saved in home directory (ps_plot).')

    savename_ts = path_to_task + 'ps_plot_ts_' + value_type + '.mat'
    savetxt = path_to_task + 'ps_plot_ts_matname.txt'
    tf = open(savetxt, 'w')
    tf.write('ps_plot_ts_' + value_type + '.mat')
    tf.close()

    ph_mm = -ph_uw_ts * lambda1 * 1000 / ( 4 * np.pi)
    
    savedict = {'ph_mm':ph_mm,
                'lonlat':ps['lonlat'],
                'unwrap_ifg_index':unwrap_ifg_index,
                'ref_ps':ref_ps,
                'day':day,
                'n_ps':n_ps,
                'lambda':lambda1,
                'ifg_list':ifg_list,
                'master_day':master_day,
                'bperp':ps['bperp']}
    
    savemat(savename_ts, savedict, oned_as = 'column')
        
    try:
       savemat(path_to_task + 'ps_plot_' + value_type + '.mat', {'ph_disp':ph_disp, 'ifg_list':ifg_list, 'lonlat':ps['lonlat']}, oned_as = 'column')
    except:
       print('* Read access only, values not saved in home directory (ps_plot).')
   
    mat2plot(path_to_task + 'ps_plot_', value_type, 'html', 'marker', ps['lonlat'], ph_disp)
        
    print('Done at', int(time.time() - begin_time), 'sec')
    
if __name__ == "__main__":
    args = sys.argv
    print(args)
    ps_plot(args[1])