import sys, os, shutil
import numpy as np
from scipy.io import savemat, loadmat

from ps_calc_ifg_std import ps_calc_ifg_std
from ps_correct_phase import ps_correct_phase
from ps_parms_default import ps_parms_default
from getparm import get_parm_value as getparm
from setpsver import setpsver
from utils import not_supported_param

# from ps_est_gamma_quick import ps_est_gamma_quick
from ps_est_gamma_quick_ZR import ps_est_gamma_quick

from ps_load_initial_gamma import ps_load_initial_gamma
from ps_select import ps_select
from ps_weed import ps_weed
from ps_merge_patches import ps_merge_patches
from ps_unwrap import ps_unwrap
from ps_calc_scla import ps_calc_scla
from ps_smooth_scla import ps_smooth_scla

def main(args):
    quick_est_gamma_flag = getparm('quick_est_gamma_flag')[0][0]
    reest_gamma_flag = getparm('select_reest_gamma_flag')[0][0]
    unwrap_method = getparm('unwrap_method')[0][0]
    unwrap_prefilter_flag = getparm('unwrap_prefilter_flag')[0][0]
    small_baseline_flag = getparm('small_baseline_flag')[0][0]
    insar_processor = getparm('insar_processor')[0][0]
    scn_kriging_flag = getparm('scn_kriging_flag')[0][0]

    if small_baseline_flag == 'y' :
        print('Small_baseline_flag == <y> not supported. Exit.')
        sys.exit(0)
        
    # print(args)
    
    try:
        start_step = int(args[1])
        end_step = int(args[2])
        
    except:
        print('Bad args')

    if 'mpi' in args:
        runmode = 'mpi'
    else:
        runmode = 'single'
        
    if len(args) < 2 or (not 'start_step' in globals() and not 'start_step' in locals()):
        start_step = 1

    if len(args) < 3 or (not 'end_step' in globals() and not 'end_step' in locals()):
        end_step = 8

    if len(args) < 4 or (not 'patches_flag' in globals() and not 'patches_flag' in locals()):
        
        if start_step < 6:
            patches_flag = 'y'
        else:
            patches_flag = 'n'

    if len(args) < 4 or (not 'est_gamma_parm' in globals() and not 'est_gamma_parm' in locals()):
        est_gamma_parm = 0

    if len(args) < 5 or (not 'patch_list_file' in globals() and not 'patch_list_file' in locals()):
        patch_list_file = 'patch.list';
        new_patch_file = 0
    else:
        new_patch_file = 1

    if len(args) < 6 or (not 'stamps_PART_limitation' in globals() and not 'stamps_PART_limitation' in locals()):
        stamps_PART_limitation = 0
    
    stamps_PART1_flag = 'y'
    stamps_PART2_flag = 'y'
    
    if stamps_PART_limitation == 1:
        stamps_PART2_flag = 'n'

    if stamps_PART_limitation == 2:
        stamps_PART1_flag = 'n'

    if patches_flag == 'y':
        if os.path.exists(patch_list_file):
            patchdir = {'name': []}
            fid = open(patch_list_file, 'r')
            for line in fid:
                patchdir['name'].append(line.strip())
            fid.close()
        else:
            not_supported_param('patches_flag', patches_flag)
            # patchdir=dir('PATCH_*');
            # patchdir = patchdir(find(~cellfun(@(x) strcmpi(x,'patch_noover.in'),{patchdir(:).name})));

        if len(patchdir.keys()) == 0:
            not_supported_param('patches_flag', patches_flag)
            # patches_flag='n';
        else:
            ps_parms_default
            patches_flag = 'y'

    if patches_flag != 'y':
        patchdir = {'name': '.'}
        print('Will process current directory only.')
    else:
        print('Will process patch subdirectories.')

    currdir = os.getcwd()

    start_step_or = start_step
    
    if stamps_PART1_flag == 'y':
        
        patches = patchdir['name']
        for i in range(len(patches)):
            if len(patches[i]) != 0:
                # Set active direcoty to patch
                os.chdir(patches[i])
                
                patchsplit = os.getcwd().split(os.path.sep)

                if not os.path.exists('no_ps_info.mat'):
                    stamps_step_no_ps = np.zeros((5, 1))
                    stamps_step_no_ps = {'stamps_step_no_ps': stamps_step_no_ps}
                    savemat('no_ps_info.mat', stamps_step_no_ps)

                if start_step_or == 0:
                    if os.path.exists('weed1.mat'):
                        start_step = 5
                        setpsver(2)
                    else:
                        if os.path.exists('select1.mat'):
                            start_step = 4
                        else:
                            if os.path.exists('pm1.mat'):
                                start_step = 3
                            else:
                                if os.path.exists('ps1.mat'):
                                    start_step = 2
                                else:
                                    start_step = 1

                    if start_step > end_step:
                        print(patchsplit[len(patchsplit) - 1] + ': already up to end stage {}'.format(str(end_step)))
                    else:
                        print(patchsplit[len(patchsplit) - 1] + ': complete up to stage {}'.format(str(end_step - 1)))

                if start_step == 1:
                    # print('\n##################\n' +
                    #       '##### Step 1 #####\n' +
                    #       '##################\n')

                    # print('Directory is {}'.format(patchsplit[len(patchsplit) - 1]))
                    

                    if insar_processor == "gamma" or insar_processor == 'snap':
                        ps_load_initial_gamma()
                    else:
                        if insar_processor == 'gsar':
                            not_supported_param('insar_processor', insar_processor)

                        else:
                            if insar_processor == 'isce':
                                not_supported_param('insar_processor', insar_processor)

                        no_ps_info = loadmat('no_ps_info.mat', squeeze_me = True)
                        
                        # Reset as we are currently re-processing
                        stamps_step_no_ps = no_ps_info['stamps_step_no_ps'].flatten()
                        stamps_step_no_ps[0:len(stamps_step_no_ps) - 1] = 0
                        stamps_step_no_ps = list(stamps_step_no_ps.reshape(-1, 1))
                        savemat('no_ps_info.mat', {"stamps_step_no_ps": stamps_step_no_ps})

                    if start_step <= 4:
                        setpsver(1)
###----------------------------------------------------------------------------
                if start_step <= 2 and end_step >= 2:
                    # print('\n##################\n' +
                    #       '##### Step 2 #####\n' +
                    #       '##################\n')

                    # print('Directory is {}'.format(patchsplit[len(patchsplit) - 1]))

                    # Check if step 1 had more than 0 PS points
                    stamps_step_no_ps = loadmat("no_ps_info.mat", squeeze_me = True)["stamps_step_no_ps"]
                    
                    # Reset as we are currently re-processing
                    stamps_step_no_ps[1:] = 0

                    # Run step 2 when there are PS left in step 1
                    if stamps_step_no_ps[1] == 0:
                        if quick_est_gamma_flag == 'y':
                            
                            ps_est_gamma_quick(runmode)
                            # ps_est_gamma_quick()
                        else:
                            not_supported_param(quick_est_gamma_flag, quick_est_gamma_flag)

                    else:
                        stamps_step_no_ps[1] = 1
                        print('No PS left in step 1, so will skip step 2')
                        
                    savemat('no_ps_info.mat', {"stamps_step_no_ps": stamps_step_no_ps})
###----------------------------------------------------------------------------
                if start_step <= 3 and end_step >= 3:
                    # print('\n##################\n' +
                    #       '##### Step 3 #####\n' +
                    #       '##################\n')
                    
                    # print('Directory is {}'.format(patchsplit[len(patchsplit) - 1]))
                    
                    # Check if step 2 had more than 0 PS points
                    stamps_step_no_ps = loadmat("no_ps_info.mat", squeeze_me = True)["stamps_step_no_ps"]
                    
                    # Reset as we are currently re-processing
                    stamps_step_no_ps[2:] = 0
                    
                    # Run step 3 when there are PS left in step 2
                    if stamps_step_no_ps[2] == 0:
                        
                        ps_select(runmode) 
                        
                    else:
                        stamps_step_no_ps[2] = 1
                        print('No PS left in step 3, so will skip step 3')                    
                    
                    savemat('no_ps_info.mat', {"stamps_step_no_ps": stamps_step_no_ps})
###----------------------------------------------------------------------------
                if start_step <= 4 and end_step >= 4:
                    # print('\n##################\n' +
                    #       '##### Step 4 #####\n' +
                    #       '##################\n')
                    
                    # print('Directory is {}'.format(patchsplit[len(patchsplit) - 1]))

                    # Check if step 3 had more than 0 PS points
                    stamps_step_no_ps = loadmat("no_ps_info.mat", squeeze_me = True)["stamps_step_no_ps"]
                    
                    # Reset as we are currently re-processing
                    stamps_step_no_ps[3:] = 0       # keep for the first 5 steps only

                    # Run step 4 when there are PS left in step 3
                    if stamps_step_no_ps[3] == 0:
                            
                        ps_weed() 
                        
                    else:
                        print('No PS left in step 4, so will skip step 4 \n')
                        stamps_step_no_ps[3] = 1

                    savemat('no_ps_info.mat', {"stamps_step_no_ps": stamps_step_no_ps})
###----------------------------------------------------------------------------
                if start_step <= 5 and end_step >= 5:
                    
                    # print('Directory is {}'.format(patchsplit[len(patchsplit) - 1]))
                    
                    no_ps_info = loadmat('no_ps_info.mat', squeeze_me = True)
                    stamps_step_no_ps = no_ps_info['stamps_step_no_ps']
                    stamps_step_no_ps[4:] = 0
                    
                    if stamps_step_no_ps[3] == 0:
                        
                        ps_correct_phase()
                    
                    else:
                        print('No PS left in step 4, so will skip step 5')
                        stamps_step_no_ps[4] = 1
                        
                    no_ps_info['stamps_step_no_ps'] = stamps_step_no_ps
                    savemat('no_ps_info.mat', no_ps_info)

                os.chdir(currdir)

    patchsplit = os.getcwd().split(os.path.sep)

    if stamps_PART2_flag == 'y':
        
        if patches_flag == 'y':
            fid = open('patch.list_new', 'w')
            for i in range(0, len(patchdir['name'])):
                filename_PS_check = patchdir['name'][i] + os.path.sep + 'no_ps_info.mat'
                keep_patch = 1
                if os.path.exists(filename_PS_check):
                    filename_PS_check = loadmat(filename_PS_check, squeeze_me = True)
                    
                    if sum(filename_PS_check['stamps_step_no_ps']) >= 1:
                        keep_patch = 0

                if keep_patch == 1:
                    fid.write(patchdir['name'][i] + '\n')

                if i == len(patchdir['name']) - 1:
                    fid.close()

            shutil.copy('patch.list', 'patch.list_old')
            shutil.copy('patch.list_new', 'patch.list')

        if start_step <= 5 and end_step >= 5:
            abord_flag = 0
            if patches_flag == 'y':
                
                # print('Directory is {}'.format(patchsplit[len(patchsplit) - 1]))
                
                ps_merge_patches()
                
            else:
                if os.path.exists('no_ps_info.mat'):
                    no_ps_info = loadmat('no_ps_info.mat', squeeze_me = True)
                    
                    if sum(no_ps_info['stamps_step_no_ps']) >= 1:
                        abord_flag = 1

            if abord_flag == 0:
                ps_calc_ifg_std()
            
            else:
                print('No PS left in step 4, so will skip step 5')

        if start_step <= 6 and end_step >= 6:
            # print('\n##################\n' +
            #       '##### Step 6 #####\n' +
            #       '##################\n')

            # print("Dirrectory is " + os.getcwd())

            ps_unwrap()

        if start_step <= 7 and end_step >= 7:
            # print('\n##################\n' +
            #       '##### Step 7 #####\n' +
            #       '##################\n')

            # print("Dirrectory is " + os.getcwd())

            ps_calc_scla()
            ps_smooth_scla()

if __name__ == "__main__":
    args = sys.argv

    main(args)