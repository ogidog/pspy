import sys, os
import numpy as np

from scipy.io import savemat

from ps_parms_default import ps_parms_default
from ps_unwrap import ps_unwrap
from ps_calc_scla import ps_calc_scla
from ps_smooth_scla import ps_smooth_scla
from getparm import get_parm_value as getparm
from utils import not_supported_param


def main(args):
    quick_est_gamma_flag = getparm('quick_est_gamma_flag')[0][0]
    reest_gamma_flag = getparm('select_reest_gamma_flag')[0][0]
    unwrap_method = getparm('unwrap_method')[0][0]
    unwrap_prefilter_flag = getparm('unwrap_prefilter_flag')[0][0]
    small_baseline_flag = getparm('small_baseline_flag')[0][0]
    insar_processor = getparm('insar_processor')[0][0]
    scn_kriging_flag = getparm('scn_kriging_flag')[0][0]

    try:
        start_step = int(args[1])
        end_step = int(args[2])
    except:
        print()

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
        not_supported_param('patches_flag', patches_flag)
        # patchdir(1).name='.';
        # logit('Will process current directory only')
    else:
        print('Will process patch subdirectories')

    currdir = os.getcwd()

    start_step_or = start_step
    if stamps_PART1_flag == 'y':
        for i in range(0, len(patchdir)):
            if 'name' in patchdir.keys():
                os.chdir(patchdir['name'][i])
                patchsplit = os.getcwd().split(os.path.sep)

                if not os.path.exists('no_ps_info.mat'):
                    stamps_step_no_ps = np.zeros((5, 1))
                    stamps_step_no_ps = {'stamps_step_no_ps': stamps_step_no_ps}
                    savemat('no_ps_info.mat', stamps_step_no_ps)

    if start_step <= 6 and end_step >= 6:
        print('\n##################\n' +
              '##### Step 6 #####\n' +
              '##################\n')

        print("Dirrectory is " + os.getcwd())

        ps_unwrap()
        if getparm('small_baseline_flag')[0][0] == 'y':
            not_supported_param('use_small_baselines', 'y')
            # sb_invert_uw

    if start_step <= 7 and end_step >= 7:
        print('\n##################\n' +
              '##### Step 7 #####\n' +
              '##################\n')

        print("Dirrectory is " + os.getcwd())

        if getparm('small_baseline_flag')[0][0] == 'y':
            not_supported_param('small_baseline_flag', 'y')
            # ps_calc_scla(1,1)   % small baselines
            # ps_smooth_scla(1)
            # ps_calc_scla(0,1) % single master
        else:
            ps_calc_scla(0, 1)
            ps_smooth_scla(0)


if __name__ == "__main__":
    args = sys.argv
    main(args)
    sys.exit(0)
