import sys, os
from ps_unwrap import ps_unwrap
from ps_calc_scla import ps_calc_scla
from ps_smooth_scla import ps_smooth_scla
from getparm import get_parm_value as getparm
from utils import not_supported_param


def main(args):
    start_step = args[1]
    end_step = args[2]

    if start_step <= 6 and end_step >= 6:
        print('\n##################\n' +
              '##### Step 6 #####\n' +
              '##################\n')

        print("Dirrectory is " + os.getcwd())

        ps_unwrap()
        if getparm('small_baseline_flag')[0][0] == 'y':
            not_supported_param('use_small_baselines', 'y')

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
