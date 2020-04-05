import sys, os
import argparse
from ps_calc_scla import ps_calc_scla
from ps_smooth_scla import ps_smooth_scla
from getparm import get_parm_value as getparm


def main(args):
    if args.start_step <= 7 and args.end_step >= 7:

        print('\n##################\n' +
              '##### Step 7 #####\n' +
              '##################\n')

        print("Dirrectory is " + os.getcwd())

        if getparm('small_baseline_flag')[0][0] == 'y':
            print("You set the param use_small_baselines={}, but not supported yet.".format(
                getparm('use_small_baselines')[0][0]))
            sys.exit()
            # ps_calc_scla(1,1)   % small baselines
            # ps_smooth_scla(1)
            # ps_calc_scla(0,1) % single master
        else:
            # ps_calc_scla(0, 1)
            ps_smooth_scla(0)
    else:
        print('Error')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PS Main Script.')
    parser.add_argument('start_step', metavar='start_step', type=int)
    parser.add_argument('end_step', metavar='end_step', type=int)
    args = parser.parse_args()

    main(args)
    sys.exit(0)
