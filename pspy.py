import argparse
from ps_calc_scla import ps_calc_scla


def main(args):
    if args.start_step <= 7 and args.end_step >= 7:
        ps_calc_scla(0, 1)
    else:
        print('Error')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PS Main Script.')
    parser.add_argument('start_step', metavar='start_step', type=int)
    parser.add_argument('end_step', metavar='end_step', type=int)
    args = parser.parse_args()

    main(args)
