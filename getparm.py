import sys, os
import numpy as np
from datetime import datetime
from ps_parms_default import ps_parms_default
from scipy.io import loadmat


def parms_value_format(value):
    if ~isinstance(value, np.ndarray):
        if isinstance(value, str):
            return str.format("'{}'", value)
        else:
            return value


def print_params(parms):
    for key, value in sorted(parms.items()):
        if '__' in key:
            continue
        print_parm(parms, key)


def print_parm(parms, key):
    if key not in parms.keys():
        print('Parameter not found')
        return

    value = parms[key]
    if isinstance(value, np.ndarray):
        if value.ndim == 2 and len(value) != 0:
            if len(value[0]) == 1:
                print(key + ': {}'.format(parms_value_format(value[0][0])))
            else:
                print(key + ': {}'.format(parms_value_format(value[0])))

        if value.ndim == 2 and len(value) == 0:
            print(key + ': {}'.format(parms_value_format(value)))

        if value.ndim == 1:
            print(key + ': {}'.format(parms_value_format(value[0])))
    else:
        print(key + ': {}'.format(parms_value_format(value)))


def get_parm_value(key):
    ps_parms_default
    parms = load_parms_file()

    return parms[key]


def load_parms_file():
    parms = {}
    parmfile = 'parms.mat'
    # localparmfile='localparms';

    if os.path.exists('./parms.mat'):
        parms = loadmat(parmfile);

    else:
        if os.path.exists('../parms.mat'):
            parmfile = '../parms';
            parms = loadmat(parmfile)
        else:
            print('parms.mat not found')

    # if exist('localparms.mat','file')
    #    localparms=load(localparmfile);
    # else
    #     localparms=struct('Created',date);
    # end

    return parms


def main(args):
    ps_parms_default

    args = args[1:]

    # if nargin<2
    #    printflag=0;
    # end

    parms = load_parms_file()

    if len(args) < 1:
        print_params(parms)
        # if size(fieldnames(localparms),1)>1
        #    localparms
        # end
    else:
        if len(args) == 1:
            print_parm(parms, args[0])


if __name__ == "__main__":
    main(sys.argv)
    sys.exit(0)
