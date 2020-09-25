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


def get_parm_value(parmname):

    ps_parms_default()

    parms, localparms = load_parms_file()
    parmnames = list(filter(lambda fkey: '__' not in fkey, list(parms.keys())))

    parmnum = [i for i in range(len(parmnames)) if parmname in parmnames[i]]
    if len(parmnum) > 1:
        print('Parameter {}  is not unique'.format(parmname))
    else:
        if len(parmnum) == 0:
            parmname = []
            value = []
        else:
            if parmname in localparms.keys():
                value = localparms[parmname]
            else:
                value = parms[parmname]

    return value, parmname


def load_parms_file():
    parms = {}
    parmfile = 'parms.mat'
    localparmfile = 'localparms.mat';

    if os.path.exists('./parms.mat'):
        parms = loadmat(parmfile);

    else:
        if os.path.exists('../parms.mat'):
            parmfile = '../parms.mat';
            parms = loadmat(parmfile)
        else:
            print('parms.mat not found')

    if os.path.exists('localparms.mat'):
        localparms = loadmat(localparmfile);
    else:
        localparms = {'Created': datetime.today().strftime('%Y-%m-%d')}

    return parms, localparms


def main(args):
    ps_parms_default()

    args = args[1:]

    if len(args) < 2:
        printflag = 0

    parms, localparms = load_parms_file()

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
