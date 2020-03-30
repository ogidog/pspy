import os, sys
from datetime import datetime
from scipy.io import loadmat


def ps_parms_default():
    parmfile = 'parms.mat';
    parent_flag = 0;

    if os.path.exists('./parms.mat'):
        parms = loadmat(parmfile);
    else:
        if os.path.exists('../parms.mat'):
            parmfile = '../parms.mat';
            parms = loadmat(parmfile);
            parent_flag = 1;
        else:
            parms = {}
            parms['Created'] = datetime.today().strftime('%Y-%m-%d')
            parms['small_baseline_flag'] = 'n'

    parmfields_before = parms.keys();
    num_fields = len(parmfields_before);

    if 'max_topo_err' not in parmfields_before:
        parms['max_topo_err']=20


    sys.exit(0)
