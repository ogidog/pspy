import numpy as np


def ps_topofit(*args):
    # cpxphase,bperp,n_trial_wraps,plotflag,asym

    if len(args) < 5:
        asym = 0

    cpxphase = args[0]
    if np.shape(cpxphase, 1) > 1:
        cpxphase = cpxphase.T

    pass
