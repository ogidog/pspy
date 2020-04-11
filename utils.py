import sys
import numpy as np
from scipy.io import loadmat, savemat


def not_supported_param(param, value):
    raise Exception("You set the param {}={}, but not supported yet.".format(param, value))
    sys.exit(0)


def compare_objects(obj, obj_name):
    obj_by_matlab = loadmat('F:\\Temp\\' + obj_name + '.mat')[obj_name]
    obj_by_py = obj

    if len(obj_by_matlab) != 0:
        diff = obj_by_matlab - obj_by_py

        if ('complex' in str(diff.dtype)):
            print(np.max(np.abs(diff)))
        else:
            print(np.max(diff))

    return True
