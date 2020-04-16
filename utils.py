import sys
import numpy as np
from scipy.io import loadmat, savemat


def not_supported_param(param, value):
    raise Exception("You set the param {}={}, but not supported yet.".format(param, value))
    sys.exit(0)


def compare_objects(obj, obj_name):
    obj_matlab = loadmat('F:\\Temp\\' + obj_name + '.mat')[obj_name]
    obj_py = obj

    if len(obj_matlab) != 0:
        diff = obj_matlab - obj_py

        if ('complex' in str(diff.dtype)):
            max_error = np.max(np.abs(diff))
        else:
            max_error = np.max(diff)

        diff_pos = np.array(np.where(diff != 0))

    diff = {'diff': diff, 'max_error': max_error, 'diff_pos': diff_pos}
    return diff


def compare_mat_file(file_name):
    mat_matlab = loadmat('F:\\Temp\\' + file_name)
    mat_py = loadmat(file_name)

    keys_matlab = mat_matlab.keys()

    diffs = {}
    for key in keys_matlab:
        if '__' not in key:
            diff = mat_py[key] - mat_matlab[key]

            if ('complex' in str(diff.dtype)):
                max_error = np.max(np.abs(diff))
            else:
                max_error = np.max(diff)

            diff_pos = np.array(np.where(diff != 0))

            diffs[key] = {'diff': diff, 'max_error': max_error, 'diff_pos': diff_pos}

    return diffs
