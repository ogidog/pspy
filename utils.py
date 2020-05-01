import sys
import numpy as np
from scipy.io import loadmat, savemat

def not_supported():
    raise Exception("not supported")
    sys.exit(0)

def not_supported_param(param, value):
    raise Exception("You set the param {}={}, but not supported yet.".format(param, value))
    sys.exit(0)


def not_supported_value(var, value):
    raise Exception("{}={} not supported.".format(var, value))
    sys.exit(0)


def compare_objects(obj, obj_name):
    obj_matlab = loadmat('F:\\Temp\\' + obj_name + '.mat')[obj_name]
    obj_py = obj

    obj_matlab[np.isnan(obj_matlab)] = 9999999
    obj_py[np.isnan(obj_py)] = 9999999

    diff = obj_matlab - obj_py
    max_error = np.max(diff)
    min_error = np.min(diff)
    diff_pos = np.array(np.where(diff != 0))
    diff = {'diff': diff, 'max_error': max_error, 'diff_pos': diff_pos, 'min_error': min_error}

    return diff


def compare_complex_objects(obj, obj_name):
    obj_matlab = loadmat('F:\\Temp\\' + obj_name + '.mat')[obj_name]
    obj_py = obj

    diff = obj_matlab - obj_py
    max_error = np.max(diff)
    min_error = np.min(diff)
    diff_pos = np.array(np.where(diff != complex(0, 0)))

    diff = {'diff': diff, 'max_error': max_error, 'diff_pos': diff_pos, 'min_error': min_error}

    return diff


def compare_mat_file(file_name, *excluded_keys):
    mat_matlab = loadmat('F:\\Temp\\' + file_name)
    mat_py = loadmat(file_name)

    keys_matlab = mat_matlab.keys()

    diffs = {}

    for key in keys_matlab:
        if '__' not in key:
            if key not in excluded_keys:
                obj_py = mat_py[key]
                obj_matlab = mat_matlab[key]
                obj_matlab[np.isnan(obj_matlab)] = 9999999
                obj_py[np.isnan(obj_py)] = 9999999

                diff = obj_py - obj_matlab
                if len(diff) > 0:
                    max_error = np.max(diff)
                    min_error = np.min(diff)
                    diff = {'diff': diff, 'max_error': max_error, 'min_error': min_error}
                    diffs[key] = diff
                else:
                    diff = {'diff': diff, 'max_error': 0, 'min_error': 0}
                    diffs[key] = diff
    return diffs


def main(args):
    diff = compare_mat_file('scla_smooth2.mat')

    print(args)


if __name__ == "__main__":
    main(sys.argv)
    sys.exit(0)
