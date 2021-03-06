import sys
import numpy as np
import collections
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
    obj_matlab = loadmat('D:\\Temp\\stamps\\' + obj_name + '.mat')[obj_name]
    obj_py = obj

    obj_matlab[np.isnan(obj_matlab)] = 0
    obj_py[np.isnan(obj_py)] = 0

    diff = obj_matlab - obj_py
    max_error = np.max(diff)
    min_error = np.min(diff)
    diff_pos = np.array(np.where(diff != 0))
    diff = {'diff': diff, 'max_error': max_error, 'diff_pos': diff_pos, 'min_error': min_error}

    return diff


def compare_complex_objects(obj, obj_name):
    obj_matlab = loadmat('D:\\Temp\\stamps\\' + obj_name + '.mat')[obj_name]
    obj_py = obj

    obj_matlab[np.isnan(obj_matlab)] = 0j
    obj_py[np.isnan(obj_py)] = 0j

    obj_matlab_phase = np.angle(obj_matlab)
    obj_matlab_magnitude = np.abs(obj_matlab)

    obj_py_phase = np.angle(obj_py)
    obj_py_magnitude = np.abs(obj_py)

    diff_phase = obj_matlab_phase - obj_py_phase
    max_error_phase = np.max(diff_phase)
    min_error_phase = np.min(diff_phase)
    diff_pos_phase = np.array(np.where(diff_phase != 0))

    diff_magnitude = obj_matlab_magnitude - obj_py_magnitude
    max_error_magnitude = np.max(diff_magnitude)
    min_error_magnitude = np.min(diff_magnitude)
    diff_pos_magnitude = np.array(np.where(diff_magnitude != 0))

    diff = {'diff_magnitude': diff_magnitude,
            'max_error_magnitude': max_error_magnitude,
            'diff_pos_magnitude': diff_pos_magnitude,
            'min_error_magnitude': min_error_magnitude,
            'diff_phase': diff_phase,
            'max_error_phase': max_error_phase,
            'diff_pos_phase': diff_pos_phase,
            'min_error_phase': min_error_phase
            }

    return diff


def compare_mat_with_number_values(file_name, matlab_path="D:\\Temp\\stamps\\", **kwargs):
    mat_matlab = loadmat(matlab_path + file_name)
    mat_py = loadmat(file_name)

    if "excluded_keys" in kwargs:
        excluded_keys = kwargs["excluded_keys"]

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


def compare_mat_misc_values(file_name, *excluded_keys):
    mat_matlab = loadmat('D:\\Temp\\stamps\\' + file_name)
    mat_py = loadmat(file_name)

    if len(mat_matlab.keys()) != len(mat_py.keys()):
        print("Different key amount")
        return

    for key in mat_matlab.keys():
        if '__' not in key and key != "Created":
            mat_matlab_value = mat_matlab[key]
            mat_py_value = mat_py[key]

            if (len(mat_matlab_value) == 0 and len(mat_py_value) == 0):
                continue

            if np.size(mat_matlab_value) > 1:
                for i in range(len(mat_matlab_value[0])):
                    if mat_matlab_value[0][i] != mat_py_value[0][i]:
                        print("Diff key: " + key + " = " + mat_matlab_value[0])
                        continue

            if np.size(mat_matlab_value) == 1:

                if isinstance(mat_matlab_value[0], str):
                    if mat_matlab_value[0] != mat_py_value[0]:
                        print("Diff key: " + key + " = " + mat_matlab_value[0])
                        continue
                else:
                    if np.isnan(mat_matlab_value[0][0]) and np.isnan(mat_py_value[0][0]):
                        continue

                    if mat_matlab_value[0][0] != mat_py_value[0][0]:
                        print("Diff key: " + key + " = " + mat_matlab_value[0][0])
                        continue


def main(args):
    # diff = compare_mat_with_list_values('parms.mat')
    # print(diff)

    compare_mat_misc_values('parms.mat')


if __name__ == "__main__":
    main(sys.argv)
    sys.exit(0)
