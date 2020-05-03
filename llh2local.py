import numpy as np
import mpmath
from utils import compare_objects


def llh2local(llh, origin):
    a = 6378137.0
    e = 0.08209443794970

    llh = llh * np.pi / 180
    origin = origin * np.pi / 180

    z = llh[1, :] != 0
    dlambda = llh[0, z] - origin[0]

    M = a * ((1 - e ** 2 / 4 - 3 * e ** 4 / 64 - 5 * e ** 6 / 256) * llh[1, z] -
             (3 * e ** 2 / 8 + 3 * e ** 4 / 32 + 45 * e ** 6 / 1024) * np.sin(2 * llh[1, z]) +
             (15 * e ** 4 / 256 + 45 * e ** 6 / 1024) * np.sin(4 * llh[1, z]) -
             (35 * e ** 6 / 3072) * np.sin(6 * llh[1, z]))

    M0 = a * ((1 - e ** 2 / 4 - 3 * e ** 4 / 64 - 5 * e ** 6 / 256) * origin[1] -
              (3 * e ** 2 / 8 + 3 * e ** 4 / 32 + 45 * e ** 6 / 1024) * np.sin(2 * origin[1]) +
              (15 * e ** 4 / 256 + 45 * e ** 6 / 1024) * np.sin(4 * origin[1]) -
              (35 * e ** 6 / 3072) * np.sin(6 * origin[1]))

    N = np.divide(a, np.sqrt(1 - e ** 2 * np.power(np.sin(llh[1, z]), 2)))
    E = np.multiply(dlambda, np.sin(llh[1, z]))

    xy = np.zeros((2, len(z)))
    cot_llh = np.array([mpmath.cot(x) for x in llh[1, z]]).astype('float64')
    xy[0, z] = np.multiply(np.multiply(N, cot_llh), np.sin(E))
    xy[1, z] = M - M0 + np.multiply(np.multiply(N, cot_llh), (1 - np.cos(E)))

    dlambda = llh[0, ~z] - origin[0]
    xy[0, ~z] = a * dlambda
    xy[1, ~z] = -M0

    xy = xy / 1000

    return xy
