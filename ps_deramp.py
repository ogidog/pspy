import os
import numpy as np
import ggf
from scipy.io import loadmat


def ps_deramp(ps, ph_all, degree):
    print('     Deramping computed on the fly. \n')

    if os.path.exists('deramp_degree.mat') == 2:
        try:
            degree = loadmat('deramp_degree.mat')['degree'][0][0]
            print('     Found deramp_degree.mat file will use that value to deramp\n')
        except:
            degree = 1

    if ps['n_ifg'][0][0] != len(ph_all[0]):
        ps['n_ifg'][0][0] = len(ph_all[0])

    if degree == 1:
        # z = ax + by+ c
        A = np.hstack((ps['xy'][:, 1:] / 1000, np.ones((ps['n_ps'][0][0], 1))))
        print('**** z = ax + by+ c\n')
    if degree == 1.5:
        # z = ax + by+ cxy + d
        # A = double([ps.xy(:,2:3)/1000 ps.xy(:,2)/1000.*ps.xy(:,3)/1000 ones([ps.n_ps 1])]);
        # print('**** z = ax + by+ cxy + d\n')
        print('Degrees other then 1 not supported yet.')
    if degree == 2:
        # z = ax^2 + by^2 + cxy + d
        # A = double([(ps.xy(:,2:3)/1000).^2 ps.xy(:,2)/1000.*ps.xy(:,3)/1000 ones([ps.n_ps 1])]);
        # print('**** z = ax^2 + by^2 + cxy + d \n')
        print('Degrees other then 1 not supported yet.')
    if degree == 3:
        # z = ax^3 + by^3 + cx^2y + dy^2x + ex^2 + by^2 + cxy + d
        # A = double([(ps.xy(:,2:3)/1000).^3  (ps.xy(:,2)./1000).^2.*ps.xy(:,3)/1000 (ps.xy(:,3)./1000).^2.*ps.xy(:,2)/1000 (ps.xy(:,2:3)/1000).^2 ps.xy(:,2)/1000.*ps.xy(:,3)/1000 ones([ps.n_ps 1])]);
        # print('**** z = ax^3 + by^3 + cx^2y + dy^2x + ex^2 + fy^2 + gxy + h \n')
        print('Degrees other then 1 not supported yet.')

    ph_ramp = np.empty((len(ph_all), len(ph_all[0]),))
    ph_ramp[:] = np.nan

    # TODO: may be an error
    for k in range(0, ps['n_ifg'][0][0]):
        ix = np.isnan(ph_all[:, k]);
        if ps['n_ps'][0][0] - np.sum(ix) > 5:
            coeff = ggf.matlab_funcs.lscov(A[~ix, :], ph_all[~ix, k])
            ph_ramp[:, k] = A.dot(coeff)
            ph_all[:, k] = ph_all[:, k] - ph_ramp[:, k]
        else:
            print(['Ifg ' + str(k) + ' is not deramped \n'])

    return [ph_all, ph_ramp]