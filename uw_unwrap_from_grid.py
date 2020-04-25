import numpy as np

from scipy.io import loadmat

from utils import compare_objects, not_supported_param


def uw_unwrap_from_grid(xy, pix_size):
    print('Unwrapping from grid...\n')

    uw = loadmat('uw_grid.mat', variable_names=('nzix', 'n_ps', 'grid_ij', 'ph_in', 'ph_in_predef'))
    uu = loadmat('uw_phaseuw.mat')

    n_ps, n_ifg = np.shape(uw['ph_in'])
    gridix = np.zeros(np.shape(uw['nzix']))
    gridix.T[uw['nzix'].astype('bool').T] = np.array([i + 1 for i in range(uw['n_ps'][0][0])])

    ph_uw = np.zeros((n_ps, n_ifg))

    ph_in_isreal = np.any(np.isreal(uw['ph_in']))

    for i in range(n_ps):
        ix = int(gridix[uw['grid_ij'][i, 0] - 1, uw['grid_ij'][i, 1] - 1])
        if ix == 0:
            ph_uw[i, :] = float('nan')
        else:
            ph_uw_pix = uu['ph_uw'][ix - 1, :].flatten()
            if ph_in_isreal:
                ph_uw[i, :] = ph_uw_pix + np.angle(np.exp(complex(0, 1) * (uw['ph_in'][i, :] - ph_uw_pix)))
            else:
                ph_uw[i, :] = ph_uw_pix + np.angle(np.multiply(uw['ph_in'][i, :], np.exp(complex(0, -1) * ph_uw_pix)))

    if len(uw['ph_in_predef']) > 0:
        not_supported_param('ph_in_predef', '')
        # predef_ix=~isnan(uw.ph_in_predef);
        # meandiff=nanmean(ph_uw-uw.ph_in_predef);
        # meandiff=2*pi*round(meandiff/2/pi);
        # uw.ph_in_predef=uw.ph_in_predef+repmat(meandiff,n_ps,1);
        # ph_uw(predef_ix)=uw.ph_in_predef(predef_ix);

    msd = uu['msd']

    return ph_uw, msd
