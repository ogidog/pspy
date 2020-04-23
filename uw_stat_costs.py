import numpy as np

from scipy.io import loadmat
from utils import *


def uw_stat_costs(*args):
    # params: unwrap_method,variance,subset_ifg_index

    if len(args) < 1:
        unwrap_method = '3D'

    costscale = 100
    nshortcycle = 200
    maxshort = 32000

    print('Unwrapping in space...\n')

    uw = loadmat('uw_grid.mat')
    ui = loadmat('uw_interp.mat')
    ut = loadmat('uw_space_time.mat')

    if len(args) < 2:
        variance = []

    if len(args) < 3:
        subset_ifg_index = np.array([i for i in range(len(uw['ph'][0]))])

    predef_flag = 'n'
    if 'predef_ix' in ut.keys() and len(ut['predef_ix']) > 0:
        predef_flag = 'y'

    nrow = len(uw['nzix'])
    ncol = len(uw['nzix'][0])

    x, y = np.nonzero(uw['nzix'].T)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    nzix = np.nonzero(uw['nzix'].T.flatten())[0].reshape(-1, 1)
    z = np.array([i for i in range(uw['n_ps'][0][0])])

    colix = ui['colix']
    rowix = ui['rowix']
    Z = ui['Z']

    grid_edges = np.concatenate((colix.T[(np.abs(colix) > 0).T], rowix.T[(np.abs(rowix) > 0).T]), axis=0).reshape(-1, 1)
    n_edges = np.histogram(np.abs(grid_edges), np.array(range(ui['n_edge'][0][0] + 1)) + 1)[0].reshape(-1, 1)

    unwrap_method = args[0]
    if unwrap_method == '2D':
        not_supported_param('unwrap_method', '2D')
        # edge_length=sqrt(diff(x(ui.edgs(:,2:3)),[],2).^2+diff(y(ui.edgs(:,2:3)),[],2).^2);
        # %sigsq_noise=ones(ui.n_edge,1);
        # if uw.pix_size==0
        #    pix_size=5;  % if we don't know resolution
        # else
        #    pix_size=uw.pix_size;
        # end
        # if isempty(variance)
        #    sigsq_noise=zeros(size(edge_length));
        # else
        #    sigsq_noise=variance(ui.edgs(:,2))+variance(ui.edgs(:,3));
        # end
        # sigsq_aps=(2*pi)^2; % fixed for now as one fringe
        # aps_range=20000; % fixed for now as 20 km
        # sigsq_noise=sigsq_noise+sigsq_aps*(1-exp(-edge_length*pix_size*3/aps_range)); % cov of dph=C11+C22-2*C12 (assume APS only contributor)
        # sigsq_noise=sigsq_noise/10; % scale it to keep in reasonable range
        # dph_smooth=ut.dph_space_uw;
    else:
        sigsq_noise = np.power((np.std(ut['dph_noise'], ddof=1, axis=1) / 2 / np.pi), 2).reshape(-1, 1)
        # sigsq_defo = (std(ut.dph_space_uw - ut.dph_noise, 0, 2) / 2 / pi). ^ 2;
        dph_smooth = ut['dph_space_uw'] - ut['dph_noise']
    del ut['dph_noise']
    nostats_ix = np.where(np.isnan(sigsq_noise))[0]
    rowix = rowix.astype(float)
    colix = colix.astype(float)
    for i in nostats_ix:
        rowix[abs(rowix) == i + 1] = float('nan')
        colix[abs(colix) == i + 1] = float('nan')
    rowix = rowix.astype('int')
    colix = colix.astype('int')

    sigsq = np.round(np.multiply(((sigsq_noise) * np.power(nshortcycle, 2)) / costscale, n_edges))
    sigsq[np.isnan(sigsq)] = 0

    sigsq[sigsq < 1] = 1

    diff = compare_objects(sigsq, 'sigsq')
    print()
