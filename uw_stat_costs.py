import numpy as np
import os

from scipy.io import loadmat
from utils import *
from writecpx import writecpx


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

    sigsq = np.round(np.multiply(((sigsq_noise) * np.power(nshortcycle, 2)) / costscale, n_edges))
    sigsq[np.isnan(sigsq)] = 0

    sigsq[sigsq < 1] = 1

    rowcost = np.zeros(((nrow - 1), ncol * 4))
    colcost = np.zeros(((nrow), (ncol - 1) * 4))

    nzrowix = np.abs(rowix) > 0
    rowstdgrid = np.ones(np.shape(rowix))

    nzcolix = np.abs(colix) > 0
    colstdgrid = np.ones(np.shape(colix))

    rowcost[:, 2::4] = maxshort
    colcost[:, 2::4] = maxshort

    stats_ix = ~np.isnan(rowix)
    rowcost[:, 3::4] = stats_ix.astype('int') * (-1 - maxshort) + 1
    stats_ix = ~np.isnan(colix)
    colcost[:, 3::4] = stats_ix.astype('int') * (-1 - maxshort) + 1

    ph_uw = np.zeros((uw['n_ps'][0][0], uw['n_ifg'][0][0]))
    ifguw = np.zeros((nrow, ncol))
    msd = np.zeros((uw['n_ifg'][0][0], 1))

    f = open('snaphu.conf', 'w')
    f.write('INFILE  snaphu.in\n')
    f.write('OUTFILE snaphu.out\n')
    f.write('COSTINFILE snaphu.costinfile\n')
    f.write('STATCOSTMODE  DEFO\n')
    f.write('INFILEFORMAT  COMPLEX_DATA\n')
    f.write('OUTFILEFORMAT FLOAT_DATA\n')
    f.close()

    for i1 in subset_ifg_index:
        print('   Processing IFG {} of {}'.format(i1 + 1, len(subset_ifg_index)))
        spread = ut['spread'][:, i1].reshape(-1, 1)
        spread = np.round(np.multiply(np.abs(spread) * np.power(nshortcycle, 2) / 6 / costscale,
                                      np.tile(n_edges, (1, len(spread[0])))))
        sigsqtot = sigsq + spread

        if predef_flag == 'y':
            not_supported_param('predef_flag', 'y')
            # sigsqtot(ut.predef_ix(:,i1))=1;
        rowstdgrid.T[nzrowix.T] = sigsqtot[np.abs(rowix.T[nzrowix.T]).astype('int') - 1].flatten()
        rowcost[:, 1::4] = rowstdgrid
        colstdgrid.T[nzcolix.T] = sigsqtot[np.abs(colix.T[nzcolix.T]).astype('int') - 1].flatten()
        colcost[:, 1::4] = colstdgrid

        offset_cycle = (np.angle(np.exp(complex(0, 1) * ut['dph_space_uw'][:, i1])) - dph_smooth[:, i1]) / 2 / np.pi
        offset_cycle = offset_cycle.reshape(-1, 1)
        offgrid = np.zeros(np.shape(rowix))
        offgrid.T[nzrowix.T] = np.round(np.multiply(offset_cycle[(np.abs(rowix.T[nzrowix.T]) - 1).astype('int')],
                                                    np.sign(rowix.T[nzrowix.T] - 1).reshape(-1,
                                                                                            1) * nshortcycle)).flatten()
        rowcost[:, 0::4] = -offgrid
        offgrid = np.zeros(np.shape(colix))
        offgrid.T[nzcolix.T] = np.round(np.multiply(offset_cycle[(np.abs(colix.T[nzcolix.T]) - 1).astype('int')],
                                                    np.sign(colix.T[nzcolix.T] - 1).reshape(-1,
                                                                                            1) * nshortcycle)).flatten()
        colcost[:, 0::4] = offgrid

        if os.path.exists('snaphu.costinfile'):
            os.remove('snaphu.costinfile')

        fid = open('snaphu.costinfile', 'wb')
        rowcost_int16 = rowcost.astype('int16')
        rowcost_int16 = rowcost_int16
        rowcost_int16.tofile(fid)
        colcost_int16 = colcost.astype('int16')
        colcost_int16 = colcost_int16
        colcost_int16.tofile(fid)
        fid.close()

        ifgw = uw['ph'][Z - 1, i1].reshape(nrow, ncol)
        writecpx('snaphu.in', ifgw)

        if os.path.exists('snaphu.log'):
            os.remove('snaphu.log')

        cmdstr = 'snaphu -d -f snaphu.conf ' + str(ncol) + ' > snaphu.log'
        os.system(cmdstr)

        fid = open('snaphu.out', 'r')
        ifguw = np.fromfile(fid, dtype='float32')
        ifguw = ifguw.reshape(np.shape(ifgw))
        fid.close()
        ifg_diff1 = ifguw[0:len(ifguw) - 1, :] - ifguw[1:, :]
        ifg_diff1 = ifg_diff1.T[(ifg_diff1 != 0).T].reshape(-1, 1)
        ifg_diff2 = ifguw[:, 0:len(ifguw[0]) - 1] - ifguw[:, 1:]
        ifg_diff2 = ifg_diff2.T[(ifg_diff2 != 0).T].reshape(-1, 1)
        msd[i1] = (np.sum(np.power(ifg_diff1, 2)) + np.sum(np.power(ifg_diff2, 2))) / (len(ifg_diff1) + len(ifg_diff2))
        ph_uw[:, i1] = ifguw.T[uw['nzix'].astype('bool').T]

    uw_phaseuw = {
        'ph_uw': ph_uw,
        'msd': msd
    }
    savemat('uw_phaseuw.mat', uw_phaseuw)