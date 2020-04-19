import numpy as np

from scipy.io import loadmat, savemat
from utils import *


def uw_sb_unwrap_space_time(day, ifgday_ix, unwrap_method, time_win, la_flag, bperp, n_trial_wraps, prefilt_win,
                            scf_flag, temp, n_temp_wraps, max_bperp_for_temp_est):
    print('\nUnwrapping in time-space...')

    uw = loadmat('uw_grid.mat');
    ui = loadmat('uw_interp.mat');

    n_ifg = uw['n_ifg'][0][0]
    n_ps = uw['n_ps'][0][0]
    nzix = uw['nzix']
    ij = uw['ij']

    if 'ph_uw_predef' in uw.keys():
        predef_flag = 'n'
    else:
        predef_flag = 'y'

    n_image = len(day)
    master_ix = np.where(day == 0)[0][0]
    nrow, ncol = np.shape(ui['Z'])

    day_pos_ix = np.where(day > 0)
    tempdummy = np.min(day[day_pos_ix])
    I = np.argmin(day[day_pos_ix])
    dph_space = np.multiply(uw['ph'][ui['edgs'][:, 2] - 1, :], np.conj(uw['ph'][ui['edgs'][:, 1] - 1, :]))
    if predef_flag == 'y':
        not_supported_param(predef_flag, 'y')
        # dph_space_uw=uw.ph_uw_predef(ui.edgs(:,3),:)-uw.ph_uw_predef(ui.edgs(:,2),:);
        # predef_ix=~isnan(dph_space_uw);
        # dph_space_uw=dph_space_uw(predef_ix);
    else:
        predef_ix = []

    uw.clear()
    tempdummy = -1

    dph_space = dph_space / np.abs(dph_space)

    ifreq_ij = [];
    jfreq_ij = [];

    G = np.zeros((n_ifg, n_image))

    ifgday_ix[:, 0] = ifgday_ix[:, 0] - 1
    for i in range(0, n_ifg):
        G[i, ifgday_ix[i, 0]] = -1
        G[i, ifgday_ix[i, 1]] = 1

    nzc_ix = np.sum(np.abs(G), axis=0) != 0
    day = day[nzc_ix]
    n_image = len(day)
    G = G[:, nzc_ix]
    zc_ix = np.where(nzc_ix == 0)
    zc_ix = -np.sort(np.multiply(zc_ix, -1))[0]  # sort an array descent

    for i in range(len(zc_ix)):
        ifgday_ix[ifgday_ix > zc_ix[i]] = ifgday_ix[ifgday_ix > zc_ix[i]] - 1
    n = len(G[0])

    if len(temp) > 0:
        temp_flag = 'y'
    else:
        temp_flag = 'n'

    if temp_flag == 'y':
        not_supported_param('temp_flag', 'y')

        # fprintf('   Estimating temperature correlation (elapsed time=%ds)\n',round(toc))
        # ix=abs(bperp)<max_bperp_for_temp_est;
        # temp_sub=temp(ix);
        # temp_range=max(temp)-min(temp);
        # temp_range_sub=max(temp_sub)-min(temp_sub);
        # dph_sub=dph_space(:,ix); % only ifgs using ith image
        # n_temp_wraps=n_temp_wraps*(temp_range_sub/temp_range);

        # trial_mult=[-ceil(8*n_temp_wraps):ceil(8*n_temp_wraps)];
        # n_trials=length(trial_mult);
        # trial_phase=temp_sub/temp_range_sub*pi/4;
        # trial_phase_mat=exp(-j*trial_phase*trial_mult);
        # Kt=zeros(ui.n_edge,1,'single');
        # coh=zeros(ui.n_edge,1,'single');
        # for i=1:ui.n_edge
        #    cpxphase=dph_sub(i,:).';
        #    cpxphase_mat=repmat(cpxphase,1,n_trials);
        #    phaser=trial_phase_mat.*cpxphase_mat;
        #    phaser_sum=sum(phaser);
        #    coh_trial=abs(phaser_sum)/sum(abs(cpxphase));
        #    [coh_max,coh_max_ix]=max(coh_trial);
        #    falling_ix=find(diff(coh_trial(1:coh_max_ix))<0); % segemnts prior to peak where falling
        #    if ~isempty(falling_ix)
        #        peak_start_ix=falling_ix(end)+1;
        #    else
        #        peak_start_ix=1;
        #    end
        #   rising_ix=find(diff(coh_trial(coh_max_ix:end))>0); % segemnts after peak where rising
        #   if ~isempty(rising_ix)
        #        peak_end_ix=rising_ix(1)+coh_max_ix-1;
        #   else
        #        peak_end_ix=n_trials;
        #    end
        #    coh_trial(peak_start_ix:peak_end_ix)=0;
        #    if coh_max-max(coh_trial)>0.1 % diff between peak and next peak at least 0.1
        #        K0=pi/4/temp_range_sub*trial_mult(coh_max_ix);
        #        resphase=cpxphase.*exp(-1i*(K0*temp_sub)); % subtract approximate fit
        #        offset_phase=sum(resphase);
        #        resphase=angle(resphase*conj(offset_phase)); % subtract offset, take angle (unweighted)
        #        weighting=abs(cpxphase);
        #        mopt=double(weighting.*temp_sub)\double(weighting.*resphase);
        #        Kt(i)=K0+mopt;
        #        phase_residual=cpxphase.*exp(-1i*(Kt(i)*temp_sub));
        #        mean_phase_residual=sum(phase_residual);
        #        coh(i)=abs(mean_phase_residual)/sum(abs(phase_residual));
        #    end
        # end

        # clear cpxphase_mat trial_phase_mat phaser
        # Kt(coh<0.31)=0; % not to be trusted;
        # dph_space=dph_space.*exp(-1i*Kt*temp');
        # if predef_flag=='y'
        #    dph_temp=Kt*temp';
        #    dph_space_uw=dph_space_uw-dph_temp(predef_ix);
        #    clear dph_temp
        # end
        # dph_sub=dph_sub.*exp(-1i*Kt*temp_sub');
    # end

    if la_flag == 'y':
        print('   Estimating look angle error )\n')

        bperp_range = (max(bperp) - min(bperp))[0]
        ix = np.where(np.abs(np.diff(ifgday_ix, 1, 1)) == 1)[0]

        if len(ix) >= len(day) - 1:
            print('   using sequential daisy chain of interferograms\n')
            dph_sub = dph_space[:, ix]
            bperp_sub = bperp[ix]
            bperp_range_sub = (max(bperp_sub) - min(bperp_sub))[0]
            n_trial_wraps = n_trial_wraps * (bperp_range_sub / bperp_range)
        else:
            ifgs_per_image = np.sum(np.abs(G), axis=0)
            max_ifgs_per_image = np.amax(ifgs_per_image)
            max_ix = np.argmax(ifgs_per_image)

            if max_ifgs_per_image >= len(day) - 2:
                print('   Using sequential daisy chain of interferograms\n')
                ix = np.array([G[:, max_ix] != 0])[0]
                gsub = G[ix, max_ix]
                sign_ix = -np.sign(gsub)
                dph_sub = dph_space[:, ix]
                bperp_sub = [bperp[ix]][0]
                bperp_sub[sign_ix == -1] = -bperp_sub[sign_ix == -1]
                bperp_sub = np.concatenate((bperp_sub, np.array([[0]])), axis=0)
                sign_ix = np.tile(sign_ix, (ui['n_edge'][0][0], 1))
                dph_sub[sign_ix == -1] = np.conj(dph_sub[sign_ix == -1])
                dph_sub = np.concatenate((dph_sub, np.mean(np.abs(dph_sub), 1).reshape(-1, 1)), axis=1)
                slave_ix = np.sum(ifgday_ix[ix, :], axis=1) - max_ix
                day_sub = day[np.concatenate((slave_ix, np.array([max_ix])), axis=0)]
                sort_ix = np.argsort(day_sub, axis=None)
                day_sub = np.sort(day_sub, axis=None).reshape(-1, 1)
                dph_sub = dph_sub[:, sort_ix]
                bperp_sub = bperp_sub[sort_ix]
                bperp_sub = np.diff(bperp_sub, 1, axis=0)
                bperp_range_sub = (max(bperp_sub) - min(bperp_sub))[0]
                n_trial_wraps = n_trial_wraps * (bperp_range_sub / bperp_range)
                n_sub = len(day_sub)
                dph_sub = np.multiply(dph_sub[:, 1:], np.conj(dph_sub[:, 0:len(dph_sub[0]) - 1]))
                dph_sub = dph_sub / np.abs(dph_sub)

            else:
                dph_sub = dph_space
                bperp_sub = bperp
                bperp_range_sub = bperp_range

        trial_mult = np.array([i for i in range(int(-np.ceil(8 * n_trial_wraps)), int(np.ceil(8 * n_trial_wraps) + 1))])
        n_trials = len(trial_mult)
        trial_phase = bperp_sub / bperp_range_sub * np.pi / 4
        trial_phase_mat = np.exp(complex(0, -1) * trial_phase * trial_mult)
        K = np.zeros((ui['n_edge'][0][0], 1))
        coh = np.zeros((ui['n_edge'][0][0], 1))

        for i in range(0, ui['n_edge'][0][0]):
            cpxphase = dph_sub[i, :].reshape(-1, 1)
            cpxphase_mat = np.tile(cpxphase, (1, n_trials))
            phaser = np.multiply(trial_phase_mat, cpxphase_mat)
            phaser_sum = sum(phaser)

            # TODO: убрать
            # diff = compare_complex_objects(phaser, 'phaser')

            print()
