import ggf
import numpy as np

from scipy.io import loadmat, savemat
from scipy.sparse import csr_matrix

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
        print('   Estimating look angle error \n')

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
            coh_trial = abs(phaser_sum) / sum(abs(cpxphase))
            coh_max = np.max(coh_trial)
            coh_max_ix = np.argmax(coh_trial)
            falling_ix = np.where(np.diff(coh_trial[0:coh_max_ix]) < 0)[0]
            if len(falling_ix) > 0:
                peak_start_ix = falling_ix[len(falling_ix) - 1] + 1
            else:
                peak_start_ix = 0
            rising_ix = np.where(np.diff(coh_trial[coh_max_ix:]) > 0)[0]
            if len(rising_ix) > 0:
                peak_end_ix = rising_ix[0] + coh_max_ix - 1
            else:
                peak_end_ix = n_trials
            coh_trial[peak_start_ix:peak_end_ix] = 0

            if coh_max - np.max(coh_trial) > 0.1:
                K0 = np.pi / 4 / bperp_range_sub * trial_mult[coh_max_ix]
                resphase = np.multiply(cpxphase, np.exp(np.multiply(complex(0, -1), (np.multiply(K0, bperp_sub)))))
                offset_phase = np.sum(resphase)
                resphase = np.angle(np.multiply(resphase, np.conj(offset_phase)))
                weighting = np.abs(cpxphase)
                A = np.multiply(weighting, bperp_sub)
                b = np.multiply(weighting, resphase)
                mopt = np.linalg.lstsq(A, b, rcond=-1)[0]
                K[i] = K0 + mopt
                phase_residual = np.multiply(cpxphase, np.exp(complex(0, -1) * (K[i] * bperp_sub)))
                mean_phase_residual = np.sum(phase_residual)
                coh[i] = np.abs(mean_phase_residual) / np.sum(abs(phase_residual))

        cpxphase_mat = []
        trial_phase_mat = []
        phaser = []
        dph_sub = []

        K[coh < 0.31] = 0
        if temp_flag == 'y':
            not_supported_param('temp_flag', 'y')
            # dph_space(K==0,:)=dph_space(K==0,:).*exp(1i*Kt(K==0)*temp')
            # Kt(K==0)=0;
            # K(Kt==0)=0;

        dph_space = np.multiply(dph_space, np.exp(complex(0, -1) * K * bperp.flatten()))
        if predef_flag == 'y':
            not_supported_param(predef_flag, 'y')
            # dph_scla=K*bperp';
            # dph_space_uw=dph_space_uw-dph_scla(predef_ix);
            # clear dph_scla

    spread = csr_matrix((ui['n_edge'][0][0], n_ifg), dtype=np.int).toarray()

    if unwrap_method == '2D':
        not_supported_param('unwrap_method', '2D')
        # dph_space_uw=angle(dph_space);
        # if strcmpi(la_flag,'y')
        #    dph_space_uw=dph_space_uw+K*bperp';   % equal to dph_space + integer cycles
        # end
        # if strcmpi(temp_flag,'y')
        #    dph_space_uw=dph_space_uw+Kt*temp';   % equal to dph_space + integer cycles
        # end
        # dph_noise=[];
        # save('uw_space_time','dph_space_uw','spread','dph_noise');
    else:
        if unwrap_method == '3D_NO_DEF':
            not_supported_param('unwrap_method', '2D')
            # dph_noise=angle(dph_space);
            # dph_space_uw=angle(dph_space);
            # if strcmpi(la_flag,'y')
            #    dph_space_uw=dph_space_uw+K*bperp';   % equal to dph_space + integer cycles
            # end
            # if strcmpi(temp_flag,'y')
            #    dph_space_uw=dph_space_uw+Kt*temp';   % equal to dph_space + integer cycles
            # end
            # save('uw_space_time','dph_space_uw','dph_noise','spread');
        else:
            print('   Smoothing in time\n')

            if unwrap_method == '3D_FULL':
                dph_smooth_ifg = np.empty((len(dph_space), len(dph_space[0])))
                dph_smooth_ifg[:] = np.nan
                for i in range(0, n_image):
                    ix = G[:, i] != 0
                    if sum(ix) >= n_image - 2:
                        gsub = G[ix, i]
                        dph_sub = dph_space[:, ix]
                        sign_ix = np.tile(-np.sign(np.transpose(gsub)), (ui['n_edge'][0][0], 1))
                        dph_sub[sign_ix == -1] = np.conj(dph_sub[sign_ix == -1])
                        slave_ix = np.sum(ifgday_ix[ix, :], axis=1) - i
                        day_sub = day[slave_ix]
                        day_sub = np.sort(day_sub)
                        sort_ix = np.argsort(day_sub, axis=None)
                        dph_sub = dph_sub[:, sort_ix]
                        dph_sub_angle = np.angle(dph_sub)
                        n_sub = len(day_sub)
                        dph_smooth = np.zeros((ui['n_edge'][0][0], n_sub))
                        dph_smooth = dph_smooth.astype(np.complex)
                        for i1 in range(0, n_sub):
                            time_diff = (day_sub[i1] - day_sub).flatten()
                            weight_factor = np.exp(-np.power(time_diff, 2) / 2 / np.power(time_win, 2))
                            weight_factor = weight_factor / sum(weight_factor)
                            dph_mean = np.sum(np.multiply(dph_sub, np.tile(weight_factor, (ui['n_edge'][0][0], 1))),
                                              axis=1).reshape(-1, 1)
                            dph_mean_adj = np.mod(dph_sub_angle - np.tile(np.angle(dph_mean), (1, n_sub)) + np.pi,
                                                  2 * np.pi) - np.pi
                            GG = np.concatenate((np.ones((n_sub, 1)), time_diff.reshape(-1, 1)), axis=1)
                            if len(GG) > 1:
                                m = ggf.matlab_funcs.lscov(GG, np.transpose(dph_mean_adj), w=weight_factor).reshape(
                                    len(GG[0]), len(dph_mean_adj))
                            else:
                                m = np.zeros((len(GG), ui['n_edge'][0][0]))
                            dph_smooth[:, i1] = np.multiply(dph_mean,
                                                            np.exp(complex(0, 1) * (m[0, :].reshape(-1, 1)))).flatten()

                        dph_smooth_sub = np.cumsum(np.concatenate((np.angle(dph_smooth[:, 0]).reshape(-1, 1), np.angle(
                            np.multiply(dph_smooth[:, 1:], np.conj(dph_smooth[:, 0:len(dph_smooth[0]) - 1])))), axis=1),
                                                   1)
                        close_master_ix = np.where(slave_ix - i > 0)[0]
                        if len(close_master_ix) == 0:
                            close_master_ix = n_sub
                        else:
                            close_master_ix = close_master_ix[0]
                            if close_master_ix > 1:
                                close_master_ix = np.concatenate(
                                    (np.array([close_master_ix - 1]), np.array([close_master_ix])), axis=0)

                        dph_close_master = np.mean(dph_smooth_sub[:, close_master_ix], axis=1).reshape(-1, 1)
                        dph_smooth_sub = dph_smooth_sub - np.tile(
                            dph_close_master - np.angle(np.exp(complex(0, 1) * dph_close_master)), (1, n_sub))
                        dph_smooth_sub = np.multiply(dph_smooth_sub, sign_ix)
                        already_sub_ix = np.where(np.isnan(dph_smooth_ifg[0, ix]) != True)[0]
                        ix = np.where(ix)[0]
                        already_ix = ix[already_sub_ix]
                        std_noise1 = np.std(
                            np.angle(np.multiply(dph_space[:, already_ix],
                                                 np.exp(complex(0, -1) * dph_smooth_ifg[:, already_ix]))))
                        std_noise1 = [] if np.any(np.isnan(std_noise1)) else std_noise1
                        std_noise2 = np.std(np.angle(np.multiply(dph_space[:, already_ix], np.exp(
                            complex(0, -1) * dph_smooth_sub[:, already_sub_ix]))))
                        std_noise2 = [] if np.any(np.isnan(std_noise2)) else std_noise2
                        keep_ix = np.ones(n_sub, dtype=bool)
                        keep_ix[already_sub_ix[std_noise1 < std_noise2]] = False
                        dph_smooth_ifg[:, ix[keep_ix]] = dph_smooth_sub[:, keep_ix]

                dph_noise = np.angle(np.multiply(dph_space, np.exp(complex(0, -1) * dph_smooth_ifg)))
                std_dph_noise = np.array([np.std(dph_noise[i], ddof=1) for i in range(len(dph_noise))])
                dph_noise[std_dph_noise > 1.2, :] = float('nan')

            else:
                not_supported_param('unwrap_method', 'unwrap_method')

                # x=(day-day(1))*(n-1)/(day(end)-day(1)); % use dates for smoothing
                # if predef_flag=='y'
                #    n_dph=size(dph_space,1);
                #    dph_space_angle=double(angle(dph_space));
                #    dph_space_angle(predef_ix)=dph_space_uw;
                #    dph_space_series=zeros(n,n_dph);
                #    for i=1:n_dph
                #        W=predef_ix(i,:)+0.01; % give more weight to the predfined unwrapped
                #        dph_space_series(2:end,i)=lscov(double(G(:,2:end)),dph_space_angle(i,:)',W);
                #    end
                # else
                #    dph_space_series=[zeros(1,ui.n_edge);double(G(:,2:end))\double(angle(dph_space))'];
                # end

                # dph_smooth_series=zeros(size(G,2),ui.n_edge,'single');

                # for i1=1:n
                #    time_diff_sq=(day(i1)-day).^2;
                #    weight_factor=exp(-time_diff_sq/2/time_win^2);
                #    weight_factor=weight_factor/sum(weight_factor);
                #    dph_smooth_series(i1,:)=sum(dph_space_series.*repmat(weight_factor,1,ui.n_edge));
                # end

                # dph_smooth_ifg=(G*dph_smooth_series)';
                # dph_noise=angle(dph_space.*exp(-1i*dph_smooth_ifg));

                # if strcmpi(unwrap_method,'3D_SMALL_DEF')|...
                #    strcmpi(unwrap_method,'3D_QUICK')
                #    not_small_ix=find(std(dph_noise,0,2)>1.3)';
                #    fprintf('   %d edges with high std dev in time (elapsed time=%ds)\n',length(not_small_ix),round(toc))
                #    dph_noise(not_small_ix,:)=nan;
                # else % 3D
                #    uw=load('uw_grid');
                #    ph_noise=angle(uw.ph.*conj(uw.ph_lowpass));
                #    clear uw
                #    dph_noise_sf=((ph_noise(ui.edgs(:,3),:)-(ph_noise(ui.edgs(:,2),:))));
                #    m_minmax=repmat([-pi,pi],5,1).*repmat([0.5;0.25;1;0.25;1],1,2);
                #    anneal_opts=[1;15;0;0;0;0;0];
                #    covm=cov((dph_noise_sf)); % estimate of covariance
                #    [W,P]=chol(inv(covm)); % weighting matrix
                #    if P~=0
                #        W=diag(1./sqrt(diag(covm)));
                #    end
                #    not_small_ix=find(std(dph_noise,0,2)>1)';
                #    fprintf('   Performing complex smoothing on %d edges (elapsed time=%ds)\n',length(not_small_ix),round(toc))

                #    n_proc=0;
                #    for i=not_small_ix
                #        dph=angle(dph_space(i,:))';
                #        dph_smooth_series(:,i)=uw_sb_smooth_unwrap(m_minmax,anneal_opts,G,W,dph,x);

                #        n_proc=n_proc+1;
                #        if round(n_proc/1000)==n_proc/1000
                #            save('uw_unwrap_time','G','dph_space','dph_smooth_series');
                #            fprintf('%d edges of %d reprocessed (elapsed time=%ds)\n',n_proc,length(not_small_ix),round(toc))
                #        end
                #    end
                #    dph_smooth_ifg=(G*dph_smooth_series)';
                #    dph_noise=angle(dph_space.*exp(-1i*dph_smooth_ifg));
                # end

            dph_space = []
            dph_space_uw = dph_smooth_ifg + dph_noise
            dph_smooth_ifg = []

            if la_flag == 'y':
                dph_space_uw = dph_space_uw + (K * bperp.flatten())

            if temp_flag == 'y':
                not_supported_param('temp_flag', 'y')
                # dph_space_uw=dph_space_uw+Kt*temp';   % equal to dph_space + integer cycles

            # TODO: убрать
            # diff = compare_complex_objects(dph_smooth, 'dph_smooth')
            diff = compare_objects(dph_space_uw, 'dph_space_uw')
            print("fff")
