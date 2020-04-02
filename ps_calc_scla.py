import os, sys
from datetime import date
from inspect import signature

import numpy as np
from scipy.io import loadmat

from getparm import get_parm_value as getparm
from ps_deramp import ps_deramp
from ps_setref import ps_setref


def ps_calc_scla(use_small_baselines, coest_mean_vel):
    print('');
    print('Estimating spatially-correlated look angle error...')

    sig = signature(ps_calc_scla)
    args = sig.parameters
    if len(args) < 1:
        use_small_baselines = 0;
    if len(args) < 2:
        coest_mean_vel = 0;

    # TODO: Implement getparm() function
    small_baseline_flag = getparm('small_baseline_flag')[0][0]
    drop_ifg_index = getparm('drop_ifg_index')[0]
    scla_method = getparm('scla_method')[0][0]
    scla_deramp = getparm('scla_deramp')[0][0]
    subtr_tropo = getparm('subtr_tropo')[0][0]
    tropo_method = getparm('tropo_method')[0][0]

    if use_small_baselines != 0:
        if small_baseline_flag != 'y':
            print('   Use small baselines requested but there are none')
            sys.exit()

    if use_small_baselines == 0:
        # TODO: Implement getparm() function
        scla_drop_index = []  # getparm('scla_drop_index',1);
    else:
        # TODO: if SBaS processing
        # TODO: Implement getparm() function
        scla_drop_index = []  # getparm('sb_scla_drop_index',1);
        print('   Using small baseline interferograms\n')

    psver = loadmat('psver.mat')['psver'][0][0]
    psname = 'ps' + str(psver)
    rcname = 'rc' + str(psver)
    pmname = 'pm' + str(psver)
    bpname = 'bp' + str(psver)
    meanvname = 'mv' + str(psver)
    ifgstdname = 'ifgstd' + str(psver)
    phuwsbresname = 'phuw_sb_res' + str(psver)
    if use_small_baselines == 0:
        phuwname = 'phuw' + str(psver)
        sclaname = 'scla' + str(psver)
        apsname_old = 'aps' + str(psver)  # renamed to old
        apsname = 'tca' + str(psver)  # the new tca option
    else:
        # TODO: if SBaS processing
        phuwname = 'phuw_sb' + str(psver)
        sclaname = 'scla_sb' + str(psver)
        apsname_old = 'aps_sb' + str(psver)  # renamed to old
        apsname = 'tca_sb' + str(psver)  # the new tca option

    if use_small_baselines == 0:
        os.system('rm -f ' + meanvname + '.mat')

    ps = loadmat(psname + '.mat')
    bp = {}
    if os.path.exists(bpname + '.mat'):
        bp = loadmat(bpname + '.mat')
    else:
        bperp = ps['bperp']
        if small_baseline_flag != 'y':
            bperp = np.concatenate((bperp[:ps['master_ix'][0][0] - 1], bperp[ps['master_ix'][0][0]:]), axis=0)
        bp['bperp_mat'] = np.tile(bperp.T, (ps['n_ps'][0][0], 1))
    uw = loadmat(phuwname);

    if small_baseline_flag == 'y' and use_small_baselines == 0:
        print("You set the param small_baseline_flag={}, but not supported yet.".format(
            getparm('small_baseline_flag')[0][0]))
        # unwrap_ifg_index = np.arange(0, ps['n_image'][0][0])
        # n_ifg = ps['n_image'][0][0]
    else:
        unwrap_ifg_index = np.setdiff1d(np.arange(0, ps['n_ifg'][0][0]), drop_ifg_index)
        n_ifg = ps['n_ifg'][0][0]

    if subtr_tropo == 'y':
        print("You set the param subtr_tropo={}, but not supported yet.".format(getparm('subtr_tropo')[0][0]))
        # Remove the tropo correction - TRAIN support
        # recompute the APS inversion on the fly as user migth have dropped
        # SB ifgs before and needs new update of the SM APS too.

        # if exist(apsname,'file')~=2
        # the tca file does not exist. See in case this is SM if it needs
        # to be inverted
        #    if strcmpi(apsname,['./tca',num2str(psver)])
        #        if strcmpi(getparm('small_baseline_flag'),'y')
        #           sb_invert_aps(tropo_method)
        #        end
        #     end
        #    aps = load(apsname);
        #    [aps_corr,fig_name_tca,tropo_method] = ps_plot_tca(aps,tropo_method);
        #    uw.ph_uw=uw.ph_uw-aps_corr;
        # end

    if scla_deramp == 'y':
        print('\n   deramping ifgs...\n')

        [ph_all, ph_ramp] = ps_deramp(ps.copy(), uw['ph_uw'].copy(), 1)
        uw['ph_uw'] = np.subtract(uw['ph_uw'], ph_ramp)

        # ph_ramp=zeros(ps.n_ps,n_ifg,'single');
        # G=double([ones(ps.n_ps,1),ps.xy(:,2),ps.xy(:,3)]);
        # for i=1:length(unwrap_ifg_index)
        #   d=uw.ph_uw(:,unwrap_ifg_index(i));
        #   m=G\double(d(:));
        #   ph_this_ramp=G*m;
        #   uw.ph_uw(:,unwrap_ifg_index(i))=uw.ph_uw(:,unwrap_ifg_index(i))-ph_this_ramp; % subtract ramp
        #   ph_ramp(:,unwrap_ifg_index(i))=ph_this_ramp;
        # end

    else:
        ph_ramp = []

    unwrap_ifg_index = np.setdiff1d(unwrap_ifg_index, scla_drop_index)

    # Check with Andy:
    # 1) should this not be placed before the ramp computation.
    # 2) if this is spatial fitlering in time - not compatible with TRAIN
    # if exist([apsname_old,'.mat'],'file')
    #   if strcmpi(subtr_tropo,'y')
    #       fprintf(['You are removing atmosphere twice. Do not do this, either do:\n use ' apsname_old ' with subtr_tropo=''n''\n remove ' apsname_old ' use subtr_tropo=''y''\n'])
    #   end
    #   aps=load(apsname_old);
    #   uw.ph_uw=uw.ph_uw-aps.ph_aps_slave;
    # end

    ref_ps = ps_setref()
    uw['ph_uw'] = np.subtract(uw['ph_uw'], np.tile(np.nanmean(uw['ph_uw'][ref_ps, :], 0), (ps['n_ps'][0][0], 1)))

    if use_small_baselines == 0:
        if small_baseline_flag == 'y':
            print("You set the param small_baseline_flag={}, but not supported yet.".format(
                getparm('small_baseline_flag')[0][0]))
            # bperp_mat=zeros(ps.n_ps,ps.n_image,'single');
            # G=zeros(ps.n_ifg,ps.n_image);
            # for i=1:ps.n_ifg
            #    G(i,ps.ifgday_ix(i,1))=-1;
            #    G(i,ps.ifgday_ix(i,2))=1;
            # end
            # if isfield(uw,'unwrap_ifg_index_sm')
            #    unwrap_ifg_index=setdiff(uw.unwrap_ifg_index_sm,scla_drop_index);
            # end
            # unwrap_ifg_index=setdiff(unwrap_ifg_index,ps.master_ix);

            # G=G(:,unwrap_ifg_index);
            # bperp_some=[G\double(bp.bperp_mat')]';
            # bperp_mat(:,unwrap_ifg_index)=bperp_some;
            # clear bperp_some
        else:
            bperp_mat = np.append(
                np.append(bp['bperp_mat'][:, 0:ps['master_ix'][0][0] - 1], np.zeros((ps['n_ps'][0][0], 1)), 1),
                bp['bperp_mat'][:, ps['master_ix'][0][0] - 1:], 1)

        day = np.diff((ps['day'][unwrap_ifg_index]), axis=0)
        ph = np.diff(uw['ph_uw'][:, unwrap_ifg_index], 1)
        bperp = np.diff(bperp_mat[:, unwrap_ifg_index], 1)

    else:
        print("You set the param use_small_baselines={}, but not supported yet.".format(
            getparm('use_small_baselines')[0][0]))

        # bperp_mat=bp.bperp_mat;
        # bperp=bperp_mat(:,unwrap_ifg_index);
        # day=ps.ifgday(unwrap_ifg_index,2)-ps.ifgday(unwrap_ifg_index,1);
        # ph=double(uw.ph_uw(:,unwrap_ifg_index));
    bp.clear()

    bprint = np.mean(bperp, 0)
    print('PS_CALC_SCLA: {} ifgs used in estimation:'.format(len(ph[0])))

    for i in range(len(ph[0])):
        if use_small_baselines != 0:
            print("You set the param use_small_baselines={}, but not supported yet.".format(
                getparm('use_small_baselines')[0][0]))
            # logit(sprintf('   %s to %s %5d days %5d m',datestr(ps.ifgday(unwrap_ifg_index(i),1)),datestr(ps.ifgday(unwrap_ifg_index(i),2)),day(i),round(bprint(i))))
        else:
            print('PS_CALC_SCLA:     {} to {} {} days {} m'.format(
                date.fromordinal(ps['day'][unwrap_ifg_index[i]][0] - 366),
                date.fromordinal(ps['day'][unwrap_ifg_index[i + 1]][0] - 366),
                day[i][0], np.round(bprint[i])))

    K_ps_uw = np.zeros((ps['n_ps'][0][0], 1))

    if coest_mean_vel == 0 or len(unwrap_ifg_index) < 4:
        G = np.insert(np.ones((len(ph[0]), 1)), 1, np.mean(bperp, 0), axis=1)
    else:
        G = np.append(np.insert(np.ones((len(ph[0]), 1)), 1, np.mean(bperp, 0), axis=1), day, axis=1)

    ifg_vcm = np.eye(ps['n_ifg'][0][0]);

    if small_baseline_flag == 'y':
        print("You set the param small_baseline_flag={}, but not supported yet.".format(
            getparm('small_baseline_flag')[0][0]))
        # if use_small_baselines==0
        #    phuwres=load(phuwsbresname,'sm_cov');
        #    if isfield(phuwres,'sm_cov')
        #        ifg_vcm=phuwres.sm_cov;
        #    end
        # else
        #    phuwres=load(phuwsbresname,'sb_cov');
        #    if isfield(phuwres,'sb_cov')
        #        ifg_vcm=phuwres.sb_cov;
        #    end
    else:
        if os.path.exists(ifgstdname + '.mat'):
            ifgstd = loadmat(ifgstdname + '.mat');
            ifg_vcm = np.diag(np.power(ifgstd['ifg_std'] * np.pi / 180, 2).T[0])
            ifgstd.clear()

    print()
