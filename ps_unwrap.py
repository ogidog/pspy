import os, sys
import numpy as np

from scipy.io import loadmat, savemat
from getparm import get_parm_value as getparm
from uw_3d import uw_3d

from utils import compare_objects, not_supported_param


def ps_unwrap():
    print('Phase-unwrapping...\n')

    small_baseline_flag = getparm('small_baseline_flag')[0][0]
    unwrap_patch_phase = getparm('unwrap_patch_phase')[0][0]
    scla_deramp = getparm('scla_deramp')[0][0]
    subtr_tropo = getparm('subtr_tropo')[0][0]
    aps_name = getparm('tropo_method')[0][0]

    psver = loadmat('psver.mat')['psver'][0][0]
    psname = 'ps' + str(psver)
    rcname = 'rc' + str(psver)
    pmname = 'pm' + str(psver)
    bpname = 'bp' + str(psver)
    goodname = 'phuw_good' + str(psver)

    if small_baseline_flag != 'y':
        sclaname = 'scla_smooth' + str(psver)
        apsname = 'tca' + str(psver)
        phuwname = 'phuw' + str(psver)
    else:
        print("You set the param small_baseline_flag={}, but not supported yet.".format(
            getparm('small_baseline_flag')[0][0]))
        sys.exit()
        # sclaname=['scla_smooth_sb',num2str(psver)];
        # apsname=['tca_sb',num2str(psver)];
        # phuwname=['phuw_sb',num2str(psver),'.mat'];

    ps = loadmat(psname + '.mat');

    drop_ifg_index = getparm('drop_ifg_index')[0]
    unwrap_ifg_index = np.setdiff1d(np.arange(0, ps['n_ifg'][0][0]), drop_ifg_index)

    bp = {}
    if os.path.exists(bpname + '.mat'):
        bp = loadmat(bpname + '.mat')
    else:
        bperp = ps['bperp']
        if small_baseline_flag != 'y':
            bperp = np.concatenate((bperp[:ps['master_ix'][0][0] - 1], bperp[ps['master_ix'][0][0]:]), axis=0)
        bp['bperp_mat'] = np.tile(bperp.T, (ps['n_ps'][0][0], 1))

    if small_baseline_flag != 'y':
        bperp_mat = np.concatenate((bp['bperp_mat'][:, 0:ps['master_ix'][0][0] - 1],
                                    np.zeros(ps['n_ps'][0][0]).reshape(-1, 1),
                                    bp['bperp_mat'][:, ps['master_ix'][0][0] - 1:]), axis=1)
    else:
        print("You set the param small_baseline_flag={}, but not supported yet.".format(
            getparm('small_baseline_flag')[0][0]))
        sys.exit()
        # bperp_mat=bp.bperp_mat;

    if unwrap_patch_phase == 'y':
        pm = loadmat(pmname);
        ph_w = np.divide(pm['ph_patch'], np.abs(pm['ph_patch']))
        pm.clear()
        if small_baseline_flag != 'y':
            ph_w = np.concatenate((ph_w[:, 0:ps['master_ix'][0][0] - 1], np.ones((ps['n_ps'][0][0], 1)),
                                   ph_w[:, ps['master_ix'][0][0] - 1:]), axis=1)
    else:
        rc = loadmat(rcname + '.mat')
        ph_w = rc['ph_rc']
        rc.clear()
        if os.path.exists(pmname + '.mat'):
            pm = loadmat(pmname + '.mat')
            if 'K_ps' in pm.keys():
                if bool(pm['K_ps'][0]):
                    ph_w = np.multiply(ph_w, np.exp(
                        np.multiply(complex(0.0, 1.0) * np.tile(pm['K_ps'], (1, ps['n_ifg'][0][0])), bperp_mat)))

    ix = np.array([ph_w[i, :] != 0 for i in range(len(ph_w))])
    ph_w[ix] = np.divide(ph_w[ix], np.abs(ph_w[ix]))

    scla_subtracted_sw = 0
    ramp_subtracted_sw = 0

    options = {'master_day': ps['master_day'][0][0]}
    unwrap_hold_good_values = getparm('unwrap_hold_good_values')[0][0]
    if small_baseline_flag != 'y' or os.path.exists(phuwname + '.mat'):
        unwrap_hold_good_values = 'n';
        print('Code to hold good values skipped')

    if unwrap_hold_good_values == 'y':
        print("You set the param unwrap_hold_good_values={}, but not supported yet.".format(
            getparm('unwrap_hold_good_values')[0][0]))
        sys.exit()
        # sb_identify_good_pixels
        # options.ph_uw_predef=nan(size(ph_w),'single');
        # uw=load(phuwname);
        # good=load(goodname);
        # if ps.n_ps==size(good.good_pixels,1) & ps.n_ps==size(uw.ph_uw,1)
        #    options.ph_uw_predef(good.good_pixels)=uw.ph_uw(good.good_pixels);
        # else
        #    fprintf('   wrong number of PS in keep good pixels - skipped...\n')
        # end
        # clear uw good;

    if small_baseline_flag != 'y' and os.path.exists(sclaname + '.mat'):
        print('   subtracting scla and master aoe...\n')
        scla = loadmat(sclaname + '.mat')
        if len(scla['K_ps_uw']) == ps['n_ps'][0][0]:
            scla_subtracted_sw = 1
            ph_w = np.multiply(ph_w, np.exp(
                np.multiply(complex(0.0, -1.0) * np.tile(scla['K_ps_uw'], (1, ps['n_ifg'][0][0])), bperp_mat)))
            ph_w = np.multiply(ph_w, np.tile(np.exp(complex(0.0, -1.0) * scla['C_ps_uw']), (1, ps['n_ifg'][0][0])))

            if scla_deramp == 'y' and 'ph_ramp' in scla.keys() and len(scla['ph_ramp']) == ps['n_ps'][0][0]:
                ramp_subtracted_sw = 1
                ph_w = np.multiply(ph_w, np.exp(complex(0.0, -1.0) * scla['ph_ramp']))
        else:
            print('   wrong number of PS in scla - subtraction skipped...\n')
            os.remove(sclaname + '.mat')
        scla.clear()

    if small_baseline_flag == 'y' and os.path.exists(sclaname + '.mat'):
        print("You set the param small_baseline_flag={}, but not supported yet.".format(
            getparm('small_baseline_flag')[0][0]))
        sys.exit()
    #    fprintf('   subtracting scla...\n')
    #    scla=load(sclaname);
    #    if size(scla.K_ps_uw,1)==ps.n_ps
    #        scla_subtracted_sw=1;
    #        ph_w=ph_w.*exp(-j*repmat(scla.K_ps_uw,1,ps.n_ifg).*bperp_mat); % subtract spatially correlated look angle error
    #        if unwrap_hold_good_values=='y'
    #            options.ph_uw_predef=options.ph_uw_predef-repmat(scla.K_ps_uw,1,ps.n_ifg).*bperp_mat; % subtract spatially correlated look angle error
    #        end
    #        if strcmpi(scla_deramp,'y') & isfield(scla,'ph_ramp') & size(scla.ph_ramp,1)==ps.n_ps
    #           ramp_subtracted_sw=1;
    #           ph_w=ph_w.*exp(-j*scla.ph_ramp); % subtract orbital ramps
    #           if unwrap_hold_good_values=='y'
    #               options.ph_uw_predef=options.ph_uw_predef-scla.ph_ramp;
    #           end
    #       end
    #   else
    #       fprintf('   wrong number of PS in scla - subtraction skipped...\n')
    #       delete([sclaname,'.mat'])
    #   end
    #   clear scla
    # end

    bp.clear()

    if os.path.exists(apsname + '.mat') and subtr_tropo == 'y':
        print("You set the param subtr_tropo={}, but not supported yet.".format(
            getparm('subtr_tropo')[0][0]))
        sys.exit()
    #    fprintf('   subtracting slave aps...\n')
    #    aps=load(apsname);
    #    [aps_corr,fig_name_tca,aps_flag] = ps_plot_tca(aps,aps_name);

    #   ph_w=ph_w.*exp(-j*aps_corr);
    #    if unwrap_hold_good_values=='y'
    #        options.ph_uw_predef=options.ph_uw_predef-aps_corr;
    #   end
    #   clear aps

    options['time_win'] = getparm('unwrap_time_win')[0][0][0]
    options['unwrap_method'] = getparm('unwrap_method')[0][0]
    options['grid_size'] = getparm('unwrap_grid_size')[0][0][0]
    options['prefilt_win'] = getparm('unwrap_gold_n_win')[0][0][0]
    options['goldfilt_flag'] = getparm('unwrap_prefilter_flag')[0][0]
    options['gold_alpha'] = getparm('unwrap_gold_alpha')[0][0][0]
    options['la_flag'] = getparm('unwrap_la_error_flag')[0][0]
    options['scf_flag'] = getparm('unwrap_spatial_cost_func_flag')[0][0]

    max_topo_err = getparm('max_topo_err')[0][0][0]
    _lambda = getparm('lambda')[0][0][0]

    rho = 830000
    if 'mean_incidence' in ps.keys():
        inc_mean = ps['mean_incidence'][0][0]
    else:
        laname = 'la' + str(psver)
        if os.path.exists(laname + '.mat'):
            la = loadmat(laname + '.mat')
            inc_mean = np.mean(la['la']) + 0.052
            la.clear()
        else:
            inc_mean = 21 * np.pi / 180.0
    max_K = max_topo_err / (_lambda * rho * np.sin(inc_mean) / 4 / np.pi)

    bperp_range = np.amax(ps['bperp']) - np.amin(ps['bperp'])
    options['n_trial_wraps'] = (bperp_range * max_K / (2 * np.pi))
    print('n_trial_wraps={}'.format(options['n_trial_wraps']))

    if small_baseline_flag == 'y':
        print("You set the param small_baseline_flag={}, but not supported yet.".format(
            getparm('small_baseline_flag')[0][0]))
        sys.exit()
        # %options.lowfilt_flag='y';
        # options.lowfilt_flag='n';
        # ifgday_ix=ps.ifgday_ix;
        # day=ps.day-ps.master_day;
    else:
        lowfilt_flag = 'n'
        ifgday_ix = np.concatenate((np.ones((ps['n_ifg'][0][0], 1)) * ps['master_ix'],
                                    np.array([x for x in range(ps['n_ifg'][0][0])]).reshape(-1, 1)), axis=1).astype(
            'int')
        master_ix = np.sum(ps['master_day'] > ps['day']) + 1
        unwrap_ifg_index = np.setdiff1d(unwrap_ifg_index, master_ix - 1)
        day = ps['day'] - ps['master_day']

    if unwrap_hold_good_values == 'y':
        print("You set the param unwrap_hold_good_values={}, but not supported yet.".format(
            getparm('small_baseline_flag')[0][0]))
        sys.exit()
        # options.ph_uw_predef=options.ph_uw_predef(:,unwrap_ifg_index);

    ph_uw_some, msd_some = uw_3d(ph_w[:, unwrap_ifg_index], ps['xy'], day, ifgday_ix[unwrap_ifg_index, :],
                                 ps['bperp'][unwrap_ifg_index], options)

    print()
