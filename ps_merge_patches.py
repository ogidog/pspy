import os

import numpy as np
from scipy.io import loadmat

from getparm import get_parm_value as getparm
from utils import compare_objects, not_supported_value, not_supported, not_supported_param


def intersect(A, B):
    if len(B) > 0:
        A = A[..., 0] + 1j * A[..., 1]
        B = B[..., 0] + 1j * B[..., 1]
        C = np.intersect1d(A, B, return_indices=True)
        return np.dstack((C[0].real, C[0].imag))[0].astype('int'), C[1], C[2]
    return [], [], []


def ps_merge_patches(*args):
    print('Merging patches...\n')

    if len(args) < 1:
        psver = 2

    small_baseline_flag = getparm('small_baseline_flag')[0][0]
    # TODO: if grid_size != 0
    grid_size = getparm('merge_resample_size')[0][0][0]
    merge_stdev = getparm('merge_standard_dev')[0][0][0]
    phase_accuracy = 10 * np.pi / 180
    min_weight = 1 / np.power(merge_stdev, 2)
    np.random.seed(seed=1001)
    max_coh = (np.abs(sum(np.exp(complex(0, 1) * np.random.randn(1000, 1) * phase_accuracy))) / 1000)[0]

    psname = 'ps' + str(psver)
    phname = 'ph' + str(psver)
    rcname = 'rc' + str(psver)
    pmname = 'pm' + str(psver)
    phuwname = 'phuw' + str(psver)
    sclaname = 'scla' + str(psver)
    sclasbname = 'scla_sb' + str(psver)
    scnname = 'scn' + str(psver)
    bpname = 'bp' + str(psver)
    laname = 'la' + str(psver)
    incname = 'inc' + str(psver)
    hgtname = 'hgt' + str(psver)

    if os.path.exists('patch.list'):
        dirname = {'name': []}
        fid = open('patch.list', 'r')
        line = fid.readline().strip()
        while line:
            dirname['name'].append(line)
            line = fid.readline().strip()
        fid.close()
    else:
        dirname = {'name': np.array([dir for dir in os.listdir() if 'PATCH_' in dir])}

    n_patch = len(dirname['name'])
    remove_ix = []
    ij = np.zeros((0, 2))
    lonlat = np.zeros((0, 2))
    ph = np.zeros((0, 0))
    ph_rc = np.zeros((0, 0))
    ph_reref = np.zeros((0, 0))
    ph_uw = np.zeros((0, 0))
    ph_patch = np.zeros((0, 0))
    ph_res = np.zeros((0, 0))
    ph_scla = np.zeros((0, 0))
    ph_scla_sb = np.zeros((0, 0))
    ph_scn_master = np.zeros((0, 0))
    ph_scn_slave = np.zeros((0, 0))
    K_ps = np.zeros((0, 0))
    C_ps = np.zeros((0, 0))
    coh_ps = np.zeros((0, 0))
    K_ps_uw = np.zeros((0, 0))
    K_ps_uw_sb = np.zeros((0, 0))
    C_ps_uw = np.zeros((0, 0))
    C_ps_uw_sb = np.zeros((0, 0))
    bperp_mat = np.zeros((0, 0))
    la = np.zeros((0, 0))
    inc = np.zeros((0, 0))
    hgt = np.zeros((0, 0))
    amp = np.zeros((0, 0))

    for i in range(0, n_patch):
        if dirname['name'][i]:
            print('   Processing {}\n'.format(dirname['name'][i]))
            os.chdir('.' + os.path.sep + dirname['name'][i])
            ps = loadmat(psname + '.mat')
            n_ifg = ps['n_ifg'][0][0]
            if 'n_image' in ps.keys():
                n_image = ps['n_image'][0][0]
            else:
                n_image = ps['n_ifg'][0][0]

            patch = {'ij': []}
            fid = open('patch_noover.in', 'r')
            line = fid.readline().strip()
            while line:
                patch['ij'].append(int(line))
                line = fid.readline().strip()
            fid.close()
            patch['ij'] = np.array(patch['ij'])
            ix = ((ps['ij'][:, 1] >= patch['ij'][2] - 1) & (ps['ij'][:, 1] <= patch['ij'][3] - 1) & (
                    ps['ij'][:, 2] >= patch['ij'][0] - 1) & (ps['ij'][:, 2] <= patch['ij'][1] - 1))
            if sum(ix) == 0:
                ix_no_ps = 1
            else:
                ix_no_ps = 0

            if grid_size == 0:
                C, IA, IB = intersect(ps['ij'][ix, 1:3], ij)
                remove_ix = np.concatenate((remove_ix, IB), axis=0)
                C, IA, IB = intersect(ps['ij'][:, 1:3], ij)
                ix_ex = np.ones((ps['n_ps'][0][0], 1)).astype('bool').flatten()
                ix_ex[IA] = 0
                ix[ix_ex] = 1
            else:
                not_supported_value('grid_size', grid_size)
                # if grid_size ~=0 && ix_no_ps~=1
                # clear g_ij
                # xy_min=min(ps.xy(ix,:),1);
                # g_ij(:,1)=ceil((ps.xy(ix,3)-xy_min(3)+1e-9)/grid_size);
                # g_ij(:,2)=ceil((ps.xy(ix,2)-xy_min(2)+1e-9)/grid_size);
                # n_i=max(g_ij(:,1));
                # n_j=max(g_ij(:,2));
                # [g_ij,I,g_ix]=unique(g_ij,'rows');
                # [g_ix,sort_ix]=sort(g_ix);
                # ix=find(ix);
                # ix=ix(sort_ix);
                # pm=load(pmname,'ph_res','coh_ps','C_ps');
                # pm.ph_res=angle(exp(j*(pm.ph_res-repmat(pm.C_ps,1,size(pm.ph_res,2))))); % centralise about zero
                # if small_baseline_flag~='y'
                #    pm.ph_res=[pm.ph_res,pm.C_ps]; %  include master noise too
                # end
                # sigsq_noise=var([pm.ph_res],0,2);
                # coh_ps_all=abs(sum(exp(j*[pm.ph_res]),2))/n_ifg;
                # coh_ps_all(coh_ps_all>max_coh)=max_coh; % % prevent unrealistic weights
                # sigsq_noise(sigsq_noise<phase_accuracy^2)=phase_accuracy^2; % prevent unrealistic weights
                # ps_weight=1./sigsq_noise(ix);
                # ps_snr=1./(1./coh_ps_all(ix).^2-1);
                # clear pm

                # l_ix=[find(diff(g_ix));size(g_ix,1)];
                # f_ix=[1;l_ix(1:end-1)+1];
                # n_ps_g=size(f_ix,1);

                # weightsave=zeros(n_ps_g,1);
                # for i=1:n_ps_g
                #    weights=ps_weight(f_ix(i):l_ix(i));
                #    weightsum=sum(weights);
                #    weightsave(i)=weightsum;
                #    if weightsave(i)<min_weight
                #        ix(f_ix(i):l_ix(i))=0;
                #    end
                # end
                # g_ix=g_ix(ix>0);

                # if isempty(g_ix)==1
                #    ix_no_ps=1;				% Remaining PS are rejected because to sum of weights is smaller than threshold min_weight
                # end

                # l_ix=[find(diff(g_ix));size(g_ix,1)];
                # f_ix=[1;l_ix(1:end-1)+1];
                # ps_weight=ps_weight(ix>0);
                # ps_snr=ps_snr(ix>0);
                # ix=ix(ix>0);
                # n_ps_g=size(f_ix,1);
                # n_ps=length(ix);

            if grid_size == 0:
                ij = np.concatenate((ij, ps['ij'][ix, 1:3]), axis=0)
                lonlat = np.concatenate((lonlat, ps['lonlat'][ix, :]), axis=0)
            else:
                not_supported_value('grid_size', grid_size)
                # if grid_size ~=0 && ix_no_ps~=1
                # ij_g=zeros(n_ps_g,2);
                # lonlat_g=zeros(n_ps_g,2);
                # ps.ij=ps.ij(ix,:);
                # ps.lonlat=ps.lonlat(ix,:);
                # for i=1:n_ps_g
                #    weights=repmat(ps_weight(f_ix(i):l_ix(i)),1,2);
                #    ij_g(i,:)=round(sum(ps.ij(f_ix(i):l_ix(i),2:3).*weights,1)/sum(weights(:,1)));
                #    lonlat_g(i,:)=sum(ps.lonlat(f_ix(i):l_ix(i),:).*weights,1)/sum(weights(:,1));
                # end
                # ij=[ij;ij_g];
                # lonlat=[lonlat;lonlat_g];

            if os.path.exists(phname + '.mat'):
                phin = loadmat(phname)
                ph_w = phin['ph']
                phin.clear()
            else:
                ph_w = ps['ph']

            if 'ph_w' in locals() or 'ph_w' in globals():
                if grid_size == 0:
                    if len(ph) == 0:
                        ph = ph.reshape(0, np.shape(ph_w)[1])
                    ph = np.concatenate((ph, ph_w[ix, :]), axis=0)
                else:
                    not_supported_value('grid_size', grid_size)
                    # if grid_size ~=0 && ix_no_ps~=1
                    # ph_w=ph_w(ix,:);
                    # ph_g=zeros(n_ps_g,n_ifg);
                    # for i=1:n_ps_g
                    #    weights=repmat(ps_snr(f_ix(i):l_ix(i)),1,n_ifg);
                    #    ph_g(i,:)=sum(ph_w(f_ix(i):l_ix(i),:).*weights,1);
                    # end
                    # ph=[ph;ph_g];
                    # clear ph_g
                ph_w = []

            rc = loadmat(rcname + '.mat')
            if grid_size == 0:
                if len(ph_rc) == 0:
                    ph_rc = ph_rc.reshape(0, np.shape(rc['ph_rc'])[1])
                ph_rc = np.concatenate((ph_rc, rc['ph_rc'][ix, :]), axis=0)
                if small_baseline_flag != 'y':
                    if len(ph_reref) == 0:
                        ph_reref = ph_reref.reshape(0, np.shape(rc['ph_reref'])[1])
                    ph_reref = np.concatenate((ph_reref, rc['ph_reref'][ix, :]), axis=0)
            else:
                not_supported_value('grid_size', grid_size)
                # if grid_size ~=0 && ix_no_ps~=1
                # rc.ph_rc=rc.ph_rc(ix,:);
                # ph_g=zeros(n_ps_g,n_ifg);
                # if ~strcmpi(small_baseline_flag,'y')
                #    rc.ph_reref=rc.ph_reref(ix,:);
                #    ph_reref_g=zeros(n_ps_g,n_ifg);
                # end
                # for i=1:n_ps_g
                #    weights=repmat(ps_snr(f_ix(i):l_ix(i)),1,n_ifg);
                #    ph_g(i,:)=sum(rc.ph_rc(f_ix(i):l_ix(i),:).*weights,1);
                #    if ~strcmpi(small_baseline_flag,'y')
                #        ph_reref_g(i,:)=sum(rc.ph_reref(f_ix(i):l_ix(i),:).*weights,1);
                #    end
                # end
                # ph_rc=[ph_rc;ph_g];
                # clear ph_g
                # if ~strcmpi(small_baseline_flag,'y')
                #    ph_reref=[ph_reref;ph_reref_g];
                #    clear ph_reref_g
                # end
            rc.clear()

            pm = loadmat(pmname + '.mat')
            if grid_size == 0:
                if len(ph_patch) == 0:
                    ph_patch = ph_patch.reshape(0, np.shape(pm['ph_patch'])[1])
                ph_patch = np.concatenate((ph_patch, pm['ph_patch'][ix, :]), axis=0)
                if 'ph_res' in pm.keys():
                    if len(ph_res) == 0:
                        ph_res = ph_res.reshape(0, np.shape(pm['ph_res'])[1])
                    ph_res = np.concatenate((ph_res, pm['ph_res'][ix, :]), axis=0)
                if 'K_ps' in pm.keys():
                    if len(K_ps) == 0:
                        K_ps = K_ps.reshape(0, np.shape(pm['K_ps'])[1])
                    K_ps = np.concatenate((K_ps, pm['K_ps'][ix, :]), axis=0)
                if 'C_ps' in pm.keys():
                    if len(C_ps) == 0:
                        C_ps = C_ps.reshape(0, np.shape(pm['C_ps'])[1])
                    C_ps = np.concatenate((C_ps, pm['C_ps'][ix, :]), axis=0)
                if 'coh_ps' in pm.keys():
                    if len(coh_ps) == 0:
                        coh_ps = coh_ps.reshape(0, np.shape(pm['coh_ps'])[1])
                    coh_ps = np.concatenate((coh_ps, pm['coh_ps'][ix, :]), axis=0)
            else:
                not_supported_value('grid_size', grid_size)
                # if grid_size ~=0 && ix_no_ps~=1
                # pm.ph_patch=pm.ph_patch(ix,:);
                # ph_g=zeros(n_ps_g,size(pm.ph_patch,2));
                # if isfield(pm,'ph_res')
                #    pm.ph_res=pm.ph_res(ix,:);
                #    ph_res_g=ph_g;
                # end
                # if isfield(pm,'K_ps')
                #    pm.K_ps=pm.K_ps(ix,:);
                #    K_ps_g=zeros(n_ps_g,1);
                # end
                # if isfield(pm,'C_ps')
                #    pm.C_ps=pm.C_ps(ix,:);
                #    C_ps_g=zeros(n_ps_g,1);
                # end
                # if isfield(pm,'coh_ps')
                #    pm.coh_ps=pm.coh_ps(ix,:);
                #    coh_ps_g=zeros(n_ps_g,1);
                # end
                # for i=1:n_ps_g
                #    weights=repmat(ps_snr(f_ix(i):l_ix(i)),1,size(ph_g,2));
                #    ph_g(i,:)=sum(pm.ph_patch(f_ix(i):l_ix(i),:).*weights,1);
                #    if isfield(pm,'ph_res')
                #        ph_res_g(i,:)=sum(pm.ph_res(f_ix(i):l_ix(i),:).*weights,1);
                #    end
                #    if isfield(pm,'coh_ps')
                #        snr=sqrt(sum(weights(:,1).^2,1));
                #        coh_ps_g(i)=sqrt(1./(1+1./snr));
                #    end
                #    weights=ps_weight(f_ix(i):l_ix(i));
                #    if isfield(pm,'K_ps')
                #        K_ps_g(i)=sum(pm.K_ps(f_ix(i):l_ix(i),:).*weights,1)./sum(weights,1);
                #    end
                #    if isfield(pm,'C_ps')
                #        C_ps_g(i)=sum(pm.C_ps(f_ix(i):l_ix(i),:).*weights,1)./sum(weights,1);
                #    end

                #    if sum(sum(isnan(C_ps_g)))>0 || sum(sum(isnan(weights)))>0 || sum(sum(isnan(ph_g)))>0 ||  sum(sum(isnan(K_ps_g)))>0 ||  sum(sum(isnan(coh_ps_g)))>0 ||  sum(sum(isnan(snr)))>0
                #        keyboard
                #    end
                # end
                # ph_patch=[ph_patch;ph_g];
                # clear ph_g
                # if isfield(pm,'ph_res')
                #    ph_res=[ph_res;ph_res_g];
                #    clear ph_res_g
                # end
                # if isfield(pm,'K_ps')
                #    K_ps=[K_ps;K_ps_g];
                #    clear K_ps_g
                # end
                # if isfield(pm,'C_ps')
                #    C_ps=[C_ps;C_ps_g];
                #    clear C_ps_g
                # end
                # if isfield(pm,'coh_ps')
                #    coh_ps=[coh_ps;coh_ps_g];
                #    clear coh_ps_g
                # end
            pm.clear()

            bp = loadmat(bpname + '.mat')
            if grid_size == 0:
                if len(bperp_mat) == 0:
                    bperp_mat = bperp_mat.reshape(0, np.shape(bp['bperp_mat'])[1])
                bperp_mat = np.concatenate((bperp_mat, bp['bperp_mat'][ix, :]), axis=0)
            else:
                not_supported_value('grid_size', grid_size)
                # if grid_size ~=0 & & ix_no_ps~=1
                #    bperp_g = zeros(n_ps_g, size(bp.bperp_mat, 2));
                #    bp.bperp_mat = bp.bperp_mat(ix,:);
                #    for i=1:n_ps_g
                #        weights = repmat(ps_weight(f_ix(i):l_ix(i)), 1, size(bperp_g, 2));
                #        weights(weights == 0) = 1e-9; % pixels with zero phase cause this problem
                #        bperp_g(i,:)=sum(bp.bperp_mat(f_ix(i): l_ix(i),:).*weights, 1) / sum(weights(:, 1));
                #    end
                #    bperp_mat = [bperp_mat;
                #    bperp_g];
                #    clear
                #    bperp_g
            bp.clear()

            if os.path.exists(laname + '.mat'):
                lain = loadmat(laname + '.mat')
                if grid_size == 0:
                    if len(la) == 0:
                        la = la.reshape(0, np.shape(lain['la'])[1])
                    la = np.concatenate((la, lain['la'][ix, :]), axis=0)
                else:
                    not_supported_value('grid_size', grid_size)
                    # if grid_size ~=0 && ix_no_ps~=1
                    #    la_g=zeros(n_ps_g,1);
                    #    lain.la=lain.la(ix,:);
                    #    for i=1:n_ps_g
                    #        weights=ps_weight(f_ix(i):l_ix(i));
                    #        la_g(i)=sum(lain.la(f_ix(i):l_ix(i)).*weights,1)/sum(weights(:,1));
                    #    end
                    #    la=[la;la_g];
                    #    clear la_g
                lain.clear()

            if os.path.exists(incname + '.mat'):
                not_supported()
                # incin = loadmat(incname + '.mat')
                # if grid_size==0
                #    inc=[inc;incin.inc(ix,:)];
                # elseif grid_size ~=0 && ix_no_ps~=1
                #    inc_g=zeros(n_ps_g,1);
                #    incin.inc=incin.inc(ix,:);
                #    for i=1:n_ps_g
                #        weights=ps_weight(f_ix(i):l_ix(i));
                #        inc_g(i)=sum(incin.inc(f_ix(i):l_ix(i)).*weights,1)/sum(weights(:,1));
                #    end
                #    inc=[inc;inc_g];
                #    clear inc_g
                # end
                # clear incin

            if os.path.exists(hgtname + '.mat'):
                hgtin = loadmat(hgtname + '.mat')
                if grid_size == 0:
                    if len(hgt) == 0:
                        hgt = hgt.reshape(0, np.shape(hgtin['hgt'])[1])
                    hgt = np.concatenate((hgt, hgtin['hgt'][ix, :]), axis=0)
                else:
                    not_supported_value('grid_size', grid_size)
                #    if grid_size ~=0 && ix_no_ps~=1
                #        hgt_g=zeros(n_ps_g,1);
                #        hgtin.hgt=hgtin.hgt(ix,:);
                #        for i=1:n_ps_g
                #            weights=ps_weight(f_ix(i):l_ix(i));
                #            hgt_g(i)=sum(hgtin.hgt(f_ix(i):l_ix(i)).*weights,1)/sum(weights(:,1));
                #        end
                #        hgt=[hgt;hgt_g];
                #        clear hgt_g
                #    end
                hgtin.clear()

            if grid_size == 0:
                if os.path.exists(phuwname + '.mat'):
                    not_supported()
                    # phuw = load(phuwname)
                    # if ~isempty(C)
                    #    ph_uw_diff=mean(phuw.ph_uw(IA,:)-ph_uw(IB,:),1);
                    #    if ~strcmpi(small_baseline_flag,'y')
                    #       ph_uw_diff=round(ph_uw_diff/2/pi)*2*pi; % round to nearest 2 pi
                    #    end
                    # else
                    #    ph_uw_diff=zeros(1,size(phuw.ph_uw,2));
                    # end
                    # ph_uw=[ph_uw;phuw.ph_uw(ix,:)-repmat(ph_uw_diff,sum(ix),1)];
                    # clear phuw
                else:
                    zeros = np.zeros((np.sum(ix), n_image))
                    ph_uw = np.concatenate((ph_uw, zeros), axis=0)

                if os.path.exists(sclaname + '.mat'):
                    not_supported()
                    # scla=load(sclaname);
                    # if ~isempty(C)
                    #    ph_scla_diff=mean(scla.ph_scla(IA,:)-ph_scla(IB,:));
                    #    K_ps_diff=mean(scla.K_ps_uw(IA,:)-K_ps_uw(IB,:));
                    #    C_ps_diff=mean(scla.C_ps_uw(IA,:)-C_ps_uw(IB,:));
                    # else
                    #    ph_scla_diff=zeros(1,size(scla.ph_scla,2));
                    #    K_ps_diff=0;
                    #    C_ps_diff=0;
                    # end
                    # ph_scla=[ph_scla;scla.ph_scla(ix,:)-repmat(ph_scla_diff,sum(ix),1)];
                    # K_ps_uw=[K_ps_uw;scla.K_ps_uw(ix,:)-repmat(K_ps_diff,sum(ix),1)];
                    # C_ps_uw=[C_ps_uw;scla.C_ps_uw(ix,:)-repmat(C_ps_diff,sum(ix),1)];
                    # clear scla

                if small_baseline_flag == 'y':
                    not_supported_param('small_baseline_flag', 'y')
                    # if exist(['./',sclasbname,'.mat'],'file')
                    #    sclasb=load(sclasbname);
                    #    ph_scla_diff=mean(sclasb.ph_scla(IA,:)-ph_scla_sb(IB,:));
                    #    K_ps_diff=mean(sclasb.K_ps_uw(IA,:)-K_ps_uw_sb(IB,:));
                    #    C_ps_diff=mean(sclasb.C_ps_uw(IA,:)-C_ps_uw_sb(IB,:));
                    #    ph_scla_sb=[ph_scla_sb;sclasb.ph_scla(ix,:)-repmat(ph_scla_diff,sum(ix),1)];
                    #    K_ps_uw_sb=[K_ps_uw_sb;sclasb.K_ps_uw(ix,:)-repmat(K_ps_diff,sum(ix),1)];
                    #    C_ps_uw_sb=[C_ps_uw_sb;sclasb.C_ps_uw(ix,:)-repmat(C_ps_diff,sum(ix),1)];
                    #    clear sclasb
                    # end

                if os.path.exists(scnname + '.mat'):
                    not_supported()
                    # scn=load(scnname);

                    # if ~isempty(C)
                    #    ph_scn_diff=mean(scn.ph_scn_slave(IA,:)-ph_scn_slave(IB,:));
                    # else
                    #    ph_scn_diff=zeros(1,size(scn.ph_scn_slave,2));
                    # end
                    # ph_scn_slave=[ph_scn_slave;scn.ph_scn_slave(ix,:)-repmat(ph_scn_diff,sum(ix),1)];
                    # clear scn

            os.chdir('..')

    # diff = compare_objects(ix, 'ix')
    print(os.getcwd())
