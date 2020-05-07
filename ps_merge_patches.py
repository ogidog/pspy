import os

import numpy as np
from scipy.io import loadmat, savemat

from getparm import get_parm_value as getparm
from llh2local import llh2local
from utils import compare_objects, compare_mat_file, not_supported_value, not_supported, not_supported_param


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

            incin = {}
            if os.path.exists(incname + '.mat'):
                not_supported()
                incin = loadmat(incname + '.mat')
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
                    if len(ph_uw) == 0:
                        ph_uw = ph_uw.reshape(0, np.shape(zeros)[1])
                    ph_uw = np.concatenate((ph_uw, zeros), axis=0)

                if os.path.exists(sclaname + '.mat'):
                    not_supported()
                    scla = loadmat(sclaname + '.mat')
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

    ps_new = ps
    n_ps_orig = len(ij)
    keep_ix = ix_ex = np.ones((n_ps_orig, 1)).astype('bool').flatten()
    keep_ix[remove_ix.astype('int')] = 0
    lonlat_save = lonlat
    coh_ps_weed = coh_ps[keep_ix]
    lonlat = lonlat[keep_ix, :]

    dummy, I = np.unique(lonlat, return_index=True, axis=0)
    dups = np.setxor1d(I, np.array([*range(len(lonlat))]))
    keep_ix_num = np.nonzero(keep_ix)[0]

    for i in range(len(dups)):
        not_supported()
        # dups_ix_weed=find(lonlat(:,1)==lonlat(dups(i),1)&lonlat(:,2)==lonlat(dups(i),2));
        # dups_ix=keep_ix_num(dups_ix_weed);
        # [dummy,I]=max(coh_ps_weed(dups_ix_weed));
        # keep_ix(dups_ix([1:end]~=I))=0; % drop dups with lowest coh

    if len(dups) > 0:
        lonlat = lonlat_save[keep_ix, :]
        print('   {} pixel with duplicate lon/lat dropped\n'.format(len(dups)))

    lonlat_save = []

    ll0 = (np.max(lonlat, axis=0) + np.min(lonlat, axis=0)) / 2
    xy = llh2local(lonlat.T, ll0) * 1000
    xy = xy.T
    sort_x = xy[np.argsort(xy[:, 0], kind='stable')]
    sort_y = xy[np.argsort(xy[:, 1], kind='stable')]

    n_pc = int(np.round(len(xy) * 0.001))
    bl = np.mean(sort_x[0:n_pc, :], axis=0)
    tr = np.mean(sort_x[len(sort_x) - n_pc - 1:, :], axis=0)
    br = np.mean(sort_y[0:n_pc, :], axis=0)
    tl = np.mean(sort_y[len(sort_y) - n_pc - 1:, :], axis=0)

    try:
        heading = getparm('heading')[0][0][0]
    except:
        heading = 0
    theta = (180 - heading) * np.pi / 180
    if theta > np.pi:
        theta = theta - 2 * np.pi

    rotm = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    xy = xy.T
    xynew = np.dot(rotm, xy)

    if max(xynew[0, :]) - min(xynew[0, :]) < max(xy[0, :]) - min(xy[0, :]) and max(xynew[1, :]) - min(
            xynew[1, :]) < max(
        xy[1, :]) - min(xy[1, :]):
        xy = xynew
        print('   Rotating xy by {}  degrees'.format(str(theta * 180 / np.pi)))

    xynew = []
    xy = xy.astype('float32').T
    xy_list = xy.tolist()
    xy_sort = np.array(sorted(xy_list, key=lambda t: (t[1], t[0])))
    sort_ix = np.array(sorted(range(len(xy_list)), key=lambda s: (xy_list[s][1], xy_list[s][0])))

    xy = xy[sort_ix, :]
    xy = np.concatenate((np.array([*range(len(xy))]).reshape(-1, 1) + 1, xy), axis=1)
    xy[:, 1:3] = np.round(xy[:, 1:3] * 1000) / 1000
    lonlat = lonlat[sort_ix, :]

    all_ix = np.array([*range(len(ij))]).reshape(-1, 1)
    keep_ix = all_ix[keep_ix]
    sort_ix = keep_ix[sort_ix]

    n_ps = len(sort_ix)
    print('   Writing merged dataset (contains {} pixels)\n'.format(str(n_ps)))

    ij = ij[sort_ix.flatten(), :]

    ph_rc = ph_rc[sort_ix.flatten(), :]
    ph_rc[ph_rc != 0] = np.divide(ph_rc[ph_rc != 0], np.abs(ph_rc[ph_rc != 0]))
    if small_baseline_flag != 'y':
        ph_reref = ph_reref[sort_ix.flatten(), :]

    rc['ph_rc'] = ph_rc
    rc['ph_reref'] = ph_reref
    savemat(rcname + '.mat', rc)
    ph_rc = []
    ph_reref = []

    if len(ph_uw) == n_ps_orig:
        ph_uw = ph_uw[sort_ix.flatten(), :]
        phuw2 = {'ph_uw': ph_uw}
        savemat(phuwname + '.mat', phuw2)
    ph_uw = []

    ph_patch = ph_patch[sort_ix.flatten(), :]
    if len(ph_res) == n_ps_orig:
        ph_res = ph_res[sort_ix.flatten(), :]
    else:
        ph_res = []
    if len(K_ps) == n_ps_orig:
        K_ps = K_ps[sort_ix.flatten(), :]
    else:
        K_ps = []
    if len(C_ps) == n_ps_orig:
        C_ps = C_ps[sort_ix.flatten(), :]
    else:
        C_ps = []
    if len(coh_ps) == n_ps_orig:
        coh_ps = coh_ps[sort_ix.flatten(), :]
    else:
        coh_ps = []
    pm['ph_patch'] = ph_patch
    pm['ph_res'] = ph_res
    pm['K_ps'] = K_ps
    pm['C_ps'] = C_ps
    pm['coh_ps'] = coh_ps
    savemat(pmname + '.mat', pm)
    ph_patch = []
    ph_res = []
    K_ps = []
    C_ps = []
    coh_ps = []

    if len(ph_scla) == n_ps:
        ph_scla = ph_scla[sort_ix.flatten(), :]
        K_ps_uw = K_ps_uw[sort_ix.flatten(), :]
        C_ps_uw = C_ps_uw[sort_ix.flatten(), :]
        scla['ph_scla'] = ph_scla
        scla['K_ps_uw'] = K_ps_uw
        scla['C_ps_uw'] = C_ps_uw
        savemat(sclaname + '.mat', scla)
    ph_scla = []
    K_ps_uw = []
    C_ps_uw = []

    if small_baseline_flag == 'y' and len(ph_scla_sb) == n_ps:
        not_supported_param('small_baseline_flag', 'y')
        # ph_scla=ph_scla_sb(sort_ix,:);
        # K_ps_uw=K_ps_uw_sb(sort_ix,:);
        # C_ps_uw=C_ps_uw_sb(sort_ix,:);
        # stamps_save(sclasbname,ph_scla,K_ps_uw,C_ps_uw);
        # clear ph_scla K_ps_uw C_ps_uw
    # clear ph_scla_sb K_ps_uw_sb C_ps_uw_sb

    if len(ph_scn_slave) == n_ps:
        not_supported()
        # ph_scn_slave=ph_scn_slave(sort_ix,:);
        # stamps_save(scnname,ph_scn_slave);
    # clear ph_scn_slave

    if len(ph) == n_ps_orig:
        ph = ph[sort_ix.flatten(), :]
    else:
        ph = []
    phin['ph'] = ph
    savemat(phname + '.mat', phin)
    ph = []

    if len(la) == n_ps_orig:
        la = la[sort_ix.flatten(), :]
    else:
        la = []
    lain['la'] = la
    savemat(laname + '.mat', lain)
    la = []

    if len(inc) == n_ps_orig:
        inc = inc[sort_ix.flatten(), :]
    else:
        inc = []
    incin['inc'] = inc
    savemat(incname + '.mat', incin)
    inc = []

    if len(hgt) == n_ps_orig:
        hgt = hgt[sort_ix.flatten(), :]
    else:
        hgt = []
    hgtin['hgt'] = hgt
    savemat(hgtname + '.mat', hgtin)
    hgt = []

    bperp_mat = bperp_mat[sort_ix.flatten(), :]
    bp['bperp_mat'] = bperp_mat
    savemat(bpname + '.mat', bp)
    bperp_mat = []

    ps_new['n_ps'] = n_ps
    ps_new['ij'] = np.concatenate(((np.array([*range(n_ps)]) + 1).reshape(-1, 1), ij), axis=1)
    ps_new['xy'] = xy
    ps_new['lonlat'] = lonlat
    savemat(psname + '.mat', ps_new)
    ps_new.clear()

    savemat('psver.mat', {'psver': psver})

    if os.path.exists('mean_amp.flt'):
        os.remove('mean_amp.flt')

    if os.path.exists('amp_mean.mat'):
        os.remove('amp_mean.mat')