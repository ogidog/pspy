import os, sys, time
import numpy as np
from datetime import datetime
import struct
from scipy.io import loadmat, savemat

from llh2local import llh2local
from readparm import readparm
# from setparm import setparm

def v2c(v):
    m = len(v)
    out = np.reshape(v, (m, 1))
    return(out)

def v2r(v):
    m = len(v)
    out = np.reshape(v, (1, m))
    return(out)

def setprm(fname, pname, data):
    prms = loadmat(fname)
    prms[pname] = data
    savemat(fname, prms)
    
def ps_load_initial_gamma(*args):
    
    begin_time = time.time()
    
    if len(args) == 2: # To debug
        path_to_task = args[0] + os.sep
        path_to_patch = path_to_task + 'PATCH_' + str(args[1]) + os.sep
    
    else: # To essential run
        path_to_task = os.sep.join(os.getcwd().split(os.path.sep)[:-1]) + os.sep
        path_to_patch = os.getcwd() + os.sep
    
    avr_flag = False
            
    print()    
    print('********* Stage 1 ********')
    print('*** Load initial gamma ***')
    print('')
    print('Work dir:', path_to_patch)

    psver = str(1) # str(loadmat(path_to_patch + 'psver.mat', squeeze_me = True)['psver'])

    phname  = path_to_patch + 'pscands.1.ph'
    ijname  = path_to_patch + 'pscands.1.ij'
    llname  = path_to_patch + 'pscands.1.ll'
    # xyname  = path_to_patch + 'pscands.1.xy'
    hgtname = path_to_patch + 'pscands.1.hgt'
    daname  = path_to_patch + 'pscands.1.da'
    
    rscname = path_to_task + 'rsc.txt'
    pscname = path_to_task + 'pscphase.in'
    
    if not os.path.exists(rscname):
        print("* File {} not exist.".format(rscname))
        sys.exit()
    
    f = open(rscname)
    rslcpar = path_to_task + 'rslc' + os.sep + os.path.split(f.read().strip())[1]
    f.close()

    if not os.path.exists(pscname):
        print("* File {} not exist.".format(pscname))
        sys.exit()

    f = open(pscname)
    ifgs = f.read()
    f.close()
    
    ifgs = ifgs.split("\n")[1:-1]
    master_day = datetime.strptime(os.path.split(ifgs[0])[1].split("_")[0], "%Y%m%d").date().toordinal() + 366
    ifgs = [path_to_task + "diff0/" + os.path.split(ifg)[1] for ifg in ifgs]
    n_ifg = len(ifgs)
    n_image = n_ifg

    day = [(datetime.strptime(str(os.path.split(ifg)[1].split("_")[1].replace(".diff", "")), "%Y%m%d")).date().toordinal() + 366 for ifg in ifgs]

    master_ix = len(np.where(np.array(day) < master_day)[0]) + 1
    if day[master_ix] != master_day:
        master_master_flag = '0'  # no null master-master ifg provided
        day.insert(master_ix - 1, master_day)
    else:
        master_master_flag = '1'  # yes, null master-master ifg provided

    heading = float(readparm(rslcpar, 'heading:'))
    setprm(path_to_task + 'parms.mat', 'heading', heading)
    
    freq = float(readparm(rslcpar, 'radar_frequency:'))
    lambda1 = 299792458 / freq
    setprm(path_to_task + 'parms.mat', 'lambda', lambda1)

    sensor = readparm(rslcpar, 'sensor:')
    if 'ASAR' in sensor:
        platform = 'ENVISAT'
    else:
        platform = sensor  # S1A for Sentinel-1A
    setprm(path_to_task + 'parms.mat', 'platform', platform)

    f = open(ijname)
    lines = f.readlines()
    f.close()
    ij = np.empty((0, 3), int)
    lines_count = len(lines)
    
    for i in range(lines_count):
        # print('* Load pixel', i + 1, 'from', lines_count, end = '\r', flush = True)
        line = lines[i].strip()
        ij = np.append(ij, np.array([line.split(" ")]).astype("int"), axis=0)
    
    print('* Loaded pixels:', lines_count)
    n_ps = np.size(ij, 0)

    # np.save('ij.npy', ij)

    rps = int(readparm(rslcpar, 'range_pixel_spacing'))
    rgn = float(readparm(rslcpar, 'near_range_slc'))
    se = float(readparm(rslcpar, 'sar_to_earth_center'))
    re = float(readparm(rslcpar, 'earth_radius_below_sensor'))
    rgc = float(readparm(rslcpar, 'center_range_slc'))
    naz = int(readparm(rslcpar, 'azimuth_lines'))
    prf = float(readparm(rslcpar, 'prf'))

    mean_az = naz / 2 - 0.5  # mean azimuth line

    rg = rgn + ij[:, 2] * rps
    look = np.arccos((se ** 2 + rg.reshape(-1, 1) ** 2 - re ** 2) / (2 * se * rg.reshape(-1, 1)))  # Satellite look angles

    bperp_mat = np.zeros((n_ps, n_image))
    for i in range(n_ifg):
        basename = ifgs[i].replace(".diff", ".base")
        
        B_TCN = readparm(basename, 'initial_baseline(TCN):', 3, 0).astype("float")
        BR_TCN = readparm(basename, 'initial_baseline_rate:', 3, 0).astype("float")
        
        bc = B_TCN[1] + BR_TCN[1] * (ij[:, 1] - mean_az) / prf
        bn = B_TCN[2] + BR_TCN[2] * (ij[:, 1] - mean_az) / prf
        bperp_mat[:, i] = (bc.reshape(-1, 1) * np.cos(look) - bn.reshape(-1, 1) * np.sin(look)).flatten()
        # bpara=bc*sin(look)+bn*cos(look)

    bperp_mat = bperp_mat.astype("float32")
    bperp = np.mean(bperp_mat, axis = 0).reshape(-1, 1).astype("float32")
    if master_master_flag == 1:
        bperp_mat = np.concatenate((bperp_mat[:, 0:master_ix - 1], bperp_mat[:, master_ix:]), axis = 1)
    else:
        bperp = np.concatenate((bperp[0:master_ix - 1], np.array([[0]]), bperp[master_ix - 1:]), axis = 0).astype("float32")

    inci = np.arccos((se ** 2 - re ** 2 - rg ** 2) / (2 * re * rg))
    mean_incidence = np.mean(inci)
    mean_range = rgc

    bad_pix = []
    
    f = open(phname, 'rb')
    ph = np.zeros((n_ps, n_ifg)).astype("complex")
    byte_count = n_ps * 8
    for j in range(n_ifg):
        binary_data = f.read(byte_count)
        for i in range(n_ps):
            a = struct.unpack_from(">f", binary_data, offset = i * 8)
            b = struct.unpack_from(">f", binary_data, offset = 8 * i + 4)
            ph[i, j] = np.complex(a[0], b[0])
            if np.isnan(a[0]) or np.isnan(b[0]):
                bad_pix.append([i,j])
    f.close()
    
    print('* Loaded images:', n_ifg)
    print('* Loaded pixels per image:', n_ps)
    print('* Bad phases found:', len(bad_pix))
    
    if master_master_flag == 1:
        ph[:, master_ix - 1] = 1
    else:
        ph = np.concatenate((ph[:, 0: master_ix - 1], np.ones(n_ps).reshape(-1, 1), ph[:, master_ix - 1:]), axis=1)
        n_ifg = n_ifg + 1
        n_image = n_image + 1

    if os.path.exists(llname):
        print("* Process ->", llname)
        f = open(llname, 'rb')
        binary_data = f.read()
        lonlat = np.zeros((n_ps, 2)).astype("float")
        bad_ll = []
        
        for i in range(n_ps):
            lonlat[i,0] = struct.unpack_from(">f", binary_data, offset = 8 * i)[0]
            lonlat[i,1] = struct.unpack_from(">f", binary_data, offset = 8 * i + 4)[0]
            if lonlat[i,0] == 0 or lonlat[i,1] == 0 or np.isnan(lonlat[i,0]) or np.isnan(lonlat[i,1]):
                bad_ll.append(i)
        f.close()
    else:
        print('* File {} does not exist'.format(llname))
        sys.exit()
    
    print('* Bad coords found:', len(bad_ll))
    
    if len(bad_ll) > 0:
        for idx in bad_ll:
            lonlat[idx,0] = np.nanmin(lonlat[:,0])
            lonlat[idx,1] = np.nanmin(lonlat[:,1])
    
    # np.save('lonlat.npy', lonlat)
    
    ll0 = (np.amax(lonlat, axis = 0) + np.amin(lonlat, axis = 0)) / 2
    xy = llh2local(lonlat.T, ll0).T * 1000

    # np.save('xy.npy', xy)

    theta = (180 - heading) * np.pi / 180
    if theta > np.pi:
        theta = theta - 2 * np.pi

    rotm = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    xy = xy.T
    xynew = rotm.dot(xy)
    
    # np.save('xy1.npy', xy)
    
    # Check that rotation is an improvement
    if (np.max(xynew[0, :]) - np.min(xynew[0, :]) < np.max(xy[0, :]) - np.min(xy[0, :])) and (np.max(xynew[1, :]) - np.min(xynew[1, :]) < np.max(xy[1, :]) - np.min(xy[1, :])):
        xy = xynew  
        print("* Rotating by degrees:", str(theta * 180 / np.pi))
    
    # np.save('xy2.npy', xy)
    
    xy = xy.T.astype('float32')
    sort_ix = np.array(sorted(range(len(xy)), key = lambda s: (xy[s][1], xy[s][0])))
    xy = xy[sort_ix, :]
    xy = np.concatenate((np.array([*range(n_ps)]).reshape(-1, 1) + 1, xy), axis = 1)
    xy[:, 1:2] = np.round(xy[:, 1:2] * 1000) / 1000  # round to mm

    ph = ph[sort_ix, :]
    ij = ij[sort_ix, :]
    ij[:, 0] = np.array([*range(n_ps)]) + 1
    lonlat = lonlat[sort_ix, :]
    bperp_mat = bperp_mat[sort_ix, :]
    
    # Save all data to .mat files
    
    pssavename = path_to_patch + "ps" + psver + ".mat"

    ps_dict = {
        "ij": ij,
        "lonlat": lonlat,
        "xy": xy,
        "bperp": v2c(bperp),
        "day": v2c(np.array(day)),
        "master_day": master_day,
        "master_ix": master_ix,
        "n_ifg": n_ifg,
        "n_image": n_image,
        "n_ps": n_ps,
        "sort_ix": v2c(sort_ix) + 1,
        "ll0": v2r(ll0),
        "mean_incidence": mean_incidence,
        "mean_range": mean_range}
    
    savemat(pssavename, ps_dict)

    savemat("psver.mat", {"psver": psver})

    phsavename = path_to_patch + 'ph' + psver + ".mat"

    savemat(phsavename, {"ph": ph})

    bpsavename = path_to_patch + 'bp' + psver + ".mat"

    savemat(bpsavename, {"bperp_mat": bperp_mat})

    lasavename = path_to_patch + 'la' + psver + ".mat"
    la = inci[sort_ix] 
    
    savemat(lasavename, {"la": v2c(la)})

    # print('* Save D_A')
    if os.path.exists(daname):
        f = open(daname)
        lines = f.readlines()
        f.close()
        D_A = np.zeros((len(lines), 1))
        for i in range(len(lines)):
            D_A[i,0] = float(lines[i])
        D_A = D_A[sort_ix]
        dasavename = path_to_patch + 'da' + psver + ".mat"

        savemat(dasavename, {"D_A": v2c(D_A)})

    if os.path.exists(hgtname):
        f = open(hgtname, 'rb')
        binary_data = f.read()
        f.close()
        hgt = np.zeros(n_ps)
        for i in range(n_ps):
            hgt[i] = struct.unpack_from(">f", binary_data, offset = 4 * i)[0]
            
        hgt = hgt.reshape(-1, 1)
        hgt = hgt[sort_ix]
        hgtsavename = path_to_patch + 'hgt' + psver + ".mat"

        savemat(hgtsavename, {"hgt": v2c(hgt)})

    print('Done at', int(time.time() - begin_time), 'sec')
    
if __name__ == "__main__":
    # For testing
    test_path = 'C:\\Users\\Ryzen\\Documents\\PYTHON\\stampsexport'
    ps_load_initial_gamma(test_path, 1)      