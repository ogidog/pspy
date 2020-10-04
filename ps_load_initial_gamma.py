import os, sys
import numpy as np
from datetime import datetime
import struct

from llh2local import llh2local
from readparm import readparm
from setparm import setparm

from utils import compare_mat_with_number_values, compare_objects, compare_complex_objects


def ps_load_initial_gamma(*args):
    if len(args) < 1:
        endian = 'b'

    print('Loading data...')

    phname = './pscands.1.ph'
    ijname = './pscands.1.ij'
    llname = './pscands.1.ll'
    xyname = './pscands.1.xy'
    hgtname = './pscands.1.hgt'
    daname = './pscands.1.da'
    rscname = '../rsc.txt'
    pscname = '../pscphase.in'

    psver = 1;
    if not os.path.exists(rscname):
        print("File {} not exist.".format(rscname))
        sys.exit()
    f = open(rscname)
    rslcpar = "../rslc/" + os.path.split(f.read().strip())[1]
    f.close()

    if not os.path.exists(pscname):
        print("File {} not exist.".format(pscname))
        sys.exit()
    f = open(pscname)
    ifgs = f.read()
    f.close()
    ifgs = ifgs.split("\n")[1:-1]
    master_day = datetime.strptime(os.path.split(ifgs[0])[1].split("_")[0], "%Y%m%d").date().toordinal() + 366
    ifgs = ["../diff0/" + os.path.split(ifg)[1] for ifg in ifgs]
    n_ifg = len(ifgs)
    n_image = n_ifg

    day = [(datetime.strptime(str(os.path.split(ifg)[1].split("_")[1].replace(".diff", "")),
                              "%Y%m%d")).date().toordinal() + 366
           for ifg in ifgs]

    master_ix = len(np.where(np.array(day) < master_day)[0]) + 1
    if day[master_ix] != master_day:
        master_master_flag = '0'  # no null master-master ifg provided
        day.insert(master_ix - 1, master_day)
    else:
        master_master_flag = '1'  # yes, null master-master ifg provided

    heading = float(readparm(rslcpar, 'heading:'))
    setparm("heading", heading)

    freq = float(readparm(rslcpar, 'radar_frequency:'))
    lambda1 = 299792458 / freq
    setparm('lambda', lambda1)

    sensor = readparm(rslcpar, 'sensor:')
    if 'ASAR' in sensor:
        platform = 'ENVISAT'
    else:
        platform = sensor  # S1A for Sentinel-1A
    setparm('platform', platform)

    f = open(ijname)
    lines = f.readlines()
    f.close()
    ij = np.empty((0, 3), int)
    for line in lines:
        line = line.strip()
        ij = np.append(ij, np.array([line.split(" ")]).astype("int"), axis=0)
    n_ps = np.size(ij, 0)

    rps = int(readparm(rslcpar, 'range_pixel_spacing'))
    rgn = float(readparm(rslcpar, 'near_range_slc'))
    se = float(readparm(rslcpar, 'sar_to_earth_center'))
    re = float(readparm(rslcpar, 'earth_radius_below_sensor'))
    rgc = float(readparm(rslcpar, 'center_range_slc'))
    naz = int(readparm(rslcpar, 'azimuth_lines'))
    prf = float(readparm(rslcpar, 'prf'))

    mean_az = naz / 2 - 0.5  # mean azimuth line

    rg = rgn + ij[:, 2] * rps
    look = np.arccos(
        (se ** 2 + rg.reshape(-1, 1) ** 2 - re ** 2) / (2 * se * rg.reshape(-1, 1)))  # Satellite look angles

    bperp_mat = np.zeros((n_ps, n_image))
    for i in range(n_ifg):
        basename = ifgs[i].replace(".diff", ".base")
        B_TCN = readparm(basename, 'initial_baseline(TCN):', 3).astype("float")
        BR_TCN = readparm(basename, 'initial_baseline_rate:', 3).astype("float")
        bc = B_TCN[1] + BR_TCN[1] * (ij[:, 1] - mean_az) / prf
        bn = B_TCN[2] + BR_TCN[2] * (ij[:, 1] - mean_az) / prf
        bperp_mat[:, i] = (bc.reshape(-1, 1) * np.cos(look) - bn.reshape(-1, 1) * np.sin(look)).flatten()
        # bpara=bc*sin(look)+bn*cos(look)

    bperp = np.mean(bperp_mat, axis=0).reshape(-1, 1)
    if master_master_flag == 1:
        bperp_mat = np.concatenate((bperp_mat[:, 0:master_ix - 1], bperp_mat[:, master_ix:]), axis=1)
    else:
        bperp = np.concatenate((bperp[0:master_ix - 1], np.array([[0]]), bperp[master_ix - 1:]), axis=0)
    # bperp=[bperp(1:master_ix-1);0;bperp(master_ix:end)]; % insert master-master bperp (zero)
    # bperp_mat=repmat(single(bperp([1:master_ix-1,master_ix+1:end]))',n_ps,1);

    inci = np.arccos((se ** 2 - re ** 2 - rg ** 2) / (2 * re * rg))
    mean_incidence = np.mean(inci)
    mean_range = rgc

    f = open(phname, 'rb')
    ph = np.zeros((n_ps, n_ifg)).astype("complex")
    byte_count = n_ps * 8
    for j in range(n_ifg):
        binary_data = f.read(byte_count)
        for i in range(n_ps):
            a = struct.unpack_from(">f", binary_data, offset=i * 8)
            b = struct.unpack_from(">f", binary_data, offset=8 * i + 4)
            ph[i, j] = a[0] + 1j * b[0]

    f.close()

    zero_ph = np.sum(ph == 0, axis=1).reshape(-1, 1)
    nonzero_ix = zero_ph <= 1  # if more than 1 phase is zero, drop node
    if master_master_flag == 1:
        ph[:, master_ix - 1] = 1
    else:
        ph = np.concatenate((ph[:, 0: master_ix - 1], np.ones(n_ps).reshape(-1, 1), ph[:, master_ix - 1:]), axis=1)
        n_ifg = n_ifg + 1
        n_image = n_image + 1

    if os.path.exists(llname):
        f = open(llname, 'rb')
        binary_data = f.read()
        lonlat = np.zeros((n_ps, 2)).astype("float")
        for i in range(n_ps):
            lonlat[i, 0] = struct.unpack_from(">f", binary_data, offset=8 * i)[0]
            lonlat[i, 1] = struct.unpack_from(">f", binary_data, offset=8 * i + 4)[0]
        f.close()
    else:
        print('File {} does not exist'.format(llname))
        sys.exit()

    ll0 = (np.amax(lonlat, axis=0) + np.amin(lonlat, axis=0)) / 2
    xy = llh2local(lonlat.T, ll0).T * 1000

    sort_x = xy[np.argsort(xy[:, 0], kind="stable")]
    sort_y = xy[np.argsort(xy[:, 1], kind="stable")]
    n_pc = int(np.round(n_ps * 0.001))
    bl = np.mean(sort_x[0:n_pc, :], axis=0)  # bottom left corner
    tr = np.mean(sort_x[-(n_pc + 1):, :], axis=0)  # top right corner
    br = np.mean(sort_y[0:n_pc, :], axis=0)  # bottom right  corner
    tl = np.mean(sort_y[-(n_pc + 1):, :], axis=0)  # top left corner

    theta = (180 - heading) * np.pi / 180
    if theta > np.pi:
        theta = theta - 2 * np.pi

    rotm = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    xy = xy.T
    xynew = rotm.dot(xy)  # rotate so that scene axes approx align with x=0 and y=0
    if (np.max(xynew[0, :]) - np.min(xynew[0, :]) < np.max(xy[0, :]) - np.min(xy[0, :])) and (
            np.max(xynew[1, :]) - np.min(xynew[1, :]) < np.max(xy[1, :]) - np.min(xy[1, :])):
        xy = xynew  # check that rotation is an improvement
        print("Rotating by {}  degrees", str(theta * 180 / np.pi))

    xy = xy.T

    #sort_ix = np.array(sorted(range(len(xy)), key=lambda s: (xy[s][1], xy[s][0])))
    #sort_ix_1= np.argsort(xy[:, 1], kind="stable")  # sort in ascending y order
    compare_objects(xy,"xy")

    #diff = compare_objects(sort_ix.reshape(-1,1)+1, 'sort_ix')
    # diff = compare_complex_objects(ph, 'ph')
    pass
