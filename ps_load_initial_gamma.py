import os, sys
import numpy as np
from datetime import datetime
import math

from readparm import readparm
from setparm import setparm

from utils import compare_mat_with_number_values, compare_objects


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
        B_TCN = readparm(basename, 'initial_baseline(TCN):', 3)
        BR_TCN = readparm(basename, 'initial_baseline_rate:', 3)

    # diff = compare_objects(look, 'look')
    pass
