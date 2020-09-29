import os, sys


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
    rslcpar = f.read().strip()
    f.close()

    if not os.path.exists(pscname):
        print("File {} not exist.".format(pscname))
        sys.exit()
    f=open(pscname)
    ifgs = f.read()
    f.close()
    #ifgs=ifgs{1}(2:end);

    print()
