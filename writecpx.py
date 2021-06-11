import os

import numpy as np

from utils import compare_objects

def writecpx(*args):
    # params: fname, vname, precision, endian 
    
    if len(args) < 2:
        print('* Syntax is writecpx(FILE_NAME, VAR_NAME, PRECISION, ENDIAN)')
        os.exit(0)

    if len(args) < 3:
        precision = 'float32'

    if len(args) < 4:
        endian = 'n'

    fname = args[0]
    vname = args[1]

    if os.path.exists(fname):
        os.remove(fname)

    fid = open(fname, 'wb')
    vname_flt = np.zeros((len(vname), len(vname[0]) * 2))
    vname_flt[:, 0::2] = np.real(vname)
    vname_flt[:, 1::2] = np.imag(vname)
    vname_flt = vname_flt.astype(precision)
    vname_flt.tofile(fid)
    fid.close()
