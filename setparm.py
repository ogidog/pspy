import os, sys
from scipy.io import loadmat, savemat

def setparm(name, value):

    parmfile = "./parms.mat"

    if not os.path.exists(parmfile):

        parmfile = "../parms.mat"
        if not os.path.exists(parmfile):

            print("Parameters file does not exist")
            sys.exit()

    parms = loadmat(parmfile)
    parms[name] = value
    savemat(parmfile, parms)
