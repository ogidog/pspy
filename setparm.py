import os


def setparm(name, value):
    parmfile = "./parms.mat"

    if not os.path.exists(parmfile):
        parmfile = "../parms.mat"


