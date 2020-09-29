import numpy as np


def readparm(*args):
    # fname,parm,numval,log_flag

    if len(args) < 3:
        numval = 1

    if len(args) < 4:
        log_flag = 1

    fname = args[0]
    parm = args[1]
    f = open(fname);
    parm_lines = f.readlines()
    f.close()

    for parm_line in parm_lines:
        if parm in parm_line:
            value = parm_line.split("\t")[1]
            if log_flag == 1:
                print(parm + "=" + value)
            break

    return value
