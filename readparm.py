import numpy as np

def readparm(*args): # fname,parm,numval,log_flag

    if len(args) < 3:
        numval = 1
    else:
        numval = args[2]

    if len(args) < 4:
        log_flag = 1
    else:
        log_flag = args[3]
        
    fname = args[0]
    parm = args[1]
    f = open(fname)
    parm_lines = f.readlines()
    f.close()

    for parm_line in parm_lines:
        if parm in parm_line:
            if numval == 1:
                value = parm_line.split("\t")[1]
                value = value.strip()
                if log_flag == 1:
                    print(parm + "=" + value)
            if numval > 1:
                value = parm_line.split("\t")[1:numval + 1]
                value = np.array(list(map(str.strip, value)))
                if log_flag == 1:
                    print(parm + "=" + " ".join(value))
            break

    return(value)
