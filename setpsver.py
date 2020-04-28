from scipy.io import loadmat, savemat


def setpsver(*args):
    psver_mat = loadmat('psver.mat')
    psver = psver_mat['psver'][0][0]
    print('psver currently: {}'.format(str(psver)))

    if len(args) > 0:
        new_psver = args[0]
        psver_mat['psver'][0][0] = new_psver
        savemat('psver.mat', psver_mat)
        print('psver now set to: {}'.format(str(new_psver)))
