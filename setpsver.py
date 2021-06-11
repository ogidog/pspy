from scipy.io import loadmat, savemat

def setpsver(*args):
    
    psver = loadmat('psver.mat', squeeze_me = True)['psver']
    # print('psver currently:', psver)

    if len(args) > 0:
        new_psver = args[0]
        savemat('psver.mat', {'psver':new_psver})
        print('psver now set to:', new_psver)
