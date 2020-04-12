from scipy.io import loadmat, savemat


def uw_interp():
    print('Interpolating grid...\n')

    uw = loadmat('uw_grid.mat')

    use_triangle = 'y'

    print()
