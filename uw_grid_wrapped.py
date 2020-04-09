import numpy as np


def uw_grid_wrapped(*args):
    # params: ph_in, xy_in, pix_size, prefilt_win, goldfilt_flag, lowfilt_flag, gold_alpha, ph_in_predef

    print('Resampling phase to grid...\n')

    if len(args) < 2:
        print('not enough arguments')

    if len(args) < 3:
        pix_size = 200

    if len(args) < 4:
        prefilt_win = 32

    if len(args) < 5:
        goldfilt_flag = 'y'

    if len(args) < 6:
        lowfilt_flag = 'y'

    if len(args) < 7:
        gold_alpha = 0.8

    if len(args) < 8:
        ph_uw_predef = []

    ph_in_predef = args[7]
    if len(ph_in_predef) == 0:
        predef_flag = 'n'
    else:
        predef_flag = 'y'

    ph_in = args[0]
    n_ps = len(ph_in)
    n_ifg = len(ph_in[0])

    print('   Number of interferograms  : {}'.format(n_ifg))
    print('   Number of points per ifg  : {}'.format(n_ps))

    if np.any(np.isreal(ph_in)) == True and sum(sum(ph_in == 0)) > 0:
        print('Some phase values are zero')

    xy_in = args[1]
    xy_in[:, 0] = np.array([i+1 for i in range(0, n_ps)])

    pix_size = args[2]
    if pix_size==0:
        grid_x_min=1
        grid_y_min=1
        n_i=max(xy_in[:,2])
        n_j=max(xy_in[:,1])
        grid_ij=[xy_in(:,3),xy_in(:,2)]
    else:
        grid_x_min=min(xy_in(:,2));
        grid_y_min=min(xy_in(:,3));

        grid_ij(:,1)=ceil((xy_in(:,3)-grid_y_min+1e-3)/pix_size);
        grid_ij(grid_ij(:,1)==max(grid_ij(:,1)),1)=max(grid_ij(:,1))-1;
        grid_ij(:,2)=ceil((xy_in(:,2)-grid_x_min+1e-3)/pix_size);
        grid_ij(grid_ij(:,2)==max(grid_ij(:,2)),2)=max(grid_ij(:,2))-1;

        n_i=max(grid_ij(:,1));
        n_j=max(grid_ij(:,2));

    print()
