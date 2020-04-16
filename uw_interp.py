import numpy as np
import functools

from matplotlib.pyplot import triplot
from matplotlib.tri import *
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from triangle import triangulate
from scipy.io import loadmat, savemat
from pyhull.delaunay import DelaunayTri
from sklearn.neighbors import NearestNeighbors, KDTree, BallTree
from scipy.spatial import Delaunay
from scipy.spatial import distance
from scipy.spatial import tsearch

from collections import defaultdict
from itertools import permutations

from utils import compare_objects


def dsearchn(x, y):
    """
    Implement Octave / Matlab dsearchn without triangulation
    :param x: Search Points in
    :param y: Were points are stored
    :return: indices of points of x which have minimal distance to points of y
    """
    IDX = []
    for line in range(y.shape[0]):
        distances = np.sqrt(np.sum(np.power(x - y[line, :], 2), axis=1))
        found_min_dist_ind = (np.min(distances, axis=0) == distances)
        length = found_min_dist_ind.shape[0]
        IDX.append(np.array(range(length))[found_min_dist_ind][0])

    return np.array(IDX)


def find_neighbors(pindex, triang):
    return triang.vertex_neighbor_vertices[1][
           triang.vertex_neighbor_vertices[0][pindex]:triang.vertex_neighbor_vertices[0][pindex + 1]]


def uw_interp():
    print('Interpolating grid...\n')

    uw = loadmat('uw_grid.mat')
    n_ps = uw['n_ps'][0][0]
    n_ifg = uw['n_ifg'][0][0]
    nzix = uw['nzix']

    y_tmp = [np.nonzero(nzix[:, j]) for j in range(0, len(nzix[0]))]
    y = (functools.reduce(lambda a, b: np.concatenate((a, b), axis=1), y_tmp)).reshape(-1, 1)
    x_tmp = [np.tile(j, len(y_tmp[j][0])) for j in range(0, len(y_tmp))]
    x = functools.reduce(lambda a, b: np.concatenate((a, b), axis=0), x_tmp).reshape(-1, 1)
    xy = np.concatenate((np.array([i for i in range(0, n_ps)]).reshape(-1, 1), x, y), axis=1)

    ######
    from collections import defaultdict
    # tri = Delaunay(xy[:,1:])
    # _neighbors = defaultdict(set)
    # for simplex in tri.vertices:
    #    for i, j in permutations(simplex, 2):
    #        _neighbors[i].add(j)

    # points = [tuple(p) for p in tri.points]
    # neighbors = {}
    # for k, v in _neighbors.items():
    #    neighbors[points[k]] = [points[i] for i in v]
    #####

    tri = triangulate({'vertices': xy[:, 1:]}, opts='e')
    n_edge = len(tri['edges'])
    edgs = np.concatenate(
        (np.array([i for i in range(0, n_edge)]).reshape(-1, 1), tri['edges'], tri['edge_markers']), axis=1)
    n_ele = len(tri['triangles'])
    ele = np.concatenate(
        (np.array([i for i in range(0, n_ele)]).reshape(-1, 1), tri['triangles']), axis=1) + 1

    z = np.array([j for j in range(0, n_ps)])
    nrow, ncol = [len(nzix), len(nzix[0])]

    X, Y = np.meshgrid(np.array([j for j in range(0, ncol)]), np.array([j for j in range(0, nrow)]))
    XY = np.concatenate((X.flatten('F').reshape(-1, 1), Y.flatten('F').reshape(-1, 1)), axis=1)

    plt.figure()
    tri1 = Triangulation(xy[0:, 1:2].T[0], xy[0:, 2:3].T[0], triangles=tri['triangles'])
    ax = plt.subplot()
    # plt.plot([0,0,1],[5,4,5], '.', markersize=1.7, color='red')
    plt.plot([0, 0, 1], [6, 7, 6], '.', markersize=1.7, color='green')
    # neig = XY[np.where(np.array([distance.euclidean(np.array([2,6]), vertice) for vertice in tri['vertices']])==1)]
    # plt.plot(neig,markersize=1.2, color='green')
    ax.triplot(tri1, linewidth=0.2, color='blue', marker='.', markersize=1)
    # plt.savefig("graph.pdf", dpi=600)

    # for i in range(0, len(XY)):
    #    get_neighbors_idx(xy[:, 1:], XY[i])

    # nearest_neighbors = NearestNeighbors(n_neighbors=1, radius=1.5, p=10)
    # nearest_neighbors.fit(xy[:, 1:])
    # radius_neighbors = nearest_neighbors.radius_neighbors(XY, radius=1.5)
    # Z = nearest_neighbors.kneighbors(XY)

    # dela1 = Delaunay(xy[:, 1:], incremental=True)
    # dela1.add_points([XY[8]], restart=True)
    # dela2 = DelaunayTri(xy[:, 1:])
    # for i in range(0, 15):
    #    # neighbors = find_neighbors(i, tri)
    #    min_dist_idx = np.argmin([distance.euclidean(XY[i], point) for point in xy[:, 1:]])
    #    print(min_dist_idx + 1)

    tree = cKDTree(xy[:, 1:])
    dd, ii = tree.query(XY, k=5, p=2, eps=0.0)
    Z = []
    for i in range(len(dd)):
        min_dd = min(dd[i])
        min_dd_idx = np.where(dd[i] == min_dd)[0]
        if len(min_dd_idx) > 1:
            sorted_ii = np.sort(ii[i][min_dd_idx])
            Z.append(sorted_ii[len(min_dd_idx) - 1])
        else:
            Z.append(ii[i][0])

    Z = np.array(Z).reshape(-1, 1) + 1
    Z = np.reshape(Z, (nrow, ncol), 'F')

    Zvec = np.ndarray.flatten(Z, 'F').reshape(-1, 1)
    grid_edges = np.concatenate((Zvec[0:len(Zvec) - nrow, 0].reshape(-1, 1), Zvec[nrow:, 0].reshape(-1, 1)), axis=1)
    Zvec = np.ndarray.flatten(Z.reshape(-1, 1), 'F').reshape(-1, 1)
    grid_edges = np.concatenate((grid_edges, np.concatenate(
        (Zvec[0:len(Zvec) - ncol, 0].reshape(-1, 1), Zvec[ncol:, 0].reshape(-1, 1)), axis=1)), axis=0)
    sort_edges = np.sort(grid_edges, axis=1)
    I_sort = np.argsort(grid_edges)
    edge_sign = I_sort[:, 1] - I_sort[:, 0]
    edge_sign = edge_sign.reshape(-1, 1)
    alledges, I, J = np.unique(sort_edges, return_index=True, return_inverse=True, axis=0)
    I = I.reshape(-1, 1)
    J = J.reshape(-1, 1)
    sameix = np.array([alledges[:, 0] == alledges[:, 1]]).reshape(-1, 1)
    for i in range(len(sameix)):
        if sameix[i] == True:
            alledges[i] = 0

    edgs, I2, J2 = np.unique(alledges, return_index=True, return_inverse=True, axis=0)
    I2 = I2.reshape(-1, 1)
    J2 = J2.reshape(-1, 1)
    n_edge = edgs.shape[0] - 1
    edgs = np.concatenate((np.array([i for i in range(1, n_edge + 1)]).reshape(-1, 1), edgs[1:, :]), axis=1)
    gridedgeix = np.multiply(J2[np.ndarray.flatten(J)], edge_sign)
    colix = np.reshape(gridedgeix[0: nrow * (ncol - 1)], (nrow, ncol - 1), 'F')
    rowix = np.reshape(gridedgeix[nrow * (ncol - 1):], (ncol, nrow - 1), 'F').T

    print('   Number of unique edges in grid: {}\n'.format(n_edge))

    uw_interp = {
        'edgs': edgs,
        'n_edge': n_edge,
        'rowix': rowix,
        'colix': colix,
        'Z': Z
    }
    savemat('uw_interp.mat', uw_interp)
