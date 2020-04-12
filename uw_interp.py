import numpy as np
import functools

from scipy.spatial import cKDTree, KDTree
from triangle import triangulate
from scipy.io import loadmat, savemat
from scipy.spatial import Delaunay

from collections import defaultdict
from itertools import permutations

from utils import compare_objects

def find_neighbors(tess, points):

    neighbors = {}
    for point in range(points.shape[0]):
        neighbors[point] = []

    for simplex in tess.simplices:
        neighbors[simplex[0]] += [simplex[1],simplex[2]]
        neighbors[simplex[1]] += [simplex[2],simplex[0]]
        neighbors[simplex[2]] += [simplex[0],simplex[1]]

    return neighbors

def uw_interp():
    print('Interpolating grid...\n')

    uw = loadmat('uw_grid.mat')
    n_ps = uw['n_ps'][0][0]
    n_ifg = uw['n_ifg'][0][0]
    nzix = uw['nzix']

    y_tmp = [np.nonzero(nzix[:, j]) for j in range(0, len(nzix[0]))]
    y = (functools.reduce(lambda a, b: np.concatenate((a, b), axis=1), y_tmp) + 1).reshape(-1, 1)
    x_tmp = [np.tile(j + 1, len(y_tmp[j][0])) for j in range(0, len(y_tmp))]
    x = functools.reduce(lambda a, b: np.concatenate((a, b), axis=0), x_tmp).reshape(-1, 1)

    xy = np.concatenate((np.array([i + 1 for i in range(0, n_ps)]).reshape(-1, 1), x, y), axis=1)

    delo = Delaunay(xy[:, 1:])
    _neighbors = defaultdict(set)
    for simplex in delo.vertices:
        for i, j in permutations(simplex, 2):
            _neighbors[i].add(j)

    points = [tuple(p) for p in delo.points]
    neighbors = {}
    for k, v in _neighbors.items():
        neighbors[points[k]] = [points[i] for i in v]


    tri = triangulate({'vertices': xy[:, 1:]}, opts='e')

    n_edge = len(tri['edges'])
    edgs = np.concatenate(
        (np.array([i + 1 for i in range(0, n_edge)]).reshape(-1, 1), tri['edges'] + 1, tri['edge_markers']), axis=1)
    n_ele = len(tri['triangles'])
    ele = np.concatenate(
        (np.array([i + 1 for i in range(0, n_ele)]).reshape(-1, 1), tri['triangles'] + 1), axis=1)

    z = np.array([j + 1 for j in range(0, n_ps)])
    nrow, ncol = [len(nzix), len(nzix[0])]

    X, Y = np.meshgrid(np.array([j + 1 for j in range(0, ncol)]), np.array([j + 1 for j in range(0, nrow)]))
    XY = np.concatenate((X.flatten('F').reshape(-1, 1), Y.flatten('F').reshape(-1, 1)), axis=1)
    tree = cKDTree(xy[:, 1:])
    dd, ii = tree.query(XY)
    Z = ii + 1
    compare_objects(Z.reshape(-1,1),'Z')
    #Z = Z.reshape(nrow, ncol)

    print()
