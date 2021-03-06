import numpy as np


def create_laplacian_2d(nx, ny, lx, ly, pbc=True):
    """Creates a discretized Laplacian in 2D
        
        Arguments:
        nx (int): number of grid points along x; ny must be greater than one
        ny (int): number of grid points along y; nx must be greater than one
        lx (float): box length along x; must be positive
        ly (float): box length along y; must be positive
        pbc (boolean): use periodic boundary conditions
        """
    if type(nx) != int or type(ny) != int:
        raise TypeError('We need an integer')
    if type(lx) != int and type(lx) != float:
        raise TypeError('We need a number')
    if type(ly) != int and type(ly) != float:
        raise TypeError('We need a number')
    if nx < 2 or ny < 2:
        raise ValueError('We need at least two grid points in each direction')
    if nx / lx < 0 or ny / ly < 0:
        raise ValueError('We need positive lengths')
    if pbc not in (True, False):
        raise TypeError('We need a boolean as pbc')
    
    n = nx * ny  # total number of grid points
    laplacian = np.zeros((n, n))
    hx = (lx / nx) ** 2
    hy = (ly / ny) ** 2
    
    for i in range(n):
        laplacian[i, i] -= (2 / hx + 2 / hy)
    for j in range(ny - 1):
        for i in range(nx):
            laplacian[j * nx + i, (j + 1) * nx + i] += 1 / hy
            laplacian[(j + 1) * nx + i, j * nx + i] += 1 / hy
    for j in range(ny):
        for i in range(nx - 1):
            laplacian[j * nx + i, j * nx + i + 1] += 1 / hx
            laplacian[j * nx + i + 1, j * nx + i] += 1 / hx

    if pbc:
        for i in range(ny):
            laplacian[i, ny * nx - ny + i - 1] += 1 / hy
            laplacian[ny * nx - ny + i - 1, i] += 1 / hy
        for i in range(ny - 1):
            laplacian[i * nx, (i + 1) * nx - 1] += 1 / hx
            laplacian[(i + 1) * nx - 1, i * nx] += 1 / hx

    return laplacian
