import numpy as np

def create_laplacian_2d(nx, ny, lx, ly, pbc=True):
    """Creates a discretized Laplacian in 2D
        
        Arguments:
        nx (int): number of grid points along x
        ny (int): number of grid points along y; nx*ny must be greater than one
        lx (float): box length along x; must be positive
        ly (float): box length along y; must be positive and nx/lx == ny/ly (regular grid)
        pbc (boolean): use periodic boundary conditions
    """
    if nx*ny < 2:
        raise ValueError('We need at least two grid points')
    if nx/lx != ny/ly or nx/lx < 0:
        raise ValueError('We need positive lengths and the grid should be regular')
    if pbc not in (True, False):
        raise TypeError('We need a boolean as pbc')

    n = nx*ny # total number of grid points
    laplacian = np.zeros((n, n))

    h = (nx/lx)**2

    for i in range(n):
        laplacian[i, i] = -4

    for j in range(nx-1):
        for i in range(ny):
            laplacian[j*nx+i, (j+1)*nx+i] = 1
            laplacian[(j+1)*nx+i, j*nx+i] = 1

    for j in range(nx):
        for i in range(ny-1):
            laplacian[j*nx+i,j*nx+i+1] = 1
            laplacian[j*nx+i+1, j*nx+i] = 1

    if pbc:
        for i in range(ny):
            laplacian[i, ny*nx-ny+i] = 1
            laplacian[ny*nx-ny+i, i] = 1
            laplacian[i*nx, (i+1)*nx-1] = 1
            laplacian[(i+1)*nx-1, i*nx] = 1

    laplacian = h*laplacian

    return laplacian