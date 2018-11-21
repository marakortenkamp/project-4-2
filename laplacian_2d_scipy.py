import numpy as np
from scipy.sparse import diags

# I found a scipy method that does the trick

def create_laplacian_2d(k, L, pbc=True):
    """ Computes discrete Laplacian for a 2d quadratic
        charge density matrix, ordered row-wise
        Args:
            k: number of grid points along each axis, n > 2
            L: length of grid, L > 0
            pbc: periodic boundry conditions, boolean
        output:
            Laplacian as k**2 by k**2 np.array  """

    if k < 2:
        raise ValueError('We need at least two grid points')
    if L <= 0:
        raise ValueError('We need a positive length')
    if not type(pbc) == bool:
        raise TypeError('We need a boolean as pbc')

    h = (k / L)**2
    diagonals = [-4*np.ones(k**2) * h,
                 [0 if i%k == 0 else h for i in range(1, k**2)],
                 [0 if i%k == 0 else h for i in range(1, k**2)],
                 np.ones(k**2-k) * h,
                 np.ones(k**2-k) * h,
                 ]
    offsets   = [0, 1, -1, k, -k]
    if pbc:
        diagonals.extend([[np.ones(k) * h], [np.ones(k) * h],
                         [h if i%k == 0 else 0 for i in range(0,k**2 -k + 1)],
                         [h if i%k == 0 else 0 for i in range(0,k**2 -k + 1)],
                         ])
        offsets.extend([k - k**2, -k + k**2, k - 1, -k + 1])

    return np.array(diags(diagonals, offsets).todense())


print(create_laplacian_2d(4, 4, pbc=True))
