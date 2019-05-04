import numpy as np
import matplotlib.pyplot as plt

def fit_pre_edge(X, energy, range_energy=None):
    """Pre-Edge fitting by least squares with a exponential function
        This function detects pre-edges of each spatial point individually.

    Parameters
    ----------
    X : array_like of shape (# of x-axis point, # of x-axis point, # of channels)
        Spectral Image,
    energy: array_like of shape (# of channels)
        Loss energy values
    range_energy: sequence of two float values or None
        The minimum, maximum energy values 
        If None,  all range of spectrum is used to fit

    Returns
    ----------
    pre_edge : array_like of shape (# of x-axis point, # of x-axis point, # of channels)
        Pre_edges fitted from input X
    """

    eps = 10**(-16)

    if range_energy is None:
        print('Use all spectrum to fit pre-edge')
        range_energy = [energy[0], energy[-1]]

    num_x, num_y, num_ch = X.shape
    X = X.reshape(num_x * num_y, num_ch)

    i = (range_energy[0] <= energy) & (energy <= range_energy[1])
    energy_clipped = energy[i]
    X_clipped = X[:, i]

    N = energy_clipped.size
    log_ene = np.log(energy_clipped)
    log_ene = log_ene.reshape(N, 1)
    log_X = np.log(X_clipped)


    b = (log_X@log_ene*N - log_ene.sum()*log_X.sum(axis=1, keepdims=True))/(N*np.sum(log_ene**2) - (log_ene.sum())**2 + eps)
    a = np.mean(log_X, axis=1, keepdims=True) - b*np.mean(log_ene)

    A, r = np.exp(a), -b

    energy = energy.reshape(1, energy.size)
    pre_edge = A*energy**(-r)
    pre_edge = pre_edge.reshape(num_x, num_y, num_ch)

    return pre_edge
