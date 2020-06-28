""" Matrix Factorization for Spectrum Imaging Data Analysis
"""
# Author: Motoki Shiga, Gifu University <shiga_m@gifu-u.ac.jp>
# License: MIT
#

import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd

class RandomMF(object):
    """Random Matrix Factorization
        Fractorize a data matrix into two low rank matrices C_ and S_ using random numbers.
    
    Parameters
    ----------
    n_components : int
        The number of components decomposed from a data matrix
    random_seed : int, optional (default = 0)
        Random number generator seed control

    Attributes
    ----------
    random_seed : int, default 0
        Random number generator seed to control
    C_ : ndarray of shape = (# of spatial data points, n_components)
        Spatial intensity distributions of factorized components
    S_ : ndarray of shape = (# of spectrum channels, n_components)
        Factorized component spectra
    E_ : ndarray of shape = (# of spatial data points in the 1st axis, # of those in 2nd axis)
        Residual spatial image (spatial image of RSME)
    """

    # constructor
    def __init__(self, n_components, random_seed=0):
        self.n_components = n_components
        self.random_seed = random_seed

    def __repr__(self):
        class_name = self.__class__.__name__
        txt = 'n_components=' + str(self.n_components) +\
              ', random_seed=' + str(self.random_seed)
        return '%s(%s)' % (class_name, txt,)

    def __str__(self):
        txt = self.__repr__()
        return txt

    def fit(self, X, channel_vals=None, unit_name=None):
        """Generate two low rank matrices by random number

        Parameters
        ----------
        X: ndarray of shape = (# of spatial data points in the 1st axis, # of those in 2nd axis, # of spectrum channels)
            Data matrix to be decomposed
        channel_vals: ndarray of shape = (# of spectrum channels), optional (default = None)
            The sequence of channel values
        unit_name: string, optional (default = None)
            The unit name of spectrum channel

        Returns
        -------
        self: instance of class RandomMF
        """

        # initialize attributes from the given spectrum imaging data
        if X.ndim == 2:
            self.num_y = 1
            self.num_x, self.num_ch = X.shape
            self.num_xy = self.num_x * self.num_y
        else:
            self.num_x, self.num_y, self.num_ch = X.shape
            self.num_xy = self.num_x * self.num_y
            X = X.reshape(self.num_xy, self.num_ch) # transform from 3D-array to 2D-array (Data Matrix)

        # set channel information
        if channel_vals is None:
            self.channel_vals = np.arange(self.num_ch)
        else:
            self.channel_vals = channel_vals
        if unit_name is None:
            self.unit_name = 'Channel'
        else:
            self.unit_name = unit_name

        # randomly pick up spectra from the data matrix as component spectra
        indices = np.random.randint(self.num_xy, size=self.n_components)
        self.S_ = X[indices, :].T

        # optimize maxtix C by minimizing the reconstruction error (MSE: Mean Squared Error)
        self.C_ = X @ np.linalg.pinv(self.S_).T

        # residual spatial image (spatial image of RSME)
        self.E_ = np.sqrt( np.mean((X - self.C_@self.S_.T)**2, axis=1) )
        self.E_ = self.E_.reshape(self.num_x, self.num_y)

        return self

    def plot_intensity(self, figsize=None, filename=None, xlabel=None):
        """Plot component intensities

        Parameters
        ----------
        figsize: list of shape = (the size of horizontal axis, that of vertical axis), optional (default = None)
            Size of horizontal axis and vertical axis of figure
        filename: string, optional (default = None)
            The name of an output image data
            If None, the image is not saved to a file. 
        xlabel: string, optional (default = None)
            The name of x-axis
        """

        if figsize is None:
            plt.figure()
        else:
            plt.figure(figsize=figsize)
        
        if xlabel is None:
            xlabel = 'Measured data point'
        
        for k in range(self.C_.shape[1]):
            plt.plot(self.C_[:, k], label=str(k + 1))

        plt.xlim([0, self.C_.shape[0]])
        plt.xlabel(xlabel)
        plt.ylabel('Intensity')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.close()

    def plot_component(self, figsize=list(), filename=None):
        '''
        Plot component intensities (data points vs intensities)
        Parameters
        ----------
        figsize: the vertical and horizontal size of the figure
        '''
        if len(figsize) == 0:
            plt.figure()
        else:
            plt.figure(figsize=figsize)
        for k in range(self.C_.shape[1]):
            plt.plot(self.C_[:, k], label=str(k + 1))
        plt.xlim([0, self.C_.shape[0]])
        plt.xlabel('Spatial data point')
        plt.ylabel('Intensity')
        plt.title('Components')
        plt.legend()
        if filename is not None:
            plt.savefig(filename)
        plt.show()

    def imshow_component(self, figsize=None, figshape=None, filename=None):
        """Display component spatial intensity distributions

        Parameters
        ----------
        figsize: list of shape = (the size of horizontal axis, that of vertical axis), optional (default = None)
            Size of horizontal axis and vertical axis of figure
        figshape: list of shape = (# of rows, # columns), optional (default = None)
            The number of rows and columns of axes
        filename: string, optional (default = None)
            The name of an output image data
            If None, the image is not saved to a file. 
        """

        if self.num_y == 1:
            self.plot_component(figsize=figsize, filename=filename)
        else:
            if figsize is None:
                plt.figure()
            else:
                plt.figure(figsize=figsize)

            # adjust figure layout
            if figshape is None:
                if self.n_components<=3:
                    ncols = self.n_components
                    nrows = 1
                else:
                    ncols = int(np.ceil(np.sqrt(self.n_components)))
                    if ncols*(ncols-1) >= self.n_components:
                        nrows = ncols-1
                    elif ncols**2 >= self.n_components:
                        nrows = ncols
                    else:
                        nrows = ncols+1
            else:
                nrows, ncols = figshape[0], figshape[1]
                if nrows*ncols < self.n_components:
                    print('Error: nrows x ncols should be larger than n_components!')
                    return -1

            # display figures
            for k in range(self.C_.shape[1]):
                plt.subplot(nrows, ncols, k + 1)
                im = np.reshape(self.C_[:, k], (self.num_x, self.num_y))
                plt.imshow(im)
                plt.title(str(k+1))
            plt.tight_layout()
            if filename is None:
                plt.show()
            else:
                plt.savefig(filename, bbox_inches='tight', pad_inches=0)
                plt.close()

    def imshow_residual_image(self, figsize=None, filename=None):
        """Display residual (RMSE or beta-divergence) image averaged over all channels

        Parameters
        ----------
        figsize: list of shape = (the size of horizontal axis, that of vertical axis), optional (default = None)
            Size of horizontal axis and vertical axis of figure
        filename: string, optional (default = None)
            The name of an output image data
            If None, the image is not saved to a file. 
        """

        if figsize is None:
            plt.figure()
        else:
            plt.figure(figsize=figsize)
        plt.imshow(self.E_)
        plt.tight_layout()
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.close()

    def plot_spectra(self, figsize=None, filename=None, normalize=True):
        """Plot component spectra

        Parameters
        ----------
        figsize: list of shape = (the size of horizontal axis, that of vertical axis), optional (default = None)
            Size of horizontal axis and vertical axis of figure
        filename: string, optional (default = None)
            The file name of an output image
        normalize: bool, optional (default = True)
            If True, each spectrum is normalized
        """

        if figsize is None:
            plt.figure()
        else:
            plt.figure(figsize=figsize)
        for k in range(self.S_.shape[1]):
            if normalize:
                Sk = self.S_[:, k] / (np.sqrt(np.sum(self.S_[:, k]**2)) + 1e-16)
            else:
                Sk = self.S_[:, k]
            plt.plot(self.channel_vals, Sk, label=str(k + 1))
        plt.xlabel(self.unit_name)
        plt.ylabel('Intensity')
        plt.xlim([self.channel_vals[0], self.channel_vals[-1]])
        plt.legend()
        plt.grid()
        plt.tight_layout()
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.close()

class SVD(RandomMF):
    """Singular Value Decomposition (SVD)

    Matrix factorization of spectrum image data by SVD

    Parameters
    ----------
    n_components : int
        The number of components decomposed from a data matrix

    Attributes
    ----------
    C_ : ndarray of shape = (# of spatial data points, n_components)
        Spatial intensity distributions of factorized components
    S_ : ndarray of shape = (# of spectrum channels, n_components)
        Factorized component spectra
    E_ : ndarray of shape = (# of spatial data points in the 1st axis, # of those in 2nd axis)
        Residual spatial image (spatial image of RSME)

    """

    # constructor
    def __init__(self, n_components):
        super(SVD, self).__init__(n_components = n_components)

    def __repr__(self):
        class_name = self.__class__.__name__
        txt = 'n_components=' + str(self.n_components)
        return '%s(%s)' % (class_name, txt,)

    def __str__(self):
        txt = self.__repr__()
        return txt

    def fit(self, X, channel_vals=None, unit_name=None):
        """Decompose a matrix into two Low-rank matrix by SVD

        Two low rank matrices are generated based on SVD.

        Parameters
        ----------
        X: ndarray of shape = (# of spatial data points in the 1st axis, # of those in 2nd axis, # of spectrum channels)
            Data matrix to be decomposed
        channel_vals: ndarray of shape = (# of spectrum channels), optional (default = None)
            The sequence of channel values
        unit_name: string, optional (default = None)
            The unit name of spectrum channel

        Returns
        -------
        self: instance of class SVD
        """

        # initialize attributes from the given spectrum imaging data
        if X.ndim == 2:
            self.num_y = 1
            self.num_x, self.num_ch = X.shape
            self.num_xy = self.num_x * self.num_y
        else:
            self.num_x, self.num_y, self.num_ch = X.shape
            self.num_xy = self.num_x * self.num_y
            X = X.reshape(self.num_xy, self.num_ch) # transform from 3D-array to 2D-array (Data Matrix)

        if channel_vals is None:
            self.channel_vals = np.arange(self.num_ch)
        else:
            self.channel_vals = channel_vals
        if unit_name is None:
            self.unit_name = 'Channel'
        else:
            self.unit_name = unit_name

        print('Training SVD...')

        # SVD and extract only components of the largest singular values
        d = self.n_components
        Ud, Sd, Vd = scipy.linalg.svd(X)  # computes the d-projection matrix
        Ud, Sd, Vd = Ud[:, :d], np.diag(Sd[:d]), Vd[:d, :]  # choose d-dimension
        self.C_ = Ud
        self.S_ = (Sd@Vd).T

        # adjust the sign of spectra
        for k in range(self.n_components):
            i = np.argmax(np.abs(self.S_[:, k]))
            if self.S_[i, k] < 0:
                self.S_[:, k] = -self.S_[:, k]
                self.C_[:, k] = -self.C_[:, k]

        # residual spatial image (spatial image of RSME)
        self.E_ = np.sqrt( np.mean((X - self.C_@self.S_.T)**2, axis=1) )
        self.E_ = self.E_.reshape(self.num_x, self.num_y)

        return self

class PCA(RandomMF):
    """Principal Components Analysis (PCA)

    Parameters
    ----------
    n_components : int
        The number of components decomposed from a data matrix

    Attributes
    ----------
    C_ : ndarray of shape = (# of spatial data points, n_components)
        Spatial intensity distributions of factorized components
    S_ : ndarray of shape = (# of spectrum channels, n_components)
        Factorized component spectra
    E_ : ndarray of shape = (# of spatial data points in the 1st axis, # of those in 2nd axis)
        Residual spatial image (spatial image of RSME)
    """

    def __init__(self, n_components):
        super(PCA, self).__init__(n_components=n_components)

    def __repr__(self):
        class_name = self.__class__.__name__
        txt = 'n_components=' + str(self.n_components)
        return '%s(%s)' % (class_name, txt,)

    def __str__(self):
        txt = self.__repr__()
        return txt

    def fit(self, X, channel_vals=None, unit_name=None):
        """
        Decompose into two Low-rank matrix by PCA

        Parameters
        ----------
        X: ndarray of shape = (# of spatial data points in the 1st axis, # of those in 2nd axis, # of spectrum channels)
            Data matrix to be decomposed
        channel_vals: ndarray of shape = (# of spectrum channels), optional (default = None)
            The sequence of channel values
        unit_name: string, optional (default = None)
            The unit name of spectrum channel

        Returns
        -------
        self: instance of class PCA
        """

        # initialize attributes from the given spectrum imaging data
        if X.ndim == 2:
            self.num_y = 1
            self.num_x, self.num_ch = X.shape
            self.num_xy = self.num_x * self.num_y
        else:
            self.num_x, self.num_y, self.num_ch = X.shape
            self.num_xy = self.num_x * self.num_y
            X = X.reshape(self.num_xy, self.num_ch) # transform from 3D-array to 2D-array (Data Matrix)

        if channel_vals is None:
            self.channel_vals = np.arange(self.num_ch)
        else:
            self.channel_vals = channel_vals
        if unit_name is None:
            self.unit_name = 'Channel'
        else:
            self.unit_name = unit_name

        self.X_mean = np.mean(X, 0, keepdims=True)
        X = X - self.X_mean

        print('Training PCA...')

        # Compute S and C via Variance-Covariance matrix
        d = self.n_components
        CovMat = X.T @ X / self.num_xy
        lam, self.S_ = scipy.linalg.eigh(CovMat,eigvals=(self.num_ch-d,self.num_ch-1))
        lam, self.S_ = lam[::-1], self.S_[:,::-1]
        for i in range(d):
            if np.max(self.S_[:,i]) < np.abs(np.min(self.S_[:,i])):
                self.S_[:, i] = -self.S_[:,i]
        self.C_ = X@self.S_

        # # SVD and extract only components of the largest singular values
        # d = self.n_components
        # Ud, Sd, Vd = scipy.linalg.svd(X)  # computes the d-projection matrix
        # Ud, Sd, Vd = Ud[:, :d], np.diag(Sd[:d]), Vd[:d, :]  # choose d-dimension
        # self.C_ = Ud
        # S = (Sd@Vd + self.X_mean).T
        #
        # # adjust the sign of spectra
        # for k in range(self.n_components):
        #     i = np.argmax(np.abs(S[:, k]))
        #     if S[i, k] < 0:
        #         Sd[k, k] = -Sd[k, k]
        #         self.C_[:, k] = -self.C_[:, k]
        # self.S_ = (Sd @ Vd).T

        # residual spatial image (spatial image of RSME)
        self.E_ = np.sqrt( np.mean((X - self.C_@self.S_.T)**2, axis=1) )
        self.E_ = self.E_.reshape(self.num_x, self.num_y)

        return self
