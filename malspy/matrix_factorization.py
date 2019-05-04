""" Matrix Factorization for Spectrum Imaging Data Analysis
"""
# Author: Motoki Shiga <shiga_m@gifu-u.ac.jp>
# License: MIT
#


import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd


class RandomMF(object):
    """Random Matrix Factorization

    Parameters
    ----------
    n_components : int 
        The number of components
    random_seed : int, default 0
        Random number generator seed control

    Attributes
    ----------
    C_ : array_like of shape (# of spatial data points, n_components)
        Spatial intensity distribution of components decomposed from data matrix X
    S_ : array_like of shape (# of spectrum channels, n_components)
        Spectra decomposed from data matrix X
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
        """
        Decompose into two Low-rank matrix by sampling spectra at random

        Parameters
        ----------
        X: array_like of shape (# of spatial data points in the 1st axis, # of those in 2nd axis, # of spectrum channels)
            Data matrix to be decomposed
        channel_vals: array_like of shape (# of spectrum channels)
            The sequence of channel values
        unit_name: string
            The unit name of spectrum channel

        Returns
        -------
        self: instance of class RandomMF
        """

        # --- Attribute initialization from a data matrix------
        if X.ndim == 2:
            self.num_y = 1
            self.num_x, self.num_ch = X.shape
            self.num_xy = self.num_x * self.num_y
        else:
            self.num_x, self.num_y, self.num_ch = X.shape
            self.num_xy = self.num_x * self.num_y
            # transform from 3D-array to 2D-array (Data Matrix)
            X = X.reshape(self.num_xy, self.num_ch)

        # set channel information
        if channel_vals is None:
            self.channel_vals = np.arange(self.num_ch)
        else:
            self.channel_vals = channel_vals
        if unit_name is None:
            self.unit_name = 'Channel'
        else:
            self.unit_name = unit_name

        # randomly pick up spectra as component spectra
        indices = np.random.randint(self.num_xy, size=self.n_components)
        self.S_ = X[indices, :].T

        # optimize maxtix C by minimizing the reconstruction error (MSE: Mean Squared Error)
        self.C_ = X @ np.linalg.pinv(self.S_).T

        return self

    def plot_intensity(self, figsize=None, filename=None):
        """Plot component intensities

        Parameters
        ----------
        figsize: sequence of two ints
            figure length of vertical and horizontal axis
        filename: string
            file name of an output image
        """

        if figsize is None:
            plt.figure()
        else:
            plt.figure(figsize=figsize)
        for k in range(self.C_.shape[1]):
            plt.plot(self.C_[:, k], label=str(k + 1))
        plt.xlim([self.channel_vals[0], self.channel_vals[-1]])
        plt.xlabel(self.unit_name)
        plt.ylabel('Intensity')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)

    def imshow_component(self, figsize=None, figshape=None, filename=None):
        """Show component spatial intensity distributions

        Parameters
        ----------
        figsize: sequence of two ints
            the vertical and horizontal size of the figure
        figshape: sequence of two ints
            the number of rows and columns of axes
        filename: string
            file name of an output image
        """

        if self.num_y == 1:
            self.plot_component(figsize=figsize, filename=filename)
        else:
            if figsize is None:
                plt.figure()
            else:
                plt.figure(figsize=figsize)

            if figshape is None:
                nrows, ncols = self.C_.shape[1], 1
            else:
                nrows, ncols = figshape[0], figshape[1]
                if nrows*ncols < self.n_components:
                    print('Error: nrows x ncols should be larger than n_components!')
                    return -1

            for k in range(self.C_.shape[1]):
                plt.subplot(nrows, ncols, k + 1)
                im = np.reshape(self.C_[:, k], (self.num_x, self.num_y))
                plt.imshow(im)
                plt.title(str(k+1))
            plt.tight_layout()
            if filename is None:
                plt.show()
            else:
                plt.savefig(filename)

    def imshow_error(self, X, figsize=None, filename=None):
        """Show residual distribution averaged over all channels

        Parameters
        ----------
        X: three-dimension array
            data cube
        figsize: sequence of two ints
            the vertical and horizontal size of the figure
        filename: string
            file name of an output image
        """

        if X.ndim!=3:
            print('Error: X should be a three-dimension array')
            return -1

        self.num_x, self.num_y, self.num_ch = X.shape
        self.num_xy = self.num_x * self.num_y
        # transform from 3D-array to 2D-array (Data Matrix)
        X = X.reshape(self.num_xy, self.num_ch)

        # compute mean squared error over channel axis
        X_hat = self.C_ @ self.S_.T
        E = X_hat - X
        im = (E**2).mean(axis=1)
        im = np.reshape(im, (self.num_x, self.num_y)) # transform to 2D-array

        if figsize is None:
            plt.figure()
        else:
            plt.figure(figsize=figsize)
        plt.imshow(im)
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)

    def plot_spectra(self, figsize=None, filename=None, normalize=True):
        """Plot component spectra

        Parameters
        ----------
        figsize: sequence of two ints
            the vertical and horizontal size of the figure
        filename: string
            file name of an output image
        normalize: Boolean, default True
            If true, each spectrum is normalized
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
            plt.savefig(filename)

    def save_spectra_to_csv(self, filename, normalize=True):
        """Save numerical values of component spectra to a csv file

        Parameters
        ----------
        filename: string
            file name of an output image
        normalize: Boolean, default True
            If true, each spectrum is normalized
        """

        Sk = self.S_.copy()
        if normalize:
            for k in range(Sk.shape[1]):
                Sk[:, k] = Sk[:, k] / (np.sqrt(np.sum(Sk[:, k]**2)) + 1e-16)
        ks = ['Comp_'+str(k+1) for k in range(Sk.shape[1])]
        c = [self.unit_name] + ks
        df = pd.DataFrame(np.c_[self.channel_vals, Sk],columns=c)
        df.to_csv(filename+'.csv',index=False)

    def save_intensity_to_csv(self, filename, normalize=True):
        """Save numerical values of component spatial intensity to csv files

        Parameters
        ----------
        filename: string
            file name of an output image
        normalize: Boolean, default True
            If true, each spectrum is normalized
        """

        for k in range(self.C_.shape[1]):
            im = np.reshape(self.C_[:, k], (self.num_x, self.num_y))
            np.savetxt('{:}_{:}.csv'.format(filename,k+1),im,delimiter=',')

class SVD(RandomMF):
    """Singular Value Decomposition (SVD)

    Parameters
    ----------
    n_components : int
        The number of components

    Attributes
    ----------
    C_ : array-like of shape(#spatial data points, n_components)
        Component intensities decomposed from data matrix X
    S_ : array-like of shape(#spectrum channels, n_components)
        Spectra decomposed from data matrix X.

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
        """Decompose into two Low-rank matrix by SVD

        Parameters
        ----------
        X: array-like of shape(#spatial data points of x, #that of y, #spectrum channels)
            Data matrix to be decomposed
        channel_vals: array-like of shape(#spectrum channels, )
            The sequence of channel numbers, or unit values
        unit_name: string
            The unit of the spectrum channel

        Returns
        -------
        self: instance of class SVD
        """

        # --- Attribute initialization from a data matrix------
        if X.ndim == 2:
            self.num_y = 1
            self.num_x, self.num_ch = X.shape
            self.num_xy = self.num_x * self.num_y
        else:
            self.num_x, self.num_y, self.num_ch = X.shape
            self.num_xy = self.num_x * self.num_y
            # transform from 3D-array to 2D-array (Data Matrix)
            X = X.reshape(self.num_xy, self.num_ch)

        if channel_vals is None:
            self.channel_vals = np.arange(self.num_ch)
        else:
            self.channel_vals = channel_vals
        if unit_name is None:
            self.unit_name = 'Channel'
        else:
            self.unit_name = unit_name

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

        return self

class PCA(RandomMF):
    """Principal Components Analysis (PCA)

    Parameters
    ----------
    n_components : int or None
        The number of components

    Attributes
    ----------
    C_ : array-like of shape(#spatial data points, n_components)
        Component intensities decomposed from data matrix X
    S_ : array-like of shape(#spectrum channels, n_components)
        Spectra decomposed from data matrix X.

    """

    # constructor
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
        X: array-like of shape(#spatial data points of x, #that of y, #spectrum channels)
            Data matrix to be decomposed
        channel_vals: array-like of shape(#spectrum channels, )
            The sequence of channel numbers, or unit values
        unit_name: string
            The unit of the spectrum channel

        Returns
        -------
        self: instance of class PCA
        """

        # --- Attribute initialization from a data matrix------
        if X.ndim == 2:
            self.num_y = 1
            self.num_x, self.num_ch = X.shape
            self.num_xy = self.num_x * self.num_y
        else:
            self.num_x, self.num_y, self.num_ch = X.shape
            self.num_xy = self.num_x * self.num_y
            # transform from 3D-array to 2D-array (Data Matrix)
            X = X.reshape(self.num_xy, self.num_ch)

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

        return self
