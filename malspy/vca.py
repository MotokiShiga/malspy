import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from .matrix_factorization import RandomMF

class VCA(RandomMF):
    """Vertex Component Analysis (VCA)

    Parameters
    ----------
    n_components : int or None
        Number of components, if n_components is not set all features
        are kept.

    Attributes
    ----------
    C_ : array_like of shape (# of spatial data points, n_components)
        Non-negative components decomposed from data X.
    S_ : array_like of shape (# of channels, n_components)
        Non-negative spectra decomposed from data X.
    Examples
    --------

    References
    ----------
    Jose' M. P. Nascimento, Jose' M. Bioucas,
     "Vertex component analysis: a fast algorithm to unmix hyperspectral data",
     IEEE Trans. on Geoscience and Remote Sensing, Vol.43, Issue: 4, 898-910, 2005.
     doi: 10.1109/TGRS.2005.844293
    """

    # constructor
    def __init__(self, n_components, proj_method='SVD'):
        self.n_components = n_components
        self.proj_method = proj_method

    def __repr__(self):
        class_name = self.__class__.__name__
        txt = 'n_components=' + str(self.n_components) \
              + ', proj_method=' + str(self.proj_method)
        return '%s(%s)' % (class_name, txt,)

    def __str__(self):
        txt = self.__repr__()
        return txt

    def fit(self, X, channel_vals=None, unit_name=None):
        """
        :param X: matrix with dimensions,  (# of pixels) x (# ofchannels)
        :param p: positive integer number of endmembers
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
        # -----------------------------------------------------

        R = X.T
        # Projection to remove observation noise
        if self.proj_method=='PCA':    # effective when small SNR
            print('PCA...')
            d = self.n_components - 1
            r_bar = np.mean(R, axis=1, keepdims=True)
            R0 = R - r_bar # data with zero-mean
            # Ud, Sd, Vd = scipy.sparse.linalg.svds(R0@R0.T/self.num_xy,d)  # computes the p-projection matrix
            Ud, Sd, Vd = scipy.linalg.svd(R@R.T/self.num_xy)  # computes the d-projection matrix
            Ud, Vd = Ud[:,:d], Vd[:,:d]
            Xp = Ud.T@R0    # project the zeros mean data onto p-subspace
            Xpori = Ud@Xp + r_bar  # again in original dimension
            c = np.sqrt(np.max(np.sum(Xp**2, axis=0)))
            Y = np.r_[Xp, c*np.ones((1,self.num_xy))]
        elif self.proj_method=='SVD':  # effective when large SNR
            print('SVD...')
            d = self.n_components
            # Ud, Sd, Vd = scipy.sparse.linalg.svds(R@R.T/self.num_xy,d)  # computes the d-projection matrix
            Ud, Sd, Vd = scipy.linalg.svd(R@R.T/self.num_xy)  # computes the d-projection matrix
            Ud, Vd = Ud[:,:d], Vd[:,:d]
            Xp = Ud.T@R # projection into d-dimensional space
            Xpori = Ud@Xp  # reconstruct spectra in original dimension (note that x_p has no null mean)
            u = np.mean(Xp, axis=1, keepdims=True)  # 低次元空間での平均ベクトル
            Y = Xp / (np.sum(Xp*u, axis=0, keepdims=True) + 10**16)
        else:
            print('Choose PCA or SVD as proj_method!')
            return

        #--- VCA main ---
        indice = np.zeros(self.n_components)
        indice = indice.astype(np.int64)
        A = np.zeros((self.n_components,self.n_components))
        A[-1,0] = 1
        for i in range(self.n_components):
            w = np.random.rand(self.n_components, 1)
            f =  w - A@np.linalg.pinv(A)@ w
            f = f / np.sqrt(f.T@f)
            v = np.abs(f.T@Y)
            indice[i] = np.argmax(v)
            A[:, i] = Y[:, indice[i]]
        self.S_ = (Xpori[:,indice])
        self.S_ori_ = (R[:,indice])

        # Optimize intensity distributions of components by Least-Squares fitting
        self.C_ = X @ np.linalg.pinv(self.S_).T
        self.C_ = (self.C_ + np.abs(self.C_))*0.5

        return self

    def plot_spectra_original(self, figsize=None, filename=None, normalize=True):
        '''
        Plot observed spectra of pure components determinined by VCA

        Parameters
        ----------
        figsize: 
            the vertical and horizontal size of the figure
        filename: string
            file name of an output image
        normalize: Boolean, default True
            If true, each spectrum is normalized
        '''
        if figsize is None:
            plt.figure()
        else:
            plt.figure(figsize=figsize)
        for k in range(self.S_ori_.shape[1]):
            if normalize:
                Sk = self.S_ori_[:, k] / (np.sqrt(np.sum(self.S_ori_[:, k]**2)) + 1e-16)
                plt.plot(self.channel_vals, Sk, label=str(k + 1))
            else:
                plt.plot(self.channel_vals, self.S_ori_[:, k], label=str(k + 1))
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

    def save_spectra_original_to_csv(self, filename, normalize=True):
        """Save numerical values of component original spectra to a csv file

        Parameters
        ----------
        filename: string
            file name of an output image
        normalize: Boolean, default True
            If true, each spectrum is normalized
        """

        Sk = self.S_ori_.copy()
        if normalize:
            for k in range(Sk.shape[1]):
                Sk[:, k] = Sk[:, k] / (np.sqrt(np.sum(Sk[:, k]**2)) + 1e-16)
        ks = ['Comp_'+str(k+1) for k in range(Sk.shape[1])]
        c = [self.unit_name] + ks
        df = pd.DataFrame(np.c_[self.channel_vals, Sk],columns=c)
        df.to_csv(filename+'.csv',index=False)

