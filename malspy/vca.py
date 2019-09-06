import numpy as np
import scipy
from scipy import spatial, special
from scipy.sparse.linalg import eigs
import pandas as pd
import matplotlib.pyplot as plt
from .matrix_factorization import RandomMF

class VCA(RandomMF):
    """Vertex Component Analysis (VCA)

    Parameters
    ----------
    n_components : int
        Number of components
    proj_method : string, optional (default='SVD')
        projection method of high dimension data ('SVD' or 'PCA')

    Attributes
    ----------
    C_ : ndarray of shape = (# of spatial data points, n_components)
        Spatial intensity distributions of factorized components
    S_ : ndarray of shape = (# of spectrum channels, n_components)
        Factorized component spectra
    E_ : ndarray of shape = (# of spatial data points in the 1st axis, # of those in 2nd axis)
        Residual spatial image (spatial image of RMSE)

    References
    ----------
    Jose' M. P. Nascimento, Jose' M. Bioucas,
     "Vertex component analysis: a fast algorithm to unmix hyperspectral data",
     IEEE Trans. on Geoscience and Remote Sensing, Vol.43, Issue: 4, 898-910, 2005.
     doi: 10.1109/TGRS.2005.844293
    """

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
        """Learn VCA model (to find endmembers of pure spectra)

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
        self: instance of class VCA
        """

        # tiny value (Machine limits for floating point types)
        eps = np.finfo(np.float64).eps

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

        R = X.T
        # Projection to remove observation noise
        if self.proj_method=='PCA':    # effective when small SNR
            print('Runing dimension reduction by PCA...')
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
            print('Runing dimension reduction by SVD...')
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
        print('Training VCA...')
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

        # residual spatial image (spatial image of RSME)
        self.E_ = np.sqrt( np.mean((X - self.C_@self.S_.T)**2, axis=1) )
        self.E_ = self.E_.reshape(self.num_x, self.num_y)

        return self

    def plot_spectra_original(self, figsize=None, filename=None, normalize=True):
        """Plot component spectra by picking from observed spectra

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
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.close()

class NFINDER(RandomMF):
    """NFinder

    Find pure component spectra by NFinder algorithm

    Parameters
    ----------
    n_components : int
        Number of components
    max_itr : int, optional (default = 5000)
        The number of update iterations

    Attributes
    ----------
    C_ : ndarray of shape = (# of spatial data points, n_components)
        Spatial intensity distributions of factorized components
    S_ : ndarray of shape = (# of spectrum channels, n_components)
        Factorized component spectra
    E_ : ndarray of shape = (# of spatial data points in the 1st axis, # of those in 2nd axis)
        Residual spatial image (spatial image of RMSE)

    Reference
    -------
    M. Winter, “Fast autonomous spectral end-member determination in
    hyperspectral data,” Proc. of 13th Int. Conf. on Appl. Geologic Remote
    Sens., Vancouver, BC, Apr. 1999, vol. 2, pp. 337–344.
    """

    # constructor
    def __init__(self, n_components, max_itr=5000):
        self.n_components = n_components
        self.max_itr = max_itr

    def __repr__(self):
        class_name = self.__class__.__name__
        txt = 'n_components=' + str(self.n_components) + ', max_itr=' + str(self.max_itr)
        return '%s(%s)' % (class_name, txt,)

    def __str__(self):
        txt = self.__repr__()
        return txt

    def nfindr(self, x_proj):
        """ NFinder 
        
        Parameters
        ----------
        x_proj : ndarray of  shape = (# of pixels, # of reduced channels)
            Projected data matrix
        
        Returns
        -------
        endm : ndarray of shape = (# of projected channels, # of end members)
            end member vectors

        """

        # data size
        Nb_pix, Nb_bandes = x_proj.shape

        Nb_endm = Nb_bandes+1

        #enveloppe QHULL
        # print(' -- convex hull')
        if Nb_bandes>1:
            hull = spatial.ConvexHull(x_proj)
            ind = hull.vertices
        else:
            ind = np.array([np.argmin(x_proj),np.argmax(x_proj)])
        envlp = x_proj[ind,:]
        Nb_sommet, Nb_bandes = envlp.shape
        
        # choix de Nb_endm pixels
        ind_perm = np.random.permutation(Nb_sommet)
        combi = np.sort(ind_perm[:Nb_endm])

        # candidate
        candidat_n = (envlp[combi,:Nb_bandes]).T
        critere_n = - np.abs( np.linalg.det( np.r_[candidat_n, np.ones((1,Nb_endm))] ) )
        candidat_opt = candidat_n
        critere_opt = critere_n

        # n+1
        for iter in range(self.max_itr):
            # candidate
            ind_perm = np.random.permutation(Nb_sommet)
            combi = np.sort(ind_perm[:Nb_endm])
            candidat = (envlp[combi,:Nb_bandes]).T
            critere = - np.abs( np.linalg.det( np.r_[candidat, np.ones((1,Nb_endm))] ) )

            #test
            delta_critere = critere-critere_n
            if delta_critere < 0:
                critere_n = critere
                if critere < critere_opt:
                    critere_opt = critere
                    candidat_opt = candidat
        endm = candidat_opt
        return endm

    def fit(self, X, channel_vals=None, unit_name=None):
        """ Find endmembers

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
        self: instance of class NFINDER
        """

        # tiny value (Machine limits for floating point types)
        eps = np.finfo(np.float64).eps

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

        L_red = self.n_components-1

        print('Training N-Finder...')

        # projection by PCA
        X_bar = np.mean(X, axis=0, keepdims=True)
        Rmat = X - X_bar
        Rmat = Rmat.T@Rmat
        D, vect_prop = eigs(Rmat, L_red)
        vect_prop = np.real(vect_prop)
        V = (vect_prop[:, :L_red]).T # first L_red eigenvectors
        matU, matP = V, V.T # projector and inverse projector
        x_proj = (X - X_bar)@matP # projection

        # Endmember findings by N-Finder
        endm_proj = self.nfindr(x_proj)
        self.S_ = (endm_proj.T @ matU).T + X_bar.T

        # Optimize intensity distributions of components by Least-Squares fitting
        self.C_ = X @ np.linalg.pinv(self.S_).T
        self.C_ = (self.C_ + np.abs(self.C_))*0.5

        # residual spatial image (spatial image of RSME)
        self.E_ = np.sqrt( np.mean((X - self.C_@self.S_.T)**2, axis=1) )
        self.E_ = self.E_.reshape(self.num_x, self.num_y)

        return self

