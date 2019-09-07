""" NMF (Nonnegative Matrix Factorization) for Spectrum Imaging Data Analysis
"""
# Author: Motoki Shiga <shiga_m@gifu-u.ac.jp>
# License: MIT


import numpy as np
from numpy import random
import numpy.linalg as lin
from scipy.special import gammaln
import matplotlib.pyplot as plt

from .matrix_factorization import RandomMF

class NMF(RandomMF):
    """Non-Negative Matrix Factorization (NMF)

    Parameters
    ----------
    n_components : int
        The number of components
    reps : int, optional (default = 3)
        The number of initializations
    max_itr : int, optional (default = 200)
        The maximum number of update iterations
    min_itr : int, optional (default = 10)
        The minimum number of update iterations
    random_seed : int, optional (default = 0)
        Random number generator seed control

    Attributes
    ----------
    C_ : ndarray of shape = (# of spatial data points, n_components)
        Spatial intensity distributions of factorized components
    S_ : ndarray of shape = (# of spectrum channels, n_components)
        Factorized component spectra
    E_ : ndarray of shape = (# of spatial data points in the 1st axis, # of those in 2nd axis)
        Residual spatial image (spatial image of RSME)
    obj_fun_ : ndarray of shape = (# of iterations)
        Learning curve of reconstruction error (Mean Squared Error)

    References
    ----------
    [1] Andrzej Cichocki, and Anh-Huy Phan,
        “Fast local algorithms for large scale nonnegative matrix and tensor factorizations”,
        IEICE transactions on fundamentals of electronics, communications and computer sciences 92 (3), 708-721, 2009.
    """

    def __init__(self, n_components, reps=3, max_itr=100, min_itr=10, random_seed=0):
        super(NMF, self).__init__(n_components=n_components, random_seed=random_seed)
        self.reps = reps
        self.max_itr = max_itr
        self.min_itr = min_itr

    def __repr__(self):
        class_name = self.__class__.__name__
        txt = 'n_components=' + str(self.n_components) \
              + ', reps=' + str(self.reps) + ', max_itr=' + str(self.max_itr) + ', min_itr=' + str(self.min_itr) + \
              ', random_seed=' + str(self.random_seed)
        return '%s(%s)' % (class_name, txt,)

    def __str__(self):
        txt = self.__repr__()
        return txt

    def fit(self, X, channel_vals=None, unit_name=None):
        """ Optimize a NMF model for spectrum imaging X

        Two matrices C_ and S_ are optimized by the HALS algorithm.

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
        self: instance of class NMF

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

        obj_best = np.inf
        random.seed(self.random_seed)  # set random seed
        print('Training NMF model....')
        for rep in range(self.reps):
            print(str(rep + 1) + 'th iteration of NMF algorithm')

            # initialization
            obj = np.zeros(self.max_itr)
            C = np.ones((self.num_xy, self.n_components))
            for j in range(self.n_components):
                C[:, j] = C[:, j] / (np.sqrt(C[:, j].T @ C[:, j]) + eps)
            cj = np.sum(C, axis=1)
            i = np.random.choice(self.num_xy, self.n_components)
            S = X[i, :].T

            # main loop
            for itr in range(self.max_itr):
                # update S
                XC = X.T @ C
                C2 = C.T @ C
                for j in range(self.n_components):
                    S[:, j] = XC[:, j] - S @ C2[:, j] + C2[j, j] * S[:, j]
                    S[:, j] = (S[:, j] + np.abs(S[:, j])) / 2  # replace negative values with zeros

                # update C
                XS = X @ S
                S2 = S.T @ S
                for j in range(self.n_components):
                    cj = cj - C[:, j]
                    C[:, j] = XS[:, j] - C @ S2[:, j] + S2[j, j] * C[:, j]
                    C[:, j] = (C[:, j] + np.abs(C[:, j])) / 2  # replace negative values with zeros
                    C[:, j] = C[:, j] / (np.sqrt(C[:, j].T @ C[:, j]+eps))  # normalize
                    cj = cj + C[:, j]

                # cost function (reconstruction error)
                X_est = C @ S.T  # reconstructed data matrix
                obj[itr] = lin.norm(X - X_est, ord='fro')**2 / X.size

                # check convergence
                if (itr > self.min_itr) & (np.abs(obj[itr - 1] - obj[itr]) < 10 ** (-10)):
                    obj = obj[0:itr]
                    break

            print('# updates: ' + str(itr))

            # choose the best result
            if obj_best > obj[-1]:
                obj_best = obj[-1]
                obj_fun_best = obj.copy()
                C_best = C.copy()
                S_best = S.copy()

        self.C_, self.S_, self.obj_fun_ = C_best, S_best, obj_fun_best

        # residual spatial image (spatial image of RSME)
        self.E_ = np.sqrt( np.mean((X - self.C_@self.S_.T)**2, axis=1) )
        self.E_ = self.E_.reshape(self.num_x, self.num_y)

        return self

    def plot_object_fun(self, figsize=None, filename=None):
        """ Plot learning curve (#iterations vs object function (error function))

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
        plt.plot(self.obj_fun_)
        plt.xlabel('The number of iterations')
        plt.xlim([1, len(self.obj_fun_)-1])
        plt.ylabel('Object function')
        plt.grid()
        plt.tight_layout()
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.close()

class NMF_SO(NMF):
    """Non-Negative Matrix Factorization with Soft orthogonality penalty (NMF-SO)

    Parameters
    ----------
    n_components : int
        The number of components
    wo : float, optional (default = 0.1)
        Weight of orthogonal penalty.
        The value should be between 0 and 1.
    reps : int, optional (default = 3)
        The number of initializations
    max_itr : int, optional (default = 200)
        The maximum number of update iterations
    min_itr : int, optional (default = 10)
        The minimum number of update iterations
    random_seed : int, optional (default = 0)
        Random number generator seed control

    Attributes
    ----------
    C_ : ndarray of shape = (# of spatial data points, n_components)
        Spatial intensity distributions of factorized components
    S_ : ndarray of shape = (# of spectrum channels, n_components)
        Factorized component spectra
    E_ : ndarray of shape = (# of spatial data points in the 1st axis, # of those in 2nd axis)
        Residual spatial image (spatial image of RSME)
    obj_fun_ : ndarray of shape = (# of iterations)
        Learning curve of reconstruction error (Mean Squared Error)

    References
    ----------
    Motoki Shiga, Kazuyoshi Tatsumi, Shunsuke Muto, Koji Tsuda,
     Yuta Yamamoto, Toshiyuki Mori, Takayoshi Tanji,
     "Sparse Modeling of EELS and EDX Spectral Imaging Data by Nonnegative Matrix Factorization",
     Ultramicroscopy, Vol.170, p.43-59, 2016.
     doi: 10.1016/j.ultramic.2016.08.006
    """

    # constructor
    def __init__(self, n_components, wo=0.1, reps=3, max_itr=100, min_itr=10, random_seed=0):
        super(NMF_SO, self).__init__(
            n_components=n_components,
            reps=reps,
            max_itr=max_itr,
            min_itr=min_itr,
            random_seed=random_seed)
        self.wo = wo

    def __repr__(self):
        class_name = self.__class__.__name__
        txt = 'n_components=' + str(self.n_components) + ', wo=' + str(self.wo) \
              + ', reps=' + str(self.reps) + ', max_itr=' + str(self.max_itr) + ', min_itr=' + str(self.min_itr) + \
              ', random_seed=' + str(self.random_seed)
        return '%s(%s)' % (class_name, txt,)

    def __str__(self):
        txt = self.__repr__()
        return txt

    def fit(self, X, channel_vals=None, unit_name=None):
        """Learn a NMF-SO model for spectral imaging X.

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
        self: instance of class NMF_SO

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

        obj_best = np.inf
        random.seed(self.random_seed)  # set the random seed
        print('Training NMF with soft orthogonal constraint....')
        for rep in range(self.reps):
            print(str(rep + 1) + 'th iteration of NMF-SO algorithm')

            # initialization
            obj = np.zeros(self.max_itr)
            C = np.ones((self.num_xy, self.n_components))
            for j in range(self.n_components):
                C[:, j] = C[:, j] / (np.sqrt(C[:, j].T @ C[:, j]) + eps)
            cj = np.sum(C, axis=1)
            i = np.random.choice(self.num_xy, self.n_components)
            S = X[i, :].T

            # main loop
            for itr in range(self.max_itr):
                # update S
                XC = X.T @ C
                C2 = C.T @ C
                for j in range(self.n_components):
                    S[:, j] = XC[:, j] - S @ C2[:, j] + C2[j, j] * S[:, j]
                    S[:, j] = (S[:, j] + np.abs(S[:, j])) / 2  # replace negative values with zeros

                # update C
                XS = X @ S
                S2 = S.T @ S
                for j in range(self.n_components):
                    cj = cj - C[:, j]
                    C[:, j] = XS[:, j] - C @ S2[:, j] + S2[j, j] * C[:, j]
                    C[:, j] = C[:, j] - self.wo * (cj.T @ C[:, j]) / (cj.T @ cj) * cj
                    C[:, j] = (C[:, j] + np.abs(C[:, j])) / 2  # replace negative values with zeros
                    C[:, j] = C[:, j] / (np.sqrt(C[:, j].T @ C[:, j] + eps))  # normalize
                    cj = cj + C[:, j]

                # cost function
                X_est = C @ S.T  # reconstructed data matrix
                obj[itr] = lin.norm(X - X_est, ord='fro')**2 / X.size

                # check of convergence
                if (itr > self.min_itr) & (np.abs(obj[itr - 1] - obj[itr]) < 10 ** (-10)):
                    obj = obj[0:itr]
                    break

            print('# updates: ' + str(itr))

            # choose the best result
            if obj_best > obj[-1]:
                obj_best = obj[-1]
                obj_fun_best = obj.copy()
                C_best = C.copy()
                S_best = S.copy()

        self.C_, self.S_, self.obj_fun_ = C_best, S_best, obj_fun_best

        # residual spatial image (spatial image of RSME)
        self.E_ = np.sqrt( np.mean((X - self.C_@self.S_.T)**2, axis=1) )
        self.E_ = self.E_.reshape(self.num_x, self.num_y)

        return self

class NMF_ARD_SO(NMF_SO):
    """Non-Negative Matrix Factorization with ARD and Soft orthogonality penalty (NMF-ARD-SO)

        Parameters
        ----------
        n_components : int
        The number of components
        wo : float, optional (default = 0.1)
            Weight of orthogonal penalty.
            The value should be between 0 and 1.
        reps : int, optional (default = 3)
            The number of initializations
        max_itr : int, optional (default = 200)
            The maximum number of update iterations
        min_itr : int, optional (default = 10)
            The minimum number of update iterations
        alpha: float, optional (default = 1e-15)
            Sparseness parameter, which must be over than 1
        threshold_merge: float, optional (default = 0.99)
            The threshold of similarity between components to judge components should be merged
        random_seed : int, optional (default = 0)
            Random number generator seed control

        Attributes
        ----------
        C_ : ndarray of shape = (# of spatial data points, n_components)
            Spatial intensity distributions of factorized components
        S_ : ndarray of shape = (# of spectrum channels, n_components)
            Factorized component spectra
        E_ : ndarray of shape = (# of spatial data points in the 1st axis, # of those in 2nd axis)
            Residual spatial image (spatial image of RSME)
        obj_fun_ : ndarray of shape = (# of iterations)
            Learning curve of reconstruction error (Mean Squared Error)
        beta_ : float
            Sparse penalty parameter (computed from alpha and data X)
        lambdas_ : ndarray of shape = (# of iterations)
            Learning curve of component intensities
        sigma2_ : float
            Estimation of noise variance

        References
        ----------
        Motoki Shiga, Kazuyoshi Tatsumi, Shunsuke Muto, Koji Tsuda,
         Yuta Yamamoto, Toshiyuki Mori, Takayoshi Tanji,
         "Sparse Modeling of EELS and EDX Spectral Imaging Data by Nonnegative Matrix Factorization",
         Ultramicroscopy, Vol.170, p.43-59, 2016.
         doi: 10.1016/j.ultramic.2016.08.006
    """

    def __init__(self, n_components, wo=0.1, reps=3, max_itr=100, min_itr=10,
                 alpha=1+10**(-15), threshold_merge=0.99, random_seed=0):
        super(NMF_ARD_SO, self).__init__(
            n_components=n_components,
            wo=wo,
            reps=reps,
            max_itr=max_itr,
            min_itr=min_itr,
            random_seed=random_seed)
        self.alpha = alpha
        self.threshold_merge = threshold_merge

    def __repr__(self):
        class_name = self.__class__.__name__
        txt = 'n_components=' + str(self.n_components) + ', wo=' + str(self.wo) \
              + ', reps=' + str(self.reps) + ', max_itr=' + str(self.max_itr) + ', min_itr=' + str(self.min_itr) + \
              ', alpha=' + str(self.alpha) + ', threshold_merge=' + str(self.threshold_merge) + ', random_seed=' + str(self.random_seed)
        return '%s(%s)' % (class_name, txt,)

    def __str__(self):
        txt = self.__repr__()
        return txt

    def fit(self, X, channel_vals=None, unit_name=None):
        """Learn a NMF-ARD-SO model for the data X.

        Parameters
        ----------
        X: ndarray, shape ``(# of spatial data points in the 1st axis, # of those in 2nd axis, # of spectrum channels)``
            Data matrix to be decomposed
        channel_vals: ndarray, shape ``(# of spectrum channels)``
            The sequence of channel values
        unit_name: string
            The unit name of spectrum channel

        Returns
        -------
        self: instance of class NMF_ARD_SO

        """

        eps = np.finfo(np.float64).eps  # tiny value (Machine limits for floating point types)

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

        mu_x = np.mean(X)
        self.beta_ = mu_x * (self.alpha - 1) * np.sqrt(self.num_ch) / self.n_components
        const = self.n_components * (gammaln(self.alpha) - self.alpha * np.log(self.beta_))
        random.seed(self.random_seed)  # set the random seed


        obj_best = np.inf  # to deposit the best object value
        print('Training NMF with ARD and soft orthogonal constraint....')
        for rep in range(self.reps):
            print(str(rep+1) + 'th iteration of NMF-ARD-SO algorithm')

            # --- Initialization ------
            C = (np.random.rand(self.num_xy, self.n_components) + 1) * (np.sqrt(mu_x / self.n_components))
            L = (np.sum(C, axis=0) + self.beta_) / (self.num_ch + self.alpha + 1)
            cj = np.sum(C, axis=1)
            i = np.random.choice(self.num_xy, self.n_components)
            S = X[i, :].T
            for j in range(self.n_components):
                c = (np.sqrt(S[:, j].T @ S[:, j]))  # normalize
                if c > 0:
                    S[:, j] = S[:, j] / c
                else:
                    S[:, j] = 1 / np.sqrt(self.num_ch)
            X_est = C @ S.T  # reconstructed data matrix
            sigma2 = np.mean((X - X_est) ** 2)
            obj = np.zeros(self.max_itr)
            lambdas = np.zeros((self.max_itr, self.n_components))
            # -------------------------

            for itr in range(self.max_itr):
                # update S (spectra)
                XC = X.T @ C
                C2 = C.T @ C
                for j in range(self.n_components):
                    S[:, j] = XC[:, j] - S @ C2[:, j] + C2[j, j] * S[:, j]
                    S[:, j] = (S[:, j] + np.abs(S[:, j])) / 2  # replace negative values with zeros
                    c = (np.sqrt(S[:, j].T @ S[:, j]))  # normalize
                    if c > 0:
                        S[:, j] = S[:, j] / c
                    else:
                        S[:, j] = 1 / np.sqrt(self.num_ch)

                # update C (component intensities)
                XS = X @ S
                S2 = S.T @ S
                for j in range(self.n_components):
                    cj = cj - C[:, j]
                    C[:, j] = XS[:, j] - C @ S2[:, j] + S2[j, j] * C[:, j]
                    C[:, j] = C[:, j] - sigma2 / L[j]
                    if (self.wo > 0):
                        C[:, j] = C[:, j] - self.wo * (cj.T @ C[:, j]) / (cj.T @ cj) * cj
                    C[:, j] = (C[:, j] + np.abs(C[:, j])) / 2  # replace negative values with zeros
                    cj = cj + C[:, j]

                # merge components if their spectra are almost same
                if itr > 3:
                    SS = S.T @ S
                    i, j = np.where(SS >= self.threshold_merge)
                    m = i < j
                    i, j = i[m], j[m]
                    for n in range(len(i)):
                        S[:, j[n]] = 1 / np.sqrt(self.num_ch)
                        C[:, i[n]] = np.sum(C[:, np.r_[i[n], j[n]]], axis=1)
                        C[:, j[n]] = 0
                if np.sum(cj) < eps:
                    C[:, :] = eps

                # update lambda(ARD parameters)
                L = (np.sum(C, axis=0) + self.beta_) / (self.num_xy + self.alpha + 1) + eps
                lambdas[itr, :] = L.copy()

                # update sigma2 (the variance of additive Gaussian noise)
                X_est = C @ S.T  # reconstructed data matrix
                sigma2 = np.mean((X - X_est) ** 2)

                # object function (negative log likelihood)
                obj[itr] = self.num_xy * self.num_ch / 2 * np.log(2 * np.pi * sigma2) + self.num_xy * self.num_ch / 2  # MSE
                obj[itr] = obj[itr] + (L ** (-1)).T @ (np.sum(C, axis=0) + self.beta_).T \
                           + (self.num_xy + self.alpha + 1) * np.sum(np.log(L), axis=0) + const

                # check of convergence
                if (itr > self.min_itr) & (np.abs(obj[itr - 1] - obj[itr]) < 10 ** (-10)):
                    obj = obj[0:itr]
                    lambdas = lambdas[0:itr, :].copy()
                    break

            print('# updates: ' + str(itr))

            # choose the best result
            if obj_best > obj[-1]:
                obj_best = obj[-1]
                obj_fun_best = obj.copy()
                C_best = C.copy()
                S_best = S.copy()
                lambdas_best = lambdas.copy()

        # for learning curve of object function
        self.obj_fun_ = obj_fun_best
        # replace tiny values with zeros
        C_best[C_best < eps] = 0
        S_best[S_best < eps] = 0
        L_best = (np.sum(C, axis=0) + self.beta_) / (self.num_xy + self.alpha + 1)
        k = np.argsort(-L_best)
        num_comp_best = np.sum(L_best[k] > eps)
        ks = k[:num_comp_best]
        self.C_, self.S_, self.L_ = C_best[:, ks], S_best[:, ks], L_best[ks]
        self.lambdas_ = lambdas_best[:, k] # leave all values to draw learning curve of ARD
        X_est = self.C_ @ self.S_.T  # reconstructed data matrix
        self.sigma2_ = np.mean((X - X_est) ** 2)

        # residual spatial image (spatial image of RSME)
        self.E_ = np.sqrt( np.mean((X - X_est)**2, axis=1) )
        self.E_ = self.E_.reshape(self.num_x, self.num_y)

        return self

    def plot_ard(self, figsize=None, filename=None):
        """
        Plot learning curve (#iterations vs component intensities)

        Parameters
        ----------
        figsize: list[int,int], shape ``(the size of horizontal axis, that of vertical axis)``, optional
            Figure size of vertical and horizontal axis
        filename: string, optional
            The file name of an output image
            If None, the image is not saved to a file. 
        """

        if figsize is None:
            plt.figure()
        else:
            plt.figure(figsize=figsize)
        for k in range(self.n_components):
            plt.plot(self.lambdas_[:, k], label=str(k + 1))
        plt.xlabel('The number of iterations')
        plt.ylabel('Component intensity')
        plt.xlim([0, self.lambdas_.shape[0]-1])
        plt.legend()
        plt.grid()
        plt.tight_layout()
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)

class BetaNMF(NMF):
    """Non-Negative Matrix Factorization with beta-divergence

    Parameters
    ----------
    n_components : int
        The number of components
    beta : float
           Parameter of divergence
           The value should be between -1 and 1.
    reps : int, optional (default = 3)
        The number of initializations
    max_itr : int, optional (default = 200)
        The number of update iterations
    min_itr : int, optional (default = 10)
        The minimum number of update iterations
    flag_random_init_C: bool, optional (default = True)
        Initialize C_ by random numbers.
        If False, initial C_ is set to same values.
    random_seed : int, optional (default = 0)
        Random number generator seed control

    Attributes
    ----------
    C_ : ndarray of shape = (# of spatial data points, n_components)
        Spatial intensity distributions of factorized components
    S_ : ndarray of shape = (# of spectrum channels, n_components)
        Factorized component spectra
    E_ : ndarray of shape = (# of spatial data points in the 1st axis, # of those in 2nd axis)
        Residual spatial image (spatial image of beta-divergence)
    obj_fun_ : ndarray of shape = (# of iterations)
        Learning curve of reconstruction error (beta-divergence)

    References
    ----------
     Keigo Kimura, Mineichi Kudo, Yuzuru Tanaka,
     "A column-wise update algorithm for nonnegative matrix factorization in Bregman divergence with an orthogonal constraint",
     Machine Learning, 103(2), 285-306, 2016.
     doi: doi.org/10.1007/s10994-016-5553-0

     Liangda Li, Guy Lebanon, Haesun Park, 
     "Fast Bregman divergence NMF using Taylor expansion and coordinate descent",
     Proc of KDD2012, 307–315, 2012.

     Andrzej Cichocki , Rafal Zdunek , Anh Huy Phan , Shun-ichi Amari,
     Nonnegative Matrix and Tensor Factorizations: Applications to Exploratory Multi-way Data Analysis and Blind Source Separation,
     Wiley Publishing, 2009.

     Motoki Shiga, Shunsuke. Muto, 
     "Non-negative matrix factorization and its extensions for spectral image data analysis", 
     e-Journal of Surface Science and Nanotechnology, 2019. (Accepted)

    """


    def __init__(self, n_components, beta, reps=3, max_itr=100, min_itr=10, flag_random_init_C=True, random_seed=0):
        super(BetaNMF, self).__init__(
            n_components=n_components,
            reps=reps,
            max_itr=max_itr,
            min_itr=min_itr,
            random_seed=random_seed)
        self.beta = beta
        self.flag_random_init_C = flag_random_init_C
        if (beta<-1)|(beta>1):
            print('Beta of the divergence parameter should be between -1 and +1!')

    def __repr__(self):
        class_name = self.__class__.__name__
        txt = 'n_components=' + str(self.n_components) + ', beta=' + str(self.beta) \
              + ', reps=' + str(self.reps) + ', max_itr=' + str(self.max_itr) + ', min_itr=' + str(self.min_itr) \
              + ', flag_random_C=' + str(self.flag_random_init_C) + ', random_seed=' + str(self.random_seed)
        return '%s(%s)' % (class_name, txt,)

    def __str__(self):
        txt = self.__repr__()
        return txt

    def fit(self, X, channel_vals=None, unit_name=None):
        """Learn a BetaNMF model for spectral imaging X.

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
        self: instance of class BetaNMF

        """

        # tiny value (Machine limits for floating point types)
        eps = np.finfo(np.float64).eps
        if np.min(X)==0:
            X = X + eps

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

        obj_best = np.inf
        random.seed(self.random_seed)  # set the random seed
        print('Training NMF with beta-divergence....')
        for rep in range(self.reps):
            print(str(rep + 1) + 'th iteration of BetaNMF_SO algorithm')

            # initialization
            obj = np.zeros(self.max_itr)
            if self.flag_random_init_C:
                C = np.random.rand(self.num_xy, self.n_components)
            else:
                C = np.ones((self.num_xy, self.n_components))
            for j in range(self.n_components):
                C[:, j] = C[:, j] / (np.sqrt(C[:, j].T @ C[:, j]) + eps)
            cj = np.sum(C, axis=1)
            i = np.random.choice(self.num_xy, self.n_components)
            S = X[i, :].T

            Xk = X - C @ S.T
            Ck = np.sum(C,axis=1, keepdims=False)

            # main loop
            for itr in range(self.max_itr):

                X_est = C@S.T
                if self.beta==-1: #Itakura-Saito divergence
                    P = 1/(X_est + eps)**2
                elif self.beta==0: #KL-divergence
                    P = 1/(X_est + eps)
                elif self.beta==1: #Euclidian distance
                    P = 1*np.ones(X_est.shape)
                else:
                    P = (X_est + eps)**(self.beta-1)

                for k in range(self.n_components):
                    Xk = Xk + np.outer(C[:,k],S[:,k])

                    S[:,k] = (P*Xk).T@C[:,k] / (P.T@(C[:,k]**2)+ eps)
                    S[:,k] = (S[:,k]+np.abs(S[:,k]))/2  + eps

                    Ck = Ck - C[:,k]
                    h = (P*Xk)@S[:,k]
                    L = Ck.T@h/(Ck.T@Ck)
                    C[:,k] = h / (P@(S[:,k]**2)+ eps)
                    C[:,k] = (C[:,k]+np.abs(C[:,k]))/2 + eps
                    C[:,k] = C[:,k] / np.sqrt(C[:,k]@C[:,k])
                    Ck = Ck + C[:,k]

                    Xk = Xk - np.outer(C[:,k], S[:,k])

                # cost function
                X_est = C @ S.T  # reconstructed data matrix
                if self.beta==-1:
                    obj[itr] = np.mean((X + eps)/(X_est + eps) - np.log((X + eps)/(X_est + eps))-1)
                elif self.beta==0:
                    obj[itr] = np.mean((X + eps)*np.log((X + eps)/(X_est + eps))- X + X_est)
                elif self.beta==1: #Euclidian distance
                    obj[itr] = lin.norm(X - X_est, ord='fro')**2 / X.size
                else:
                    obj[itr] = np.mean(1/self.beta/(self.beta+1)*(X**(self.beta+1)-X_est**(self.beta+1)\
                                - (self.beta+1)*X_est**(self.beta)*(X-X_est)))    

                # check of convergence
                if (itr > self.min_itr) & (np.abs(obj[itr - 1] - obj[itr]) < 10 ** (-10)):
                    obj = obj[0:itr-1]
                    break

            print('# updates: ' + str(itr))

            # choose the best result
            if obj_best > obj[-1]:
                obj_best = obj[-1]
                obj_fun_best = obj.copy()
                C_best = C.copy()
                S_best = S.copy()

        self.C_, self.S_, self.obj_fun_ = C_best, S_best, obj_fun_best

        # residual spatial image (spatial image of beta-divergence)
        X_est = self.C_ @ self.S_.T  # reconstructed data matrix
        if self.beta==-1:
            self.E_ = np.mean((X + eps)/(X_est + eps) - np.log((X + eps)/(X_est + eps))-1, axis=1)
        elif self.beta==0:
            self.E_ = np.mean((X + eps)*np.log((X + eps)/(X_est + eps))- X + X_est, axis=1)
        elif self.beta==1: #Euclidian distance
            self.E_ = np.mean((X - self.C_@self.S_.T)**2, axis=1)
        else:
            self.E_ = np.mean(1/self.beta/(self.beta+1)*(X**(self.beta+1)-X_est**(self.beta+1)\
                        - (self.beta+1)*X_est**(self.beta)*(X-X_est)), axis=1)  
        self.E_ = self.E_.reshape(self.num_x, self.num_y)

        return self

class BetaNMF_SO(BetaNMF):
    """Non-Negative Matrix Factorization with beta-divergence and soft orthogonality constraint

    Parameters
    ----------
    n_components : int
        The number of components
    beta : float
           Parameter of divergence
           The value should be between -1 and 1.
    wo : float, optional (default = 0.1)
           Weight parameter of orthogonality
           The value should be between 0 and 1.
    reps : int, optional (default = 3)
        The number of initializations
    max_itr : int, optional (default = 200)
        The number of update iterations
    min_itr : int, optional (default = 10)
        The minimum number of update iterations
    flag_random_init_C: bool, optional (default = True)
        Initialize C_ by random numbers.
        If False, initial C_ is set to same values.
    random_seed : int, optional (default = 0)
        Random number generator seed control

    Attributes
    ----------
    C_ : ndarray of shape = (# of spatial data points, n_components)
        Spatial intensity distributions of factorized components
    S_ : ndarray of shape = (# of spectrum channels, n_components)
        Factorized component spectra
    E_ : ndarray of shape = (# of spatial data points in the 1st axis, # of those in 2nd axis)
        Residual spatial image (spatial image of beta-divergence)
    obj_fun_ : ndarray of shape = (# of iterations)
        Learning curve of reconstruction error (beta-divergence)

    References
    ----------
     Motoki Shiga, Shunsuke. Muto, 
     "Non-negative matrix factorization and its extensions for spectral image data analysis", 
     e-Journal of Surface Science and Nanotechnology, 2019. (Accepted)
    """

    def __init__(self, n_components, beta, wo, reps=3, max_itr=100, min_itr=10, flag_random_init_C=True, random_seed=0):
        super(BetaNMF_SO, self).__init__(
            n_components=n_components,
            beta = beta,
            reps=reps,
            max_itr=max_itr,
            min_itr=min_itr,
            flag_random_init_C=flag_random_init_C,
            random_seed=random_seed)
        self.wo = wo

    def __repr__(self):
        class_name = self.__class__.__name__
        txt = 'n_components=' + str(self.n_components) + ', beta=' + str(self.beta) + ', wo=' + str(self.wo) \
              + ', reps=' + str(self.reps) + ', max_itr=' + str(self.max_itr) + ', min_itr=' + str(self.min_itr) \
              + ', flag_random_C=' + str(self.flag_random_init_C) + ', random_seed=' + str(self.random_seed)
        return '%s(%s)' % (class_name, txt,)

    def __str__(self):
        txt = self.__repr__()
        return txt

    def fit(self, X, channel_vals=None, unit_name=None):
        """Learn a BetaNMF model with soft orthogonality penalty for spectral imaging X.

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
        self: instance of class BetaNMF_SO

        """

        # tiny value (Machine limits for floating point types)
        eps = np.finfo(np.float64).eps
        if np.min(X)==0:
            X = X + eps

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

        obj_best = np.inf
        random.seed(self.random_seed)  # set the random seed
        print('Training NMF with beta-divergence and soft orthogonality....')
        for rep in range(self.reps):
            print(str(rep + 1) + 'th iteration of BetaNMF_SO algorithm')

            # initialization
            obj = np.zeros(self.max_itr)
            if self.flag_random_init_C:
                C = np.random.rand(self.num_xy, self.n_components)
            else:
                C = np.ones((self.num_xy, self.n_components))
            for j in range(self.n_components):
                C[:, j] = C[:, j] / (np.sqrt(C[:, j].T @ C[:, j]) + eps)
            cj = np.sum(C, axis=1)
            i = np.random.choice(self.num_xy, self.n_components)
            S = X[i, :].T

            Xk = X - C @ S.T
            Ck = np.sum(C,axis=1, keepdims=False)

            # main loop
            for itr in range(self.max_itr):

                X_est = C@S.T
                if self.beta==-1: #Itakura-Saito divergence
                    P = 1/(X_est + eps)**2
                elif self.beta==0: #KL-divergence
                    P = 1/(X_est + eps)
                elif self.beta==1: #Euclidian distance
                    P = 1*np.ones(X_est.shape)
                else:
                    P = (X_est + eps)**(self.beta-1)

                for k in range(self.n_components):
                    Xk = Xk + np.outer(C[:,k],S[:,k])

                    S[:,k] = (P*Xk).T@C[:,k] / (P.T@(C[:,k]**2)+ eps)
                    S[:,k] = (S[:,k]+np.abs(S[:,k]))/2  + eps

                    Ck = Ck - C[:,k]
                    h = (P*Xk)@S[:,k]
                    L = Ck.T@h/(Ck.T@Ck)
                    C[:,k] = (h - self.wo*L*Ck)  / (P@(S[:,k]**2)+ eps)
                    C[:,k] = (C[:,k]+np.abs(C[:,k]))/2 + eps
                    C[:,k] = C[:,k] / np.sqrt(C[:,k]@C[:,k])
                    Ck = Ck + C[:,k]

                    Xk = Xk - np.outer(C[:,k], S[:,k])

                # cost function
                X_est = C @ S.T  # reconstructed data matrix
                if self.beta==-1:
                    obj[itr] = np.mean((X + eps)/(X_est + eps) - np.log((X + eps)/(X_est + eps))-1)
                elif self.beta==0:
                    obj[itr] = np.mean((X + eps)*np.log((X + eps)/(X_est + eps))- X + X_est)
                elif self.beta==1: #Euclidian distance
                    obj[itr] = lin.norm(X - X_est, ord='fro')**2 / X.size
                else:
                    obj[itr] = np.mean(1/self.beta/(self.beta+1)*(X**(self.beta+1)-X_est**(self.beta+1)\
                                - (self.beta+1)*X_est**(self.beta)*(X-X_est)))    

                # check of convergence
                if (itr > self.min_itr) & (np.abs(obj[itr - 1] - obj[itr]) < 10 ** (-10)):
                    obj = obj[0:itr-1]
                    break

            print('# updates: ' + str(itr))

            # choose the best result
            if obj_best > obj[-1]:
                obj_best = obj[-1]
                obj_fun_best = obj.copy()
                C_best = C.copy()
                S_best = S.copy()

        self.C_, self.S_, self.obj_fun_ = C_best, S_best, obj_fun_best

        # residual spatial image (spatial image of beta-divergence)
        X_est = self.C_ @ self.S_.T  # reconstructed data matrix
        if self.beta==-1:
            self.E_ = np.mean((X + eps)/(X_est + eps) - np.log((X + eps)/(X_est + eps))-1, axis=1)
        elif self.beta==0:
            self.E_ = np.mean((X + eps)*np.log((X + eps)/(X_est + eps))- X + X_est, axis=1)
        elif self.beta==1: #Euclidian distance
            self.E_ = np.mean((X - self.C_@self.S_.T)**2, axis=1)
        else:
            self.E_ = np.mean(1/self.beta/(self.beta+1)*(X**(self.beta+1)-X_est**(self.beta+1)\
                        - (self.beta+1)*X_est**(self.beta)*(X-X_est)), axis=1)  
        self.E_ = self.E_.reshape(self.num_x, self.num_y)

        return self
