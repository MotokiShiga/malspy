# MALSpy
Python package for MAchine Learning based Spectral imaging data analysis

<b>Author</b>: Motoki Shiga (shiga_m at gifu-u.ac.jp)

This package provides major spectral imaging analysis methods based on machine learning such as SVD, PCA, VCA<sup>[1]</sup>, NMF<sup>[2]</sup>, NMF-SO<sup>[3]</sup>, NMF-ARD-SO<sup>[3]</sup>. 
In the new version (0.4.0), BetaNMF<sup>[4,5]</sup> and BetaNMF_SO<sup>[5]</sup>, which assumes a generalized noise model of beta-divergence including  Poisson noise model and Gaussian noise model, have been added. Please enjoy demo (example/demo.ipynb).

Please cite [3] and [5] if someone uses this package in research publications. 

<b>Reference:</b><br>
[1] J. M. P. Nascimento and J. M. Bioucas, "Vertex component analysis: a fast algorithm to unmix hyperspectral data", IEEE Trans. on Geoscience and Remote Sensing, 43(4), 898-910, 2005.

[2] A. Cichocki and A.-H. Phan, “Fast local algorithms for large scale nonnegative matrix and tensor factorizations”, IEICE transactions on fundamentals of electronics, communications and computer sciences 92(3), 708-721, 2009.

[3] M. Shiga, K. Tatsumi, S. Muto, K. Tsuda, Y. Yamamoto, T. Mori, and T. Tanji, "Sparse Modeling of EELS and EDX Spectral Imaging Data by Nonnegative Matrix Factorization", Ultramicroscopy, 170, 43-59, 2016.

[4] K. Kimura, M. Kudo, and Y. Tanaka, "A column-wise update algorithm for nonnegative matrix factorization in Bregman divergence with an orthogonal constraint", Machine Learning, 103(2), 285-306, 2016.

[5] M. Shiga and S. Muto, "Non-negative matrix factorization and its extensions for spectral image data analysis", e-Journal of Surface Science and Nanotechnology, 17, 148-154, 2019.
