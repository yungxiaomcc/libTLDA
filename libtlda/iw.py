#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as st
from scipy.spatial.distance import cdist
import sklearn as sk
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, \
    RidgeClassifier, RidgeClassifierCV
from sklearn.model_selection import cross_val_predict
from os.path import basename
from cvxopt import matrix, solvers

from .util import is_pos_def


class ImportanceWeightedClassifier(object):

    def __init__(self, loss_function='logistic', l2_regularization=None,
                 weight_estimator='lr', smoothing=True, clip_max_value=-1,
                 kernel_type='rbf', bandwidth=1):
        """
        参数
        ----------
        loss : str
            损失函数，可选的由，logistic，quadratic，hinge
        l2_regularization : float
            正则化参数
        iwe : str
            权重估计器，可选的有，lr，nn，rg，kmm
        smoothing : bool
            最小邻近平滑度
        clip : float
            最大允许重要性权重值
        kernel_type : str
            内核类型，可选值有 diste，rbf
        bandwidth : float
            内核 bandwidth参数

        返回值
        -------
        None

        """
        self.loss = loss_function
        self.l2 = l2_regularization
        self.iwe = weight_estimator
        self.smoothing = smoothing
        self.clip = clip_max_value
        self.kernel_type = kernel_type
        self.bandwidth = bandwidth

        if self.loss in ('lr', 'logr', 'logistic'):
            if l2_regularization:
                self.clf = LogisticRegression(C=self.l2, solver='lbfgs')
            else:
                self.clf = LogisticRegressionCV(cv=5, solver='lbfgs')
        elif self.loss in ('squared', 'qd', 'quadratic'):
            if l2_regularization:
                self.clf = RidgeClassifier(alpha=self.l2)
            else:
                self.clf = RidgeClassifierCV(cv=5)
        elif self.loss in ('hinge', 'linsvm', 'linsvc'):
            self.clf = LinearSVC()
        else:
            raise NotImplementedError('Loss function not implemented.')

        self.is_trained = False

        self.iw = []

    def iwe_ratio_gaussians(self, X, Z):
        
        N, DX = X.shape
        M, DZ = Z.shape

        if not DX == DZ:
            raise ValueError('Dimensionalities of X and Z should be equal.')

        mu_X = np.mean(X, axis=0)
        mu_Z = np.mean(Z, axis=0)

        Si_X = np.cov(X.T)
        Si_Z = np.cov(Z.T)

        if not (is_pos_def(Si_X) or is_pos_def(Si_Z)):
            print('Warning: covariate matrices not PSD.')

            regct = -6
            while not (is_pos_def(Si_X) or is_pos_def(Si_Z)):
                print('Adding regularization: ' + str(1**regct))

                Si_X += np.eye(DX)*10.**regct
                Si_Z += np.eye(DZ)*10.**regct

                regct += 1

        pT = st.multivariate_normal.pdf(X, mu_Z, Si_Z)
        pS = st.multivariate_normal.pdf(X, mu_X, Si_X)

        if np.any(np.isnan(pT)) or np.any(pT == 0):
            raise ValueError('Source probabilities are NaN or 0.')
        if np.any(np.isnan(pS)) or np.any(pS == 0):
            raise ValueError('Target probabilities are NaN or 0.')

        return pT / pS

    def iwe_kernel_densities(self, X, Z):
      
        N, DX = X.shape
        M, DZ = Z.shape

        if not DX == DZ:
            raise ValueError('Dimensionalities of X and Z should be equal.')

        pT = st.gaussian_kde(Z.T).pdf(X.T)
        pS = st.gaussian_kde(X.T).pdf(X.T)

        if np.any(np.isnan(pT)) or np.any(pT == 0):
            raise ValueError('Source probabilities are NaN or 0.')
        if np.any(np.isnan(pS)) or np.any(pS == 0):
            raise ValueError('Target probabilities are NaN or 0.')

        return pT / pS

    def iwe_logistic_discrimination(self, X, Z):
      
        N, DX = X.shape
        M, DZ = Z.shape

        if not DX == DZ:
            raise ValueError('Dimensionalities of X and Z should be equal.')

        y = np.concatenate((np.zeros((N, 1)),
                            np.ones((M, 1))), axis=0)

        XZ = np.concatenate((X, Z), axis=0)

        if self.l2:

            lr = LogisticRegression(C=self.l2, solver='lbfgs')

        else:
            lr = LogisticRegressionCV(cv=5, solver='lbfgs')

        preds = cross_val_predict(lr, XZ, y[:, 0], cv=5)

        return preds[:N]

    def iwe_nearest_neighbours(self, X, Z):
        
      
        N, DX = X.shape
        M, DZ = Z.shape

        if not DX == DZ:
            raise ValueError('Dimensionalities of X and Z should be equal.')

        d = cdist(X, Z, metric='euclidean')

        ix = np.argmin(d, axis=1)
        iw, _ = np.array(np.histogram(ix, np.arange(N+1)))

        if self.smoothing:
            iw = (iw + 1.) / (N + 1)

        if self.clip > 0:
            iw = np.minimum(self.clip, np.maximum(0, iw))

        return iw

    def iwe_kernel_mean_matching(self, X, Z):

        N, DX = X.shape
        M, DZ = Z.shape

        if not DX == DZ:
            raise ValueError('Dimensionalities of X and Z should be equal.')

        KXX = cdist(X, X, metric='euclidean')
        KXZ = cdist(X, Z, metric='euclidean')

        if not np.all(KXX >= 0):
            raise ValueError('Non-positive distance in source kernel.')
        if not np.all(KXZ >= 0):
            raise ValueError('Non-positive distance in source-target kernel.')

        if self.kernel_type == 'rbf':
            KXX = np.exp(-KXX / (2*self.bandwidth**2))
            KXZ = np.exp(-KXZ / (2*self.bandwidth**2))

        KXZ = N/M * np.sum(KXZ, axis=1)

        Q = matrix(KXX, tc='d')
        p = matrix(KXZ, tc='d')
        G = matrix(np.concatenate((np.ones((1, N)), -1*np.ones((1, N)),
                                   -1.*np.eye(N)), axis=0), tc='d')
        h = matrix(np.concatenate((np.array([N/np.sqrt(N) + N], ndmin=2),
                                   np.array([N/np.sqrt(N) - N], ndmin=2),
                                   np.zeros((N, 1))), axis=0), tc='d')

        sol = solvers.qp(Q, p, G, h)

        return np.array(sol['x'])[:, 0]

    def fit(self, X, y, Z):
        """
        模型拟合.

        参数
        ----------
        X : array
            源数据（NxD，N个样本，D个特征）
        y : array
            源标签（Nx1）
        Z : array
            目标数据（MxD，M个样本，D个特征）

        返回值
        -------
        None

        """
        N, DX = X.shape
        M, DZ = Z.shape

        if not DX == DZ:
            raise ValueError('Dimensionalities of X and Z should be equal.')

        if self.iwe == 'lr':
            self.iw = self.iwe_logistic_discrimination(X, Z)
        elif self.iwe == 'rg':
            self.iw = self.iwe_ratio_gaussians(X, Z)
        elif self.iwe == 'nn':
            self.iw = self.iwe_nearest_neighbours(X, Z)
        elif self.iwe == 'kde':
            self.iw = self.iwe_kernel_densities(X, Z)
        elif self.iwe == 'kmm':
            self.iw = self.iwe_kernel_mean_matching(X, Z)
        else:
            raise NotImplementedError('Estimator not implemented.')

        self.clf.fit(X, y, self.iw)

        self.is_trained = True

        self.train_data_dim = DX

    def predict(self, Z):
        
        M, D = Z.shape

        if self.is_trained:
            if not self.train_data_dim == D:
                raise ValueError('''Test data is of different dimensionality
                                 than training data.''')

        preds = self.clf.predict(Z)

        if self.loss == 'quadratic':
            preds = (np.sign(preds)+1)/2.

        return preds

    def predict_proba(self, Z):
        
        M, D = Z.shape

        if self.is_trained:
            if not self.train_data_dim == D:
                raise ValueError('''Test data is of different dimensionality
                                 than training data.''')

        if self.loss in ['logistic']:

            probs = self.clf.predict_proba(Z)

        else:
            raise NotImplementedError('''Posterior probabilities for quadratic
                                      and hinge losses not implemented yet.''')

        return probs

    def get_params(self):
        if self.is_trained:
            return self.clf.get_params()
        else:
            raise ValueError('Classifier is not yet trained.')

    def get_weights(self):
        if self.is_trained:
            return self.iw
        else:
            raise ValueError('Classifier is not yet trained.')

    def is_trained(self):
        return self.is_trained
