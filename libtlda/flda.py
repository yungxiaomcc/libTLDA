#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as st
from scipy.optimize import minimize
from scipy.sparse.linalg import eigs
from scipy.spatial.distance import cdist
import sklearn as sk
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_predict
from os.path import basename

from .util import is_pos_def, one_hot

# 特征层级领域自适应
class FeatureLevelDomainAdaptiveClassifier(object):


    def __init__(self, l2=0.0, loss='logistic', transfer_model='blankout',
                 max_iter=100, tolerance=1e-5, verbose=True):
        """
        参数
        ----------
        l2 : 浮点类型
            l2正则化参数值
        loss : 字符串
            分类器的损失函数，如果logistic或者quadratic
        transfer_model : 字符串
            迁移模型的分布，可选的由 dropout 和 blankout
        max_iter : 整形
            最大迭代次数
        tolerance : 浮点
            x上的收敛准则阈值
        verbose : 布尔
            用来显示化训练进度

        返回值
        -------
        None

        """
        # 分类器选择
        self.l2 = l2
        self.loss = 'logistic'
        self.transfer_model = transfer_model

        # 优化参数
        self.max_iter = max_iter
        self.tolerance = tolerance

        # 模型是否被训练
        self.is_trained = False

        # 训练数据的维度
        self.train_data_dim = 0

        # 分类参数
        self.theta = 0

        # Verbosity
        self.verbose = verbose

    def mle_transfer_dist(self, X, Z, dist='blankout'):
        """
        迁移模型参数最大似然估计器

        参数
        ----------
        X : array
            元数据集（D特征，N个样本）
        Z : array
            目标数据集（D个特征，M个样本）
        dist : str
            迁移模型的分布

        返回值
        -------
        iota : array
            估计的迁移模型参数

        """
        N, DX = X.shape
        M, DZ = Z.shape

        if not DX == DZ:
            raise ValueError('Dimensionalities of X and Z should be equal.')

        if (dist == 'blankout') or (dist == 'dropout'):

            eta = np.mean(X > 0, axis=0)
            zeta = np.mean(Z > 0, axis=0)

            iota = np.clip(1 - zeta / eta, 0, None)

        else:
            raise ValueError('Distribution unknown.')

        return iota

    def moments_transfer_model(self, X, iota, dist='blankout'):
      
        N, D = X.shape

        if (dist == 'dropout'):

            E = (1-iota) * X

            V = np.zeros((D, D, N))
            for i in range(N):
                V[:, :, i] = np.diag(iota * (1-iota)) * (X[i, :].T*X[i, :])

        elif (dist == 'blankout'):

            E = X

            V = np.zeros((D, D, N))
            for i in range(N):
                V[:, :, i] = np.diag(iota * (1-iota)) * (X[i, :].T * X[i, :])

        else:
            raise NotImplementedError('Transfer distribution not implemented')

        return E, V

    def flda_log_loss(self, theta, X, y, E, V, l2=0.0):
    
        N, D = X.shape

        if not np.all(np.sort(np.unique(y)) == (-1, 1)):
            raise NotImplementedError('Labels can only be {-1, +1} for now.')

        Xt = np.dot(X, theta)
        Et = np.dot(E, theta)
        alpha = np.exp(Xt) + np.exp(-Xt)
        beta = np.exp(Xt) - np.exp(-Xt)
        gamma = (np.exp(Xt).T * X.T).T + (np.exp(-Xt).T * X.T).T
        delta = (np.exp(Xt).T * X.T).T - (np.exp(-Xt).T * X.T).T

        A = np.log(alpha)

        dA = beta / alpha

        d2A = 1 - beta**2 / alpha**2

        L = np.zeros((N, 1))
        for i in range(N):
            L[i] = -y[i] * Et[i] + A[i] + dA[i] * (Et[i] - Xt[i]) + \
                   1./2*d2A[i]*np.dot(np.dot(theta.T, V[:, :, i]), theta)

        R = np.mean(L, axis=0)

        return R + l2*np.sum(theta**2, axis=0)

    def flda_log_grad(self, theta, X, y, E, V, l2=0.0):
      
        N, D = X.shape

        if not np.all(np.sort(np.unique(y)) == (-1, 1)):
            raise NotImplementedError('Labels can only be {-1, +1} for now.')

        Xt = np.dot(X, theta)
        Et = np.dot(E, theta)
        alpha = np.exp(Xt) + np.exp(-Xt)
        beta = np.exp(Xt) - np.exp(-Xt)
        gamma = (np.exp(Xt).T * X.T).T + (np.exp(-Xt).T * X.T).T
        delta = (np.exp(Xt).T * X.T).T - (np.exp(-Xt).T * X.T).T

        A = np.log(alpha)

        dA = beta / alpha

        d2A = 1 - beta**2 / alpha**2

        dR = 0
        for i in range(N):

            t1 = -y[i]*E[i, :].T

            t2 = beta[i] / alpha[i] * X[i, :].T

            t3 = (gamma[i, :] / alpha[i] - beta[i]*delta[i, :] /
                  alpha[i]**2).T * (Et[i] - Xt[i])

            t4 = beta[i] / alpha[i] * (E[i, :] - X[i, :]).T

            t5 = (1 - beta[i]**2 / alpha[i]**2) * np.dot(V[:, :, i], theta)

            t6 = -(beta[i] * gamma[i, :] / alpha[i]**2 - beta[i]**2 *
                   delta[i, :] / alpha[i]**3).T * np.dot(np.dot(theta.T,
                                                         V[:, :, i]), theta)

            dR += t1 + t2 + t3 + t4 + t5 + t6

        dR += l2*2*theta

        return dR

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

        # 特征维度判断
        if not DX == DZ:
            raise ValueError('Dimensionalities of X and Z should be equal.')


        Y, labels = one_hot(y, one_not=True)
        K = len(labels)
        iota = self.mle_transfer_dist(X, Z)
        E, V = self.moments_transfer_model(X, iota)

        # 选择损失函数
        if (self.loss == 'logistic'):

            theta = np.random.randn(DX, K)

            for k in range(K):

                def L(theta): return self.flda_log_loss(theta, X, Y[:, k],
                                                        E, V, l2=self.l2)

                def J(theta): return self.flda_log_grad(theta, X, Y[:, k],
                                                        E, V, l2=self.l2)

                results = minimize(L, theta[:, k], jac=J, method='BFGS',
                                   options={'gtol': self.tolerance,
                                            'disp': self.verbose})

                theta[:, k] = results.x

        elif (self.loss == 'quadratic'):

            theta = np.inv(E.T*E + np.sum(V, axis=2) + l2*np.eye(D))\
                         * (E.T * Y)
        self.theta = theta

        self.classes = labels

        self.is_trained = True

        self.train_data_dim = DX

    def predict(self, Z_):
      
        M, D = Z_.shape

        if self.is_trained:
            if not self.train_data_dim == D:
                raise ValueError('''Test data is of different dimensionality
                                 than training data.''')

        preds = np.argmax(np.dot(Z_, self.theta), axis=1)

        preds = self.classes[preds]

        return preds

    def get_params(self):
        """Get classifier parameters."""
        return self.clf.get_params()

    def is_trained(self):
        """Check whether classifier is trained."""
        return self.is_trained
