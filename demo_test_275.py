import xlwt
import math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from collections import OrderedDict
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from copy import deepcopy
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import accuracy_score
from time import time
from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_X_y
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score
from mord import LogisticAT
from sklearn.model_selection import train_test_split

class KELMOR(ClassifierMixin, BaseEstimator):

    def __init__(self, C=100, method="full", S=None, eps=1e-5, kernel="linear", gamma=0.1, degree=3, coef0=1, kernel_params=None):
        self.C = C
        self.kernel = kernel
        self.method = method
        self.S = S
        self.eps = eps
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.X, self.y = X, y
        n, d = X.shape
        #  ---------------规范化类别标签：0,1,2,3,4,5-----------------
        self.le_ = preprocessing.LabelEncoder()
        self.le_.fit(y)
        y = self.le_.transform(y)
        #  --------------------------------------------------------
        classes = np.unique(y)
        nclasses = len(classes)

        self.M = np.array([[(i - j) ** 2 for i in range(nclasses)] for j in range(nclasses)])
        T = self.M[y, :]
        K = self._get_kernel(X)
        if self.method == "full":
            self.beta = np.linalg.inv((1 / self.C) * np.eye(n) + K).dot(T)
        else:
            raise ValueError("Invalid value for argument 'method'.")
        return self

    def predict(self, X):
        K = self._get_kernel(X, self.X)
        coded_preds = K.dot(self.beta)
        # print("coded_preds::",coded_preds.shape)
        predictions = np.argmin(np.linalg.norm(coded_preds[:, None] - self.M, axis=2, ord=1), axis=1)
        predictions = self.le_.inverse_transform(predictions)
        return predictions

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {'gamma': self.gamma,
                      'degree': self.degree,
                      'coef0': self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel, filter_params=True, **params)

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def predict_proba(self,X):
        K = self._get_kernel(X, self.X)
        coded_tmp = K.dot(self.beta)
        predictions = np.linalg.norm(coded_tmp[:, None] - self.M, axis=2, ord=2)
        predictions = -predictions
        predictions = np.exp(predictions)
        predictions_sum = np.sum(predictions, axis=1, keepdims=True)
        proba_matrix = predictions / predictions_sum
        return proba_matrix

class eSVM_rbf():
    def __init__(self):
        self.gamma = 0.1
        self.C = 100
        self.eX = self.ey = None

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.int32)
        self.nSample, self.nDim = X.shape
        self.labels = list(np.sort(np.unique(y)))
        self.nClass = len(self.labels)
        self.nTheta = self.nClass - 1
        self.extend_part = np.eye(self.nClass-1)
        self.label_dict = self.get_label_dict()
        self.eX, self.ey = self.train_set_construct(X=self.X, y=self.y)
        self.gram_train = self.get_gram_train()
        self.model = SVC(kernel='precomputed', C=100, probability=True)
        self.model.fit(self.gram_train, y=self.ey)
        return self

    def get_gram_train(self):
        gram_train_1 = rbf_kernel(X=self.eX[:,:self.nDim],gamma=self.gamma)
        gram_train_2 = self.eX[:,self.nDim:] @ self.eX[:,self.nDim:].T
        gram_train = gram_train_1 + gram_train_2
        return gram_train

    def get_label_dict(self):
        label_dict = OrderedDict()
        for i, lab in enumerate(self.labels):
            tmp_label = np.ones(self.nTheta)
            for k, pad in enumerate(self.labels[:-1]):
                if lab <= pad:
                    tmp_label[k] = 1
                else:
                    tmp_label[k] = -1
            label_dict[lab] = tmp_label
        return label_dict

    def train_set_construct(self, X, y):
        eX = np.zeros((self.nSample * self.nTheta, self.nDim + self.nTheta))
        ey = np.zeros(self.nSample * self.nTheta)

        for i in range(self.nSample):
            eXi = np.hstack((np.tile(X[i], (self.nTheta, 1)), self.extend_part))
            eX[self.nTheta * i: self.nTheta * i + self.nTheta] = eXi
            ey[self.nTheta * i: self.nTheta * i + self.nTheta] = self.label_dict[y[i]]
        return eX, ey


    def test_set_construct(self, X_test):
        nTest = X_test.shape[0]
        eX = np.zeros((nTest * self.nTheta, self.nDim + self.nTheta))
        for i in range(nTest):
            eXi = np.hstack((np.tile(X_test[i],(self.nTheta,1)), self.extend_part))
            eX[self.nTheta * i: self.nTheta * i + self.nTheta] = eXi
        return eX

    def get_gram_test(self, eX_test):
        gram_test_1 = rbf_kernel(X=eX_test[:,:self.nDim], Y=self.eX[:,:self.nDim],gamma=self.gamma)
        gram_test_2 = eX_test[:,self.nDim:] @ self.eX[:,self.nDim:].T
        gram_test = gram_test_1 + gram_test_2
        return gram_test

    def predict(self, X_test):
        nTest = X_test.shape[0]
        eX_test = self.test_set_construct(X_test=X_test)
        gram_test = self.get_gram_test(eX_test)
        y_extend = self.model.predict(gram_test)
        y_tmp = y_extend.reshape(nTest,self.nTheta)
        y_pred = np.sum(y_tmp < 0, axis=1).astype(np.int32)
        return y_pred

    def predict_proba(self, X_test):
        nTest = X_test.shape[0]
        eX_test = self.test_set_construct(X_test=X_test)
        gram_test = self.get_gram_test(eX_test)
        dist_tmp = self.model.decision_function(gram_test)
        dist_matrix = dist_tmp.reshape(nTest, self.nTheta)
        accumulative_proba = expit(dist_matrix)
        prob = np.pad(
            accumulative_proba,
            pad_width=((0, 0), (1, 1)),
            mode='constant',
            constant_values=(0, 1))
        prob = np.diff(prob)
        return prob

    def distant_to_theta(self, X_test):
        nTest = X_test.shape[0]
        eX_test = self.test_set_construct(X_test=X_test)
        gram_test = self.get_gram_test(eX_test)
        dist_tmp = self.model.decision_function(gram_test)
        dist_matrix = dist_tmp.reshape(nTest, self.nTheta)
        return dist_matrix



class RED_logist():
    def __init__(self):
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.int32)
        self.nSample, self.nDim = X.shape
        self.labels = list(np.sort(np.unique(y)))
        self.nClass = len(self.labels)
        self.nTheta = self.nClass - 1
        self.extend_part = np.eye(self.nClass-1)
        self.label_dict = self.get_label_dict()
        self.eX, self.ey = self.train_set_construct(X=self.X, y=self.y)
        self.model = LogisticRegression()
        self.model.fit(X=self.eX, y=self.ey)
        return self

    def get_label_dict(self):
        label_dict = OrderedDict()
        for i, lab in enumerate(self.labels):
            tmp_label = np.ones(self.nTheta)
            for k, pad in enumerate(self.labels[:-1]):
                if lab <= pad:
                    tmp_label[k] = 1
                else:
                    tmp_label[k] = -1
            label_dict[lab] = tmp_label
        return label_dict

    def train_set_construct(self, X, y):
        eX = np.zeros((self.nSample * self.nTheta, self.nDim + self.nTheta))
        ey = np.zeros(self.nSample * self.nTheta)

        for i in range(self.nSample):
            eXi = np.hstack((np.tile(X[i], (self.nTheta, 1)), self.extend_part))
            eX[self.nTheta * i: self.nTheta * i + self.nTheta] = eXi
            ey[self.nTheta * i: self.nTheta * i + self.nTheta] = self.label_dict[y[i]]
        return eX, ey

    def test_set_construct(self, X_test):
        nTest = X_test.shape[0]
        eX = np.zeros((nTest * self.nTheta, self.nDim + self.nTheta))
        for i in range(nTest):
            eXi = np.hstack((np.tile(X_test[i],(self.nTheta,1)), self.extend_part))
            eX[self.nTheta * i: self.nTheta * i + self.nTheta] = eXi
        return eX

    def predict(self, X):
        nTest = X.shape[0]
        eX_test = self.test_set_construct(X_test=X)
        y_extend = self.model.predict(X=eX_test)

        y_tmp = y_extend.reshape(nTest,self.nTheta)
        y_pred = np.sum(y_tmp < 0, axis=1).astype(np.int32)
        return y_pred


# name = "cleveland"
# name = "HDI2"
# name = "glass"
# name = "balance-scale"
# name = "newthyroid"
# name = "automobile"
# name = "Obesity2"
# name = "SWD"
# name = "housing-5bin"
# name = "machine-5bin"
# name = "stock-5bin"
# name = "stock-10bin"
# name = "winequality-red"
# name = "abalone-5bin"
# name = "penbased"
# name = "optdigits"
# name = "thyroid"
# name = "computer-5bin"
# name = "PowerPlant-10bin"
name = "ARWU2020-5bin"




data_path = Path(r"D:\OCdata")
read_data_path = data_path.joinpath(name + ".csv")
data = np.array(pd.read_csv(read_data_path, header=None))
X = np.asarray(data[:, :-1], np.float64)
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = data[:, -1]
y -= y.min()
ACC_list_1 = []
ACC_list_2 = []
ACC_list_3 = []
ACC_list_4 = []
Time_list_1 = []
Time_list_2 = []
Time_list_3 = []
Time_list_4 = []

# SKF = StratifiedKFold(n_splits=0.5, shuffle=True)
# for train_idx, test_idx in SKF.split(X, y):
#     X_train = X[train_idx]
#     y_train = y[train_idx].astype(np.int32)
#     X_test = X[test_idx]
#     y_test = y[test_idx]
for i in range(5):
    X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.7)
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    S_time = time()
    model_1 = RED_logist()
    model_1.fit(X=X_train, y=y_train)
    y_hat_1 = model_1.predict(X=X_test)
    acc_1 = accuracy_score(y_hat_1, y_test)
    ACC_list_1.append(acc_1)
    Time_list_1.append(time() - S_time)

    S_time = time()
    model_2 = KELMOR(C=100, kernel='rbf', gamma=0.1)
    model_2.fit(X=X_train, y=y_train)
    y_hat_2 = model_2.predict(X=X_test)
    acc_2 = accuracy_score(y_hat_2, y_test)
    ACC_list_2.append(acc_2)
    Time_list_2.append(time() - S_time)

    S_time = time()
    model_3 = LogisticAT()
    model_3.fit(X=X_train, y=y_train)
    y_hat_3 = model_3.predict(X=X_test)
    acc_3 = accuracy_score(y_hat_3, y_test)
    ACC_list_3.append(acc_3)
    Time_list_3.append(time() - S_time)

    S_time = time()
    model_4 = eSVM_rbf()
    model_4.fit(X=X_train, y=y_train)
    y_hat_4 = model_4.predict(X_test)
    acc_4 = accuracy_score(y_hat_4, y_test)
    ACC_list_4.append(acc_4)
    Time_list_4.append(time() - S_time)

print("Redlogist::",np.mean(ACC_list_1), "  Time:",np.mean(Time_list_1))
print("RedSVM ::",np.mean(ACC_list_4), "  Time:",np.mean(Time_list_4))
print("KELMOR ::",np.mean(ACC_list_2), "  Time:",np.mean(Time_list_2))
print("LogistAT ::",np.mean(ACC_list_3), "  Time:",np.mean(Time_list_3))



