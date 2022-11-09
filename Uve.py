import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from numpy.linalg import matrix_rank as rank
import numpy as np

class UVE:
    def __init__(self, x, y, ncomp=10, nrep=228, testSize=30):

        '''
        X : 预测变量矩阵
        y ：标签
        ncomp : 结果包含的变量个数,8
        testSize: PLS中划分的数据集
        return ：波长筛选后的光谱数据
        '''

        self.x = x
        self.y = y
        # The number of latent components should not be larger than any dimension size of independent matrix
        self.ncomp = min([ncomp, rank(x)])
        self.nrep = nrep
        self.testSize = testSize
        self.criteria = None

        self.featureIndex = None
        self.featureR2 = np.full(self.x.shape[1], np.nan)
        self.selFeature = None

    def calcCriteria(self):
        PLSCoef = np.zeros((self.nrep, self.x.shape[1]))
        ss = ShuffleSplit(n_splits=self.nrep, test_size=self.testSize)
        step = 0
        for train, test in ss.split(self.x, self.y):
            xtrain = self.x[train, :]
            ytrain = self.y[train]
            plsModel = PLSRegression(min([self.ncomp, rank(xtrain)]))
            plsModel.fit(xtrain, ytrain)
            PLSCoef[step, :] = plsModel.coef_.T
            step += 1
        meanCoef = np.mean(PLSCoef, axis=0)
        stdCoef = np.std(PLSCoef, axis=0)
        self.criteria = meanCoef / stdCoef
        # print('self.criteria:',self.criteria) 

    def evalCriteria(self, cv=3):
        self.featureIndex = np.argsort(-np.abs(self.criteria))
        for i in range(self.x.shape[1]):
            xi = self.x[:, self.featureIndex[:i + 1]]
            if i<self.ncomp:
                regModel = LinearRegression()
            else:
                regModel = PLSRegression(min([self.ncomp, rank(xi)]))

            cvScore = cross_val_score(regModel, xi, self.y, cv=cv)
            self.featureR2[i] = np.mean(cvScore)

    def cutFeature(self, *args):
        cuti = np.argmax(self.featureR2)
        # self.selFeature = np.argsort(self.featureR2)[::-1][:self.ncomp] #选出前ncomp个featureR2
        
        self.selFeature = self.featureIndex[:cuti+1]
        # print('000')
        print('feature-shape:',np.array(self.selFeature).shape) 
        # print('111')
        data_feature = np.array(self.selFeature)
        fea = pd.DataFrame(data = data_feature)
    

        if len(args) != 0:
            returnx = list(args)
            i = 0
            for argi in args:
                if argi.shape[1] == self.x.shape[1]:
                    returnx[i] = argi[:, self.selFeature]
                i += 1
        # print('feature:',self.selFeature)
        return returnx,fea