import sys
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
sys.path.append('/home/liuruirui/OpenSA-main/OpenSA')
sys.path.append('/home/liuruirui/NIRS/local/code/doing/')
from NIRS_source import *
from WaveSelect.Lar import Lar
from WaveSelect.Spa import SPA
from WaveSelect.Uve import UVE
from WaveSelect.Cars import CARS_Cloud
from WaveSelect.Pca import Pca
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor


def SpctrumFeatureSelcet(method, X, y):
    """
       :param method: 波长筛选/降维的方法,包括:Cars, Lars, Uve, Spa, Pca
       :param X: 光谱数据, shape (n_samples, n_features)
       :param y: 光谱数据对应标签：格式：(n_samples,)
       :return: X_Feature: 波长筛选/降维后的数据, shape (n_samples, n_features)
                y:光谱数据对应的标签, (n_samples,)
    """
    if method == "None":
        x_selceted = X
    elif method== "Cars":
        Featuresecletidx = CARS_Cloud(X, y)
        x_selceted = X[:, Featuresecletidx]
    elif method == "Lars":
        Featuresecletidx = Lar(X, y)
        x_selceted = X[:, Featuresecletidx]
    elif method == "Uve":
        Uve = UVE(X, y, 8)
        Uve.calcCriteria()
        Uve.evalCriteria(cv=5)
        Featuresecletidx = Uve.cutFeature(X)
        x_selceted = Featuresecletidx[0]
    elif method == "Spa":
        Xcal, Xval, ycal, yval = train_test_split(X, y, test_size=0.25)
        Featuresecletidx = SPA().spa(
            Xcal= Xcal, ycal=ycal, m_min=5, m_max=20, Xval=Xval, yval=yval, autoscaling=1)
        x_selceted = X[:, Featuresecletidx]
    elif method == "Pca":
        x_selceted = Pca(X)
    else:
        print("no this method of SpctrumFeatureSelcet!")

    return x_selceted, y


def main(argv=None):
    # data1 = pd.read_excel(r'/home/liuruirui/NIRS/local/data/raw_data_all.xlsx', header=None)
    # data = data1.iloc[1:, ]
    # data_arr = np.array(data)
    # list_index = []
    # for i in range(0,1557,9):
    #         list_index.append(i)

    # # print(size(list_index)) 519
    # d_173 = data_arr[:, list_index]  
    # label1 = pd.read_excel(r'/home/liuruirui/NIRS/local/data/label.xls', )
    # label = np.array(label1)
    # label_Gu = label[:, 4]

    D1 = pd.read_csv(r'/home/liuruirui/NIRS/local/data/dataD1.csv', header=None)   
    D2 = pd.read_csv(r'/home/liuruirui/NIRS/local/data/dataD2.csv', header=None)   
    Lwave = pd.read_csv(r'/home/liuruirui/NIRS/local/data/dataWave.csv', header=None) 

    
    lgbm = LGBMRegressor()
    model_lr = LinearRegression()  # 建立普通线性回归模型对象
    # bagging = BaggingRegressor(n_estimators=10, random_state=0)
    # model_gbr = GradientBoostingRegressor()  # 建立梯度增强回归模型对象
    plsr = PLSRegression(n_components=10) 

    data_names = ['raw','pre_data','D1','D2','wave']  # 'DT','SNV','MSC','MC','SG',
    data_dic = [r_data, data, D1, D2, Lwave ] #DT, SNV, MSC,MC,SG,
    model_names = ['LGBM','linear','plsr10']  # ,'NGB''linear','plsr10','ada','Bagging','gbr'不同模型的名称列表'BayesianRidge', 'ada','GBR-new',
    model_dic = [lgbm,model_lr,plsr] 

    for x,named in zip(data_dic, data_names):  # 读出每个回归模型对象
        x = np.array(x)
        Featuresecletidx = Lar(x, label_protein)
        x_selceted = x[:, Featuresecletidx]
        # X, Y = SpctrumFeatureSelcet("Lar", x, label_protein)    
        x_train, x_test, y_train, y_test = train_test_split(x_selceted, label_protein,test_size=30,shuffle=True)
        for model,namem in zip(model_dic, model_names):  # 读出每个回归模型对象
            m = model.fit(x_train, y_train)
            pre_train=m.predict(x_train)
            pre_test=m.predict(x_test) 
            print('====protein=Lar40==',named,'===',namem,'==')
            evaluating(y_train,pre_train,'train')
            evaluating(y_test,pre_test,'test')


    print('success')


if __name__ == "__main__":
    sys.exit(main())