import sys
from lightgbm import LGBMRegressor
# from lightgbm import LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.feature_selection import RFE, VarianceThreshold
# from yellowbrick.regressor import ResidualsPlot
sys.path.append('/home/liuruirui/NIRS')
from local.code.doing import CARS
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import BayesianRidge, LinearRegression,Ridge,ElasticNet,Lasso
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest,f_regression,mutual_info_regression
from sklearn.metrics import *
from sklearn.decomposition import PCA
from Preprocess import *
from numpy import *
import time
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import learning_curve


# 油菜籽：
data1 = pd.read_excel(r'/home/liuruirui/NIRS/local/data/raw_data_all.xlsx', header=None)
data = data1.iloc[1:, ]
label1 = pd.read_excel(r'/home/liuruirui/NIRS/local/data/label.xls', )
label0 = label1.iloc[1:,1:]
label = np.array(label1)
data_all = pd.concat([data, label0], axis=1)

rawdata = pd.read_excel(r'/home/liuruirui/NIRS/local/data/data_all.xlsx', header=None)
r_data = rawdata.iloc[1:, ]
r_data_arr = np.array(r_data)  # 波长对应数值

# data_all_arr = np.array(data_all)
# 11亮氨酸、10异亮氨酸、2苏氨酸、8缬氨酸、14赖氨酸、13苯丙氨酸、15组氨酸
label_TMD_arr = label[:, 1]
label_Gu = label[:, 4]
label_Gan = label[:, 5]

label_liang = label[:, 11]
label_yiliang = label[:, 10]
label_su = label[:, 2]
label_jie = label[:, 8]
label_lai = label[:, 14]
label_benbing = label[:, 13]
label_zu = label[:, 15]
label_protein = label[:, 19]


r2 = []
rmse = []
rpd =[]
n_components = []


wave1 = data1.iloc[0, ]
wave1.to_csv(r'/home/liuruirui/NIRS/local/data/wave.csv', float_format='%.2f')  # 保留两位小数

wave_arr = np.array(wave1)  # 波长
data_arr = np.array(data)  # 波长对应数值

dataD1 = pd.read_csv(r'/home/liuruirui/NIRS/local/data/dataD1.csv', header=None)
D1_arr = np.array(dataD1)

dataD2 = pd.read_csv(r'/home/liuruirui/NIRS/local/data/dataD2.csv', header=None)
D2_arr = np.array(dataD2)
# print(D2_arr.shape)  (228, 1555)

dataWave = pd.read_csv(r'/home/liuruirui/NIRS/local/data/dataWave.csv', header=None)
wave_arr = np.array(dataWave)
# print(wave_arr.shape)  (228, 1558)

lgbm = LGBMRegressor()
model_lr = LinearRegression()  # 建立普通线性回归模型对象
bagging = BaggingRegressor(n_estimators=10, random_state=0)
model_gbr = GradientBoostingRegressor()  # 建立梯度增强回归模型对象
plsr = PLSRegression(n_components=10) 
svr = SVR(C=2, gamma=1e-07, kernel='linear')

model_names = ['LGBM','linear','plsr10','svr']  # ,'NGB''linear','plsr10','ada','Bagging','gbr'不同模型的名称列表'BayesianRidge', 'ada','GBR-new',
model_dic = [lgbm,model_lr,plsr,svr] 


def CARS_select():
    for i in range(0, 5):
        lis1 = CARS.CARS_Cloud(data_arr,label_TMD_arr)
        print('#'*80)
        print("获取波段数:", len(lis1))
        print(lis1)
        print('#'*80)


def draw_spec(y, method):
    # x = np.array(x)
    y = np.array(y)
    plt.figure(500)
    x_col = np.linspace(0, 1557, num=y.shape[1])
    # x_col = x
    y_col = np.transpose(y)  # 数组逆序np.transpose
    plt.plot(x_col, y_col)
    plt.xlim(0, 1557)
    # plt.xlabel("Wavenumber($nm$)")
    # plt.ylabel('Absorbance')

    # plt.title("The spectrum of the " + method + " for dataset",
            #   fontweight="semibold", fontsize='large')  # 记得改名字MSC
    plt.title("Attention Weight",fontweight="semibold", fontsize='large') 

    plt.savefig('/home/liuruirui/Rapeseed/figure_result/'+method+'.png', dpi=600, format='png')
    # plt.show()

# draw_spec()


# 随机划分数据集
def train_test(x_data, y_data, test_ratio):
    # test_ratio = 0.3
    x_train, x_test, Y_train, Y_test = train_test_split(
        x_data, y_data, test_size=test_ratio, shuffle=True, random_state = 22)
    return x_train, x_test, Y_train, Y_test

# 几种特征选择方法
def PCA_(data, X_train, X_test):
    # PCA降维
    pca = PCA(n_components=100)  # n_components == min(n_samples, n_features)
    pca.fit(data)
    X_train_reduction = pca.transform(X_train)
    X_test_reduction = pca.transform(X_test)

    # print("特征个数:", pca.n_components_)  228
    # print('n_features:', pca.n_features_) 1557
    # print(X_train_reduction.shape)  (171, 228)

    # PCA后特征数据
    # plt.figure(100)
    # plt.scatter(X_train_reduction[:, 0], X_train_reduction[:, 1], marker='o')
    # plt.xlabel("Wavenumber(nm)")
    # plt.ylabel("Absorbance")
    # plt.title("The  PCA for corn dataset", fontweight="semibold", fontsize='large')
    # plt.savefig('.//Result//PCA.png')
    # plt.show()
    return X_train_reduction, X_test_reduction

def VarianceSelect(x):
    var=VarianceThreshold(threshold=0.0025)
    x_new=var.fit_transform(x)
    return x_new

def SelectK_f(k0,X,y): #方差分析
    x_new = SelectKBest(f_regression, k=k0).fit_transform(X, y)
    return x_new

def SelectK_m(k0,X,y): #互信息
    x_new = SelectKBest(mutual_info_regression, k=k0).fit_transform(X, y)
    return x_new

def Rfe(X,y):
    estimator = SVR(kernel="linear")
    e1 = LinearRegression()
    selector = RFE(estimator, n_features_to_select=15, step=1)
    selector = selector.fit(X, y)
    x_new = selector.transform(X)    
    return x_new


def draw_result(true, pred, name):
    # 绘制拟合图片
    plt.figure(500)
    x_col = np.linspace(0, true.shape[0], true.shape[0])  # 数组逆序

    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
    plt.scatter(x_col,true, label='Ture', color='blue')
    plt.plot(x_col, pred, label='predict', marker='D', color='red')
    plt.legend(loc='best')
    plt.xlabel("samples")
    plt.ylabel("the value of samples")
    plt.title("The Result of prediction")
    plt.savefig('/home/liuruirui/NIRS/local/Result/prediction_result/'+name+'.png') 

def draw_result_line(true, pred, name):
    # 绘制拟合图片
    plt.figure(dpi=300,figsize=(8,8))
    x_col = np.linspace(true.min(), true.max(), true.shape[0])  # 数组逆序
    plt.ylim(true.min(), true.max())
    plt.grid(True) #网格线
    plt.scatter(pred,true, label='True', color='blue')
    # plt.scatter(pred,pred, label='Predict', color='red')
    # plt.plot(x_col, pred, label='Predict', marker='D', color='red')
    plt.plot(x_col,x_col,color='black',label="best line")
    plt.legend(loc='best')
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.title("The Result of prediction")
    plt.savefig('/home/liuruirui/NIRS/local/Result/prediction_result/line-'+name+'.png') 

# 回归预测
def Svr_regre(x_train, y_train, x_test, k):
    # 训练SVR模型

    # 训练规模
    train_size = 100
    # 初始化SVR
    svr = GridSearchCV(SVR(kernel=k), cv=5,
                       param_grid={
                                   "gamma": np.logspace(-2, 2, 5)})   # 'poly', 'rbf', 'sigmoid'
    # 记录训练时间

   

    print(k,'start training:')
    t0 = time.time()
    # 训练
    svr.fit(x_train, y_train)
    time_train = time.time() - t0
    print('time_train:', time_train)
    print('验证参数:\n', svr.best_params_)

    # 测试
    print('start testing:')
    t0 = time.time()
    test_pre = svr.predict(x_test)
    time_predict = time.time() - t0
    print('time_predict:', time_predict)

    train_pre = svr.predict(x_train)
    return train_pre, test_pre

def pls2(x_train, y_train, x_test, i):
    # pls预测
    train_pred_list = []
    test_pred_list = []
    # for i in range(10, 17):
    pls2 = PLSRegression(n_components=i)
    pls2.fit(x_train, y_train)
    pred1 = pls2.predict(x_train)
    train_pred_list.append(pred1)
                    # score_train.append(pls2.score(pred1, y_train))

    pred = pls2.predict(x_test)
    test_pred_list.append(pred)
                    # score_test.append(pls2.score(pred, y_test))
    n_components.append(i)
    # pls2.fit(X_train_reduction, y_train)

    # train_pred = pls2.predict(X_train_reduction)
    # pred = pls2.predict(X_test_reduction)

    return pred1, pred #, score_train, score_test, n_components


def evaluating(true, pre, name):
    """
    :param true: (n_samples, )
    :param pre: (n_samples, )
    :return: None
    """
    # evs_ = explained_variance_score(true, pre)
    # mae_ = mean_absolute_error(true, pre)
    mse_ = mean_squared_error(true, pre)
    r2_ = r2_score(true, pre)
    rmse_ = mean_squared_error(true, pre, squared=False)
    rpd_ = np.std(true)/rmse_
    r2.append(r2_)
    rmse.append(rmse_)
    rpd.append(rpd_)

    print(name + ' R2    MSE    RMSE    RPD')
    print('结果    %6.4f %6.4f   %6.4f    %6.4f' % (r2_, mse_, rmse_, rpd_))
    print("*"*70)


def draw_labelDistribute(y,name):
    # 标签浓度的分布图
    
    # y = label[:, i]
    plt.clf()
    sns.distplot(y)
    plt.xlabel('Protein(mg/100mg)')
    plt.savefig('/home/liuruirui/Rapeseed/figure_result/rapeseed_protein_label_'+name+'.png')
    plt.show()

def draw_scatter(y,name):
    x= np.linspace(0, y.shape[0], y.shape[0])
    plt.scatter(x, y)
    yList = y.tolist()
    for i,j in zip(x, y):
        yIndex = yList.index(j)
        plt.annotate("%s" % yIndex, xy=(i,j), xytext=(-15,-10), textcoords='offset points')
        
             
    # plt.xlabel('Protein(mg/100mg)')
    plt.savefig('/home/liuruirui/NIRS/local/Result/label/'+name+'.png')
 
def get_mahalanobis(x, i, j): #获取马氏距离
    xT = x.T  # 求转置
    D = np.cov(xT)  # 求协方差矩阵
    invD = np.linalg.inv(D)  # 协方差逆矩阵
    assert 0 <= i < x.shape[0], "点 1 索引超出样本范围。"
    assert -1 <= j < x.shape[0], "点 2 索引超出样本范围。"
    x_A = x[i]
    x_B = x.mean(axis=0) if j == -1 else x[j]
    tp = x_A - x_B
    return np.sqrt(dot(dot(tp, invD), tp.T))

def spxy(x, y, test_size=0.25): #SPXY法划分数据集
    """
    :param x: shape (n_samples, n_features)
    :param y: shape (n_sample, )
    :param test_size: the ratio of test_size
    :return: spec_train :(n_samples, n_features)
             spec_test: (n_samples, n_features)
             target_train: (n_sample, )
             target_test: (n_sample, )
    """
    x_backup = x
    y_backup = y   
    M = x.shape[0]            
    N = round((1-test_size) * M)
    samples = np.arange(M)    
 
    y = (y - np.mean(y))/np.std(y)
    D = np.zeros((M, M))
    Dy = np.zeros((M, M))
 
    for i in range(M-1):
        xa = x[i, :]
        ya = y[i]
        for j in range((i+1), M):
            xb = x[j, :]
            yb = y[j]
            D[i, j] = np.linalg.norm(xa-xb)   
            Dy[i, j] = np.linalg.norm(ya - yb) 
 
    Dmax = np.max(D)       
    Dymax = np.max(Dy)
    D = D/Dmax + Dy/Dymax 
 
    maxD = D.max(axis=0)              
    index_row = D.argmax(axis=0)     
    index_column = maxD.argmax()      
 
    m = np.zeros(N)
    m[0] = index_row[index_column]
    m[1] = index_column
    m = m.astype(int)        
 
    dminmax = np.zeros(N)
    dminmax[1] = D[m[0], m[1]]  
 
    for i in range(2, N): 
        pool = np.delete(samples, m[:i])
        dmin = np.zeros(M-i)  
        for j in range(M-i): 
            indexa = pool[j] 
            d = np.zeros(i)  
            for k in range(i):  
                indexb = m[k] 
                if indexa < indexb:  
                    d[k] = D[indexa, indexb]
                else:
                    d[k] = D[indexb, indexa]
            dmin[j] = np.min(d)    
        dminmax[i] = np.max(dmin)   
        index = np.argmax(dmin)    
        m[i] = pool[index]        
 
    m_complement = np.delete(np.arange(x.shape[0]), m)     
 
    spec_train = x[m, :]
    target_train = y_backup[m]
    spec_test = x[m_complement, :]
    target_test = y_backup[m_complement]
 
    return spec_train, spec_test, target_train, target_test

def ks(x, y, test_size): #KS法划分数据集
    """
    :param x: shape (n_samples, n_features)
    :param y: shape (n_sample, )
    :param test_size: the ratio of test_size (float)
    :return: spec_train: (n_samples, n_features)
             spec_test: (n_samples, n_features)
             target_train: (n_sample, )
             target_test: (n_sample, )
    """
    M = x.shape[0]             
    N = round((1-test_size) * M)
    samples = np.arange(M)     
 
    D = np.zeros((M, M))       
 
    for i in range((M-1)):
        xa = x[i, :]
        for j in range((i+1), M):
            xb = x[j, :]
            D[i, j] = np.linalg.norm(xa-xb) 
 
    maxD = np.max(D, axis=0)             
    index_row = np.argmax(D, axis=0)    
    index_column = np.argmax(maxD)      
 
    m = np.zeros(N)
    m[0] = np.array(index_row[index_column])
    m[1] = np.array(index_column)
    m = m.astype(int)                   
    dminmax = np.zeros(N)
    dminmax[1] = D[m[0], m[1]]  
 
    for i in range(2, N):  
        pool = np.delete(samples, m[:i]) 
        dmin = np.zeros((M-i))        
        for j in range((M-i)):        
            indexa = pool[j]         
            d = np.zeros(i)           
            for k in range(i):         
                indexb = m[k]         
                if indexa < indexb:   
                    d[k] = D[indexa, indexb]
                else:
                    d[k] = D[indexb, indexa]
            dmin[j] = np.min(d)    
        dminmax[i] = np.max(dmin)   
        index = np.argmax(dmin)     
        m[i] = pool[index]          
 
    m_complement = np.delete(np.arange(x.shape[0]), m)    
 
    spec_train = x[m, :]
    target_train = y[m]
    spec_test = x[m_complement, :]
    target_test = y[m_complement]
    return spec_train, spec_test, target_train, target_test

def kennardstonealgorithm(x_variables, k):  
    '''
    k为训练集样本个数
    a=kennardstonealgorithm(x_variables, 72)
    #得到训练集和测试集索引值
    train=a[0]
    test=a[1]
    data=np.array(x_variables)
    #根据索引值获取对应的样本实际值
    train_samples=data[train]
    test_samples=data[test]
    '''
    x_variables = np.array(x_variables)
    original_x = x_variables
    distance_to_average = ((x_variables - np.tile(x_variables.mean(axis=0), (x_variables.shape[0], 1))) ** 2).sum(
        axis=1)
    max_distance_sample_number = np.where(distance_to_average == np.max(distance_to_average))
    max_distance_sample_number = max_distance_sample_number[0][0]
    selected_sample_numbers = list()
    selected_sample_numbers.append(max_distance_sample_number)
    remaining_sample_numbers = np.arange(0, x_variables.shape[0], 1)
    x_variables = np.delete(x_variables, selected_sample_numbers, 0)
    remaining_sample_numbers = np.delete(remaining_sample_numbers, selected_sample_numbers, 0)
    for iteration in range(1, k):
        selected_samples = original_x[selected_sample_numbers, :]
        min_distance_to_selected_samples = list()
        for min_distance_calculation_number in range(0, x_variables.shape[0]):
            distance_to_selected_samples = ((selected_samples - np.tile(x_variables[min_distance_calculation_number, :],
                                                                        (selected_samples.shape[0], 1))) ** 2).sum(
                axis=1)
            min_distance_to_selected_samples.append(np.min(distance_to_selected_samples))
        max_distance_sample_number = np.where(
            min_distance_to_selected_samples == np.max(min_distance_to_selected_samples))
        max_distance_sample_number = max_distance_sample_number[0][0]
        selected_sample_numbers.append(remaining_sample_numbers[max_distance_sample_number])
        x_variables = np.delete(x_variables, max_distance_sample_number, 0)
        remaining_sample_numbers = np.delete(remaining_sample_numbers, max_distance_sample_number, 0)

    return selected_sample_numbers, remaining_sample_numbers

def draw_r2(y_train,p_train,y_test,p_test,title,name):
    plt.clf()
    plt.figure(500)
    #, marker='o', edgecolors='blue'
    plt.scatter(y_train,p_train,color='blue',alpha = 0.65,label="train data")
    plt.scatter(y_test,p_test,color='red',alpha = 0.65,label="test data")

    # 绘制y=x
    x_col = np.linspace(15, 32)
    plt.plot(x_col,x_col,color='black',linewidth=1,label="y = x")
    # 添加图标标签
    plt.legend(loc=2,edgecolor='black',fontsize = 11)
    plt.xlim(15,32)
    plt.ylim(15,32)
    plt.title(title)
    plt.xlabel("Actual Values(%)",fontsize = 12)
    plt.ylabel("Predictive Values(%)",fontsize = 12)
    plt.savefig('/home/liuruirui/Rapeseed/figure_result/r2_fitting/'+name+'.png')

def main(argv=None):

  
    x_train, x_test, y_train, y_test = train_test_split(data_arr, label_protein, test_size=30, shuffle=True, random_state = 22)
    draw_labelDistribute(y_train,'y_train')
    draw_labelDistribute(y_test,'y_test')
   
    for model,name in zip(model_dic,model_names):
         m = model.fit(x_train, y_train)
         p_train=m.predict(x_train)                                                                                                                                                                                                                                                                                    
         p_test=m.predict(x_test)
         print('=====',name,'=====')
         evaluating(y_train,p_train,'train')
         evaluating(y_test,p_test,'test')

    # plt.figure(500)
    # #, marker='o', edgecolors='blue'
    # plt.scatter(y_train,p_train,color='blue',alpha = 0.65,label="train data")
    # plt.scatter(y_test,p_test,color='red',alpha = 0.65,label="test data")


    # # 绘制最佳拟合线
    
    
    # x_col = np.linspace(15, 32)
    # plt.plot(x_col,x_col,color='black',linewidth=1,label="y = x")
    # # 添加图标标签
    # plt.legend(loc=2,edgecolor='black',fontsize = 14)
    # plt.xlim(15,32)
    # plt.ylim(15,32)
    # plt.xlabel("actual values(%)")
    # plt.ylabel("predictive values(%)")
    # plt.savefig('/home/liuruirui/Rapeseed/figure_result/att-r2-line.png')
    # # print('======plsr9====')
    # # evaluating(y_train,p_train,'train')
    # # evaluating(y_test,p_test,'test')
    # # draw_result_line(y_test, p_test, 'protein-plsr12-1')
    # # draw_result_line(y_train,p_train,'protein-train-plsr12')
    # print('ok')





if __name__ == "__main__":
    sys.exit(main())
