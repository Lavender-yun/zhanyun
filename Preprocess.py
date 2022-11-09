from copy import deepcopy
import sys
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import signal
from sklearn.linear_model import LinearRegression
import pywt


# 最大最小值归一化
def MMS(data):
    return MinMaxScaler().fit_transform(data)


# 标准化
def SS(data):
    return StandardScaler().fit_transform(data)


# 均值中心化
def MC(data):
    for i in range(data.shape[0]):
        MEAN = np.mean(data[i])
        data[i] = data[i] - MEAN
    return data


# 标准正态变换
def SNV(data):
    m = data.shape[0]
    n = data.shape[1]
    print(m, n)  #
    # 求标准差
    data_std = np.std(data, axis=1)  # 每条光谱的标准差
    # 求平均值
    data_average = np.mean(data, axis=1)  # 每条光谱的平均值
    # SNV计算
    data_snv = [[((data[i][j] - data_average[i]) / data_std[i])
                 for j in range(n)] for i in range(m)]
    return data_snv


'''
移动窗口平滑、SavitzkyGolay 消除随机噪声，
基本思想是对指定宽度为 2𝑛 + 1 个点的“窗口”内各点的数据进行重现拟合，
使其相邻数据点之间更加平滑。
其中 S-G 平滑是基于最小二乘原理提出的卷积平滑方法，在NIRS数据预处理应用比较广泛。
'''
# 移动平均平滑


def MA(a, WSZ=21): # wsz=11
    for i in range(a.shape[0]):
        out0 = np.convolve(a[i], np.ones(WSZ, dtype=int),
                           'valid') / WSZ  # WSZ是窗口宽度，是奇数
        r = np.arange(1, WSZ - 1, 2)
        start = np.cumsum(a[i, :WSZ - 1])[::2] / r
        stop = (np.cumsum(a[i, :-WSZ:-1])[::2] / r)[::-1]
        a[i] = np.concatenate((start, out0, stop))
    return a


# Savitzky-Golay平滑滤波
def SG(data, w=21, p=3):  # wsz=11,p=2
    return signal.savgol_filter(data, w, p)


# 一阶导数
def D1(data):
    n, p = data.shape
    Di = np.ones((n, p - 1))
    for i in range(n):
        Di[i] = np.diff(data[i])
    return Di


# 二阶导数
def D2(data):
    n, p = data.shape
    Di = np.ones((n, p - 2))
    for i in range(n):
        Di[i] = np.diff(np.diff(data[i]))
    return Di


# 趋势校正(DT)
def DT(data):
    lenth = data.shape[1]
    x = np.asarray(range(lenth), dtype=np.float32)
    out = np.array(data)
    l = LinearRegression()
    for i in range(out.shape[0]):
        l.fit(x.reshape(-1, 1), out[i].reshape(-1, 1))
        k = l.coef_
        b = l.intercept_
        for j in range(out.shape[1]):
            out[i][j] = out[i][j] - (j * k + b)
    return out

# 多元散射校正
# MSC(数据)
def MSC(Data):
    # 计算平均光谱
    n, p = Data.shape
    msc = np.ones((n, p))

    for j in range(n):
        mean = np.mean(Data, axis=0)

    # 线性拟合
    for i in range(n):
        y = Data[i, :]
        l = LinearRegression()
        l.fit(mean.reshape(-1, 1), y.reshape(-1, 1))
        k = l.coef_
        b = l.intercept_
        msc[i, :] = (y - b) / k
    return msc

def wave(data):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :return: data after wave :(n_samples, n_features)
    """
    data = deepcopy(data)
    if isinstance(data, pd.DataFrame):
        data = data.values
    def wave_(data):
        w = pywt.Wavelet('db8')  # 选用Daubechies8小波
        maxlev = pywt.dwt_max_level(len(data), w.dec_len)
        coeffs = pywt.wavedec(data, 'db8', level=maxlev)
        threshold = 0.04
        for i in range(1, len(coeffs)):
            coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
        datarec = pywt.waverec(coeffs, 'db8')
        return datarec

    tmp = None
    for i in range(data.shape[0]):
        if (i == 0):
            tmp = wave_(data[i])
        else:
            tmp = np.vstack((tmp, wave_(data[i])))

    return tmp

def draw_spec(data_arr, method):
    plt.figure(500)
    x_col = np.linspace(10000, 4000, num=data_arr.shape[1])
    # x_col = wave_arr
    y_col = np.transpose(data_arr)  # 数组逆序np.transpose
    plt.plot(x_col, y_col)
    plt.xlim(10000, 4000)
    plt.xlabel("Wavenumber($cm^-1$)")
    plt.ylabel('Absorbance')

    plt.title("The spectrum of the " + method + " for dataset",
              fontweight="semibold", fontsize='large')  # 记得改名字MSC

    plt.savefig('/home/liuruirui/NIRS/local/data/wheat/'+method+'.png', dpi=600, format='png')

def draw_wspec(y, method):
    # x = np.array(x)
    y = np.array(y)
    plt.figure(500)
    x_col = np.linspace(730, 1100, num=y.shape[1])
    # x_col = x
    y_col = np.transpose(y)  # 数组逆序np.transpose
    plt.plot(x_col, y_col)
    plt.xlim(730, 1100)
    plt.xlabel("Wavenumber($nm$)")
    plt.ylabel('Absorbance')

    plt.title("The spectrum of the " + method + " for dataset",
              fontweight="semibold", fontsize='large')  # 记得改名字MSC

    plt.savefig('/home/liuruirui/NIRS/local/data/wheat/'+method+'.png', dpi=600, format='png')

def main(argv=None):

  
    dataSG=SG(data_arr)
    np.savetxt('/home/liuruirui/NIRS/local/data/dataSG_D1.csv', dataSG_D1, delimiter=',')

    # SNV = pd.read_csv(r'/home/liuruirui/NIRS/local/data/dataSNV.csv', header=None)   
    # SNV = np.array(SNV)
    # draw_spec(SNV, 'SNV')

    # MSC = pd.read_csv(r'/home/liuruirui/NIRS/local/data/dataMSC.csv', header=None)   
    # MSC = np.array(MSC)
    # draw_spec(MSC, 'MSC')   


    MA1 = MA(wdata, 11)
    draw_wspec(MA1, 'MA')  

    np.savetxt('/home/liuruirui/NIRS/local/data/wheat/wMA.csv', MA1, delimiter=',' )

    DT1 = DT(wdata)
    draw_wspec(DT1, 'DT')  

    np.savetxt('/home/liuruirui/NIRS/local/data/wheat/wDT.csv', DT1, delimiter=',' )
    


    # data1 = pd.read_excel(r'/home/liuruirui/NIRS/local/data/raw_data_all.xlsx', header=None)
    # data = data1.iloc[1:, ]
    # data_arr = np.array(data)  # 波长对应数值
    # x_data_wave = wave(data_arr)
    # np.savetxt('/home/liuruirui/NIRS/local/data/dataWave.csv', x_data_wave, delimiter=',')

    # draw_spec(data_arr, 'DT')
    # x_data_SG = SG(wdata)
    # np.savetxt('/home/liuruirui/NIRS/local/data/wheat/wSG.csv', x_data_SG, delimiter=',')
    # x_data_MC = MC(wdata)
    # np.savetxt('/home/liuruirui/NIRS/local/data/wheat/wMC.csv', x_data_MC, delimiter=',')
    # x_data_SNV = SNV(wdata)
    # np.savetxt('/home/liuruirui/NIRS/local/data/wheat/wSNV.csv', x_data_SNV, delimiter=',')
    # x_data_D1 = D1(wdata)
    # np.savetxt('/home/liuruirui/NIRS/local/data/wheat/wD1.csv', x_data_D1, delimiter=',')
    # x_data_DT = DT(wdata)
    # np.savetxt('/home/liuruirui/NIRS/local/data/wDT.csv', x_data_DT, delimiter=',')
    # x_data_MSC = MSC(wdata)
    # np.savetxt('/home/liuruirui/NIRS/local/data/wheat/wMSC.csv', x_data_MSC, delimiter=',')

    
    
   
    print('success')


if __name__ == "__main__":
    sys.exit(main())