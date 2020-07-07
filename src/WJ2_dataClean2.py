# -*- coding:utf-8 -*-
import datetime
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 1000)#pd的输出最大多少列
pd.set_option('display.width', 1000)#pd的输出宽度
pd.set_option('display.max_rows', 1000)#pd的输出最大多少行
pd.set_option('display.max_colwidth', 1000)#pd的输出最大列宽
import matplotlib.pyplot as plt

df1=''

'''读入文件'''
def ReadFile():
    df1 = pd.read_csv('./1Data_csv/data2.csv')
    f = lambda x: datetime.datetime.strptime(x, "%Y/%m/%d %H:%M:%S.000.") #把时间字符串变为时间对象
    df1["时间"] = df1["时间"].apply(f)
    return df1

'''单属性分析'''
def GPS_V():
    #GPS车速
    GPS_V = df1['GPS车速']
    # print('\n-----查看是否有nan的记录-----\n'); print(GPS_V[GPS_V.isna()])#没有nan
    print('\n-----均值-----\n');print(GPS_V.mean())
    print('\n-----标准差-----\n');print(GPS_V.std())
    print('\n-----最大最小值-----\n');print(GPS_V.max(), '\t', GPS_V.min())
    print('\n-----中位数-----\n'); print(GPS_V.median())
    print('\n-----四分位数-----\n');print(GPS_V.quantile([0.25, 0.5, 0.75]))
    print('\n-----偏度-----\n');print(GPS_V.skew())
    print('\n-----峰度-----\n');print(GPS_V.kurt())
    print('\n-----直方图数据-----\n');print(np.histogram(GPS_V.values))

'''数据预处理'''
# (2) 最大加速度：(100*1000)/(7*3600) = 3.96825 m/s^2      (100-0)/(7/3600)=51428.57km/h^2       ；最大减速度：7.5~8 m/s^2    (7.5/1000)/(1/(3600*3600))=97200km/h^2 ~ (8/1000)/(1/(3600*3600))=103680km/h^2
def Max_A_D():
    global df1
    df1["瞬时加速度"]=round(df1["GPS车速"].diff().shift(-1)*(5/18),4)   #变成m/s^2，保留4位小数
    # print(df1[["时间", "GPS车速", "瞬时加速度"]])
    # 遍历每行，判断下一行的时间是不是比当前大1s，如果是，什么也不做；如果不是，则把当前行的“瞬时加速度”赋NaN。
    # 遍历每行，判断当前行的加速度值是否在 -8~3.96825m/s^2之间,若是什么也不做，若不是则把当前行的“瞬时加速度”赋NaN。
    count1=0
    count2=0
    for (index,se) in df1.iterrows():
        if (index!=df1.shape[0]-1): #如果不是最后的下标
            if (df1["时间"][index]+datetime.timedelta(seconds=1)!=df1["时间"][index+1]):
                df1["瞬时加速度"][index]=np.nan
                count1+=1
            if df1["瞬时加速度"][index]<-8 or df1["瞬时加速度"][index]>3.9682:
                df1["瞬时加速度"][index] = np.nan
                count2+=1

    print(count1)#2375
    print(count2)#395



'''写到文件'''
def WriteFile(path):
    df1.to_csv(path,index=False)

if __name__ == '__main__':
    '''下面的代码是在新文件./1Data_csv/data2.csv上执行的'''

    '''读入文件'''
    df1 = ReadFile()
    # print(df1)

    '''数据预处理'''
    # (2) 最大加速度：(100*1000)/(7*3600) = 3.96825 m/s^2      (100-0)/(7/3600)=51428.57km/h^2       ；最大减速度：7.5~8 m/s^2    (7.5/1000)/(1/(3600*3600))=97200km/h^2 ~ (8/1000)/(1/(3600*3600))=103680km/h^2
    Max_A_D()
    # WriteFile('./3MidData_csv2/data2_(Max_A_D)_YESnan_' + str(df1.shape[0]) + '.csv')
    df1 = df1.dropna()
    WriteFile('./3MidData_csv2/data2_(Max_A_D)_NOnan_' + str(df1.shape[0]) + '.csv')
    # print(df1[["时间", "GPS车速", "瞬时加速度"]])
    # print(np.histogram(df1["瞬时加速度"].dropna().values))
