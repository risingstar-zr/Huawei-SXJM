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
DropLianXuDuan_List=[]
'''读入文件'''
def ReadFile():
    df1 = pd.read_csv('./3MidData_csv3/data3_(DropGPSV0)_163668.csv')
    f = lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")  # 把时间字符串变为时间对象
    df1["时间"] = df1["时间"].apply(f)
    return df1

#画图
def DrawPic(x,y,xlab,ylab,title):
    plt.plot(x, y)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.savefig(title+'.png')
    plt.show()

#断断续续低速行驶（最高车速小于10km/h），大于等于0km/h当做怠速处理。#GPS车速小于10的连续时间段 规律。
def GPS_V10_EveryHowLong():
    global df1
    global DropLianXuDuan_List
    df1_GPS_V10 = df1[(df1["GPS车速"]<10)&(df1["GPS车速"]>=0)]
    df1_GPS_V10["原始索引"] = df1_GPS_V10.index
    df1_GPS_V10.index = range(len(df1_GPS_V10))  # 从0开始顺序建索引
    LianXuDuan = [] #所有连续时间段[[currentLXD1],[currentLXD2],..]。
    currentLXD = ['start', 'end', 0, 0, 0]  # 开始时刻、结束时刻、记录数、原始开始索引、原始结束索引
    for (index, se) in df1_GPS_V10.iterrows():
        if index == 0:
            currentLXD[0] = df1_GPS_V10['时间'][index]
            currentLXD[1] = df1_GPS_V10['时间'][index]
            currentLXD[2] += 1
            currentLXD[3] = df1_GPS_V10['原始索引'][index]
            currentLXD[4] = df1_GPS_V10['原始索引'][index]
        else:
            # 是否和上一个连续。是，则更新当前连续段尾巴；否，收尾上个连续段，开启新连续段。
            if (df1_GPS_V10['时间'][index] - datetime.timedelta(seconds=1) == df1_GPS_V10['时间'][index - 1]):
                currentLXD[1] = df1_GPS_V10['时间'][index]
                currentLXD[4] = df1_GPS_V10['原始索引'][index]
                currentLXD[2] += 1
            else:
                LianXuDuan.append(currentLXD)
                currentLXD = [df1_GPS_V10['时间'][index], df1_GPS_V10['时间'][index], 1, df1_GPS_V10['原始索引'][index],
                              df1_GPS_V10['原始索引'][index]]
    df_LianXuDuan = pd.DataFrame(LianXuDuan, columns=['起始时刻', '结束时刻', '记录数', '原始开始索引', '原始结束索引'])

    # 画图
    DrawPic(df_LianXuDuan['起始时刻'].tolist(), df_LianXuDuan['记录数'].tolist(), '起始时刻', '持续时间(s)', 'GPS车速大于等于0小于10的连续时间段规律3')

    # 排序只是为了打印看看效果。
    # df_LianXuDuan=df_LianXuDuan.sort_values(by='记录数',ascending=False)
    # print(df_LianXuDuan)

    # 准备删除
    DropLianXuDuan_List = df_LianXuDuan[df_LianXuDuan['记录数'] > 180][['原始开始索引', '原始结束索引']].values.tolist()


#删除data1_(DropGPSV0)_178588.csv中的 GPS车速大于等于0且小于10的连续时间段大于180的时间段。
def DropYuanShiFile():
    global df1
    global DropLianXuDuan_List
    # print(DropLianXuDuan_List)
    for i in DropLianXuDuan_List:
        df1 = df1.drop(labels=range(i[0],i[0]+(i[1]-i[0]+1)-180),axis=0)
        # print(range(i[0],i[0]+(i[1]-i[0]+1)-180))


if __name__ == '__main__':
    '''下面的代码是在data3_(DropGPSV0)_163668.csv上执行的'''

    '''读入文件'''
    df1 = ReadFile()
    # print(df1)

    '''数据预处理'''
    #断断续续低速行驶（最高车速小于10km/h），大于0km/h当做怠速处理。#GPS车速小于10的连续时间段 规律。
    GPS_V10_EveryHowLong()

    # GPS车速大于等于0且小于10的连续时间段大于180的时间段。
    DropYuanShiFile()
    df1.to_csv('./3MidData_csv3/data3_(DropGPSV10)_' + str(len(df1)) + '.csv', index=False)
