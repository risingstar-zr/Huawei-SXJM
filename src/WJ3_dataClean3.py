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
    df1 = pd.read_csv('./3MidData_csv3/data3_(Max_A_D)_NOnan_163668.csv')
    f = lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")  # 把时间字符串变为时间对象
    df1["时间"] = df1["时间"].apply(f)
    return df1

#GPS车速为0的连续时间段 规律。
def GPS_V0_EveryHowLong():
    global df1
    global DropLianXuDuan_List
    df1_GPS_V0=df1[df1["GPS车速"]==0]
    df1_GPS_V0["原始索引"]=df1_GPS_V0.index#
    df1_GPS_V0.index=range(len(df1_GPS_V0))#从0开始顺序建索引
    LianXuDuan=[]
    currentLXD=['start','end',0,0,0]    #开始时刻、结束时刻、记录数、原始开始索引、原始结束索引
    for (index, se) in df1_GPS_V0.iterrows():
        if index==0:
            currentLXD[0]=df1_GPS_V0['时间'][index]
            currentLXD[1]=df1_GPS_V0['时间'][index]
            currentLXD[2]+=1
            currentLXD[3]=df1_GPS_V0['原始索引'][index]
            currentLXD[4]=df1_GPS_V0['原始索引'][index]
        else:
            # 是否和上一个连续。是，则更新当前连续段尾巴；否，收尾上个连续段，开启新连续段。
            if (df1_GPS_V0['时间'][index]-datetime.timedelta(seconds=1)==df1_GPS_V0['时间'][index-1]):
                currentLXD[1] = df1_GPS_V0['时间'][index]
                currentLXD[4] = df1_GPS_V0['原始索引'][index]
                currentLXD[2] += 1
            else:
                LianXuDuan.append(currentLXD)
                currentLXD = [df1_GPS_V0['时间'][index], df1_GPS_V0['时间'][index], 1,df1_GPS_V0['原始索引'][index],df1_GPS_V0['原始索引'][index]]
    df_LianXuDuan = pd.DataFrame(LianXuDuan,columns=['起始时刻','结束时刻','记录数','原始开始索引','原始结束索引'])

    # 画图
    DrawPic(df_LianXuDuan['起始时刻'].tolist(), df_LianXuDuan['记录数'].tolist(), '起始时刻', '持续时间(s)', 'GPS车速为0的连续时间段规律3')

    # 排序只是为了看看效果。
    df_LianXuDuan=df_LianXuDuan.sort_values(by='记录数',ascending=False)
    print(df_LianXuDuan)

    # 准备删除
    DropLianXuDuan_List = df_LianXuDuan[df_LianXuDuan['记录数'] > 6000][['原始开始索引', '原始结束索引']].values.tolist()




#删除data1_(Max_A_D)_NOnan_184971.csv中的 GPS车速为0连续时间段大于180的时间段。
def DropYuanShiFile():
    global df1
    global DropLianXuDuan_List
    for i in DropLianXuDuan_List:
        df1 = df1.drop(labels=range(i[0],i[1]+1),axis=0)



#画图
def DrawPic(x,y,xlab,ylab,title):
    plt.plot(x, y)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.savefig(title+'.png')
    plt.show()


'''写到文件'''
def WriteFile(path):
    df1.to_csv(path,index=False)


if __name__ == '__main__':
    '''下面的代码是在新文件data3_(Max_A_D)_NOnan_163668.csv上执行的'''

    '''读入文件'''
    df1 = ReadFile()
    # print(df1)

    '''数据预处理'''
    #长期停车所采集的异常数据。#GPS车速为0的连续时间段 规律。
    GPS_V0_EveryHowLong()

    # 删除原始文件中的 GPS车速为0连续时间段大于180的时间段。
    DropYuanShiFile()
    df1.to_csv('./3MidData_csv3/data3_(DropGPSV0)_' + str(len(df1)) + '.csv', index=False)
    # GPS_V0_EveryHowLong()
