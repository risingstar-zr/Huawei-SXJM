# -*- coding:utf-8 -*-
import datetime
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 1000)#pd的输出最大多少列
pd.set_option('display.max_rows', 1000)#pd的输出最大多少行
pd.set_option('display.width', 1000)#pd的输出宽度
pd.set_option('display.max_colwidth', 1000)#pd的输出最大列宽
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import preprocessing

df1=''

'''1.读入文件'''
def ReadFile():#读入 合并3个文件的运动学片段特征（包含簇号）
    df1 = pd.read_csv('HeBing3GeWenJianDeYunDongXuePianDuanTeZheng(BaoHanCuHao).csv',encoding='utf-8')
    f = lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")  # 把时间字符串变为时间对象
    df1["起始时刻"] = df1["起始时刻"].apply(f)
    df1["结束时刻"] = df1["结束时刻"].apply(f)
    return df1

'''2.标准化'''
def BiaoZhunHua(data):
    global df1
    data_standardized = preprocessing.scale(data)  # 标准化    #<class 'numpy.ndarray'>
    data_standardized=pd.DataFrame(data_standardized,columns=['S运行时间','S加速时间','S减速时间','S怠速时间','S匀速时间','S运行距离','S最大速度','S平均速度','S行驶速度','S速度标准偏差','S最大加速度','S加速度段平均加速度','S最小减速度','S减速段平均减速度','S加速度标准偏差','S怠速时间比','S加速时间比','S匀速时间比','S减速时间比'])
    data_standardized = pd.concat((df1,data_standardized), axis=1)  #横着串联起来
    return data_standardized


'''3.求（所有簇）每个运动学片段的特征之和，放在DF后面。'''
def EveryMotionSeries_FeatureSum(temp):
    # print(temp)
    FeatureSum=temp.sum(axis=1) #求每行的和。返回series
    return FeatureSum

'''4.分别对每个簇内部，求每个特征的均值，无量纲化后，加起来'''
def EveryCu_Feature_AVG_Standard_Sum(temp):
    # print(len(temp['簇号'].value_counts()))
    result = list(range(len(temp['簇号'].value_counts())))#每个簇最终的结果[]。先占位。
    for i in range(len(temp['簇号'].value_counts())):
        EveryFeatureAVG = temp[temp['簇号']==i].mean()#返回Series。每列的均值
        data_standardized = preprocessing.scale(EveryFeatureAVG)#无量纲化，即标准化
        result[i]=data_standardized.sum()
    return result

'''5.计算每个簇 在曲线中 所占时间的比例，乘以1200s'''
def EveryCu_Time(temp):
    # print(temp['簇号'].value_counts())
    result_Time = list(range(len(temp['簇号'].value_counts())))  # 每个簇在曲线中占的时间[]。先占位。
    for i in range(len(temp['簇号'].value_counts())):
        result_Time[i]=round(1200*(temp[temp['簇号']==i]['运行时间'].sum()/temp['运行时间'].sum()),0)#时间就不要小数了。
    return result_Time

'''6.每个运动学片段的特征之和 - 它自己所在簇的特征计算值        再求绝对值，依然放在DF后面。然后从小到大排序作为备选工况，	直到时间大于等于5中计算的   该簇在曲线中的时长'''
#每个运动学片段的特征之和 - 它自己所在簇的特征计算值        再求绝对值,放在DF后面
def JueDuiZhi(temp,cuFeature):
    # print(temp)
    # print(cuFeature)
    result_JDZ=pd.Series()#装绝对值
    for i in range(len(temp)):
        result_JDZ.loc[i]=abs(temp['每个YDX片段的特征和'][i]-cuFeature[temp['簇号'][i]])
    return result_JDZ
#曲线由哪些运动学片段组成，返回(簇号,起始时刻,结束时刻,时间,片段特征减去簇特征的绝对值)
def Choose_YDXSeries(temp,EveryCu_Time):
    # print(temp)
    # print(EveryCu_Time) #每个簇在曲线中的时间
    QuXianZuChengIndex = [] #[(簇号,起始时刻,结束时刻,时间,片段特征减去簇特征的绝对值),(),()...]
    # for i in range(len(EveryCu_Time)):#遍历每个簇，挑它们的课代表。结束条件是大于等于当前簇要求的时间。按照0、2、3、1的顺序
    for i in (0,2,3,1):
        times=0
        kuochongTimes=100#弹性的100秒
        current_Cu=temp[temp['簇号']==i].sort_values(by='片段特征减去簇特征的绝对值',ascending=True)#当前簇
        for j, value in current_Cu.iterrows():
            if times>=EveryCu_Time[i]:  #时间已经够了
                break
            # if (times+current_Cu['运行时间'][j]-EveryCu_Time[i])>25: continue #不让这个簇超过太多时间
            if(times+current_Cu['运行时间'][j]-EveryCu_Time[i])>kuochongTimes: continue #当前加入的 片段 超时时 必须要在弹性100秒内
            if(times+current_Cu['运行时间'][j]-EveryCu_Time[i])<=kuochongTimes and (times+current_Cu['运行时间'][j]-EveryCu_Time[i])>0: kuochongTimes-=(times+current_Cu['运行时间'][j]-EveryCu_Time[i])
            times+=current_Cu['运行时间'][j]
            QuXianZuChengIndex.append([current_Cu['簇号'][j],current_Cu['起始时刻'][j],current_Cu['结束时刻'][j],current_Cu['运行时间'][j],current_Cu['片段特征减去簇特征的绝对值'][j]])
    QuXianZuChengIndex=pd.DataFrame(QuXianZuChengIndex,columns=['簇号','起始时刻','结束时刻','运行时间','片段特征减去簇特征的绝对值'])
    return QuXianZuChengIndex

'''7.画速度曲线图'''
def DrawVT(QuXianZuCheng_index):
    # print(QuXianZuCheng_index)#簇号  起始时刻  结束时刻  运行时间  片段特征减去簇特征的绝对值
    #读 预处理最后一步 三个文件的合集data123_(DropGPSV10)_474048.csv
    data123 = pd.read_csv('data123_(DropGPSV10)_474048.csv', encoding='utf-8')
    f = lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")  # 把时间字符串变为时间对象
    data123["时间"] = data123["时间"].apply(f)

    tempDf=pd.DataFrame()#把曲线 里的运动学片段 对应时间的数据 都保存起来。
    for i, value in QuXianZuCheng_index.iterrows():#遍历曲线 里的运动学片段
        # print(QuXianZuCheng_index[(QuXianZuCheng_index['起始时刻']>datetime.datetime.strptime('2017-12-19 22:17:11', "%Y-%m-%d %H:%M:%S"))])#datatime可以直接比较大小
        startTime=value['起始时刻']
        endTime=value['结束时刻']
        tempDf = pd.concat((tempDf, data123[(data123['时间'] >= startTime) & (data123['时间'] <= endTime)]), axis=0)  # 竖着串联起来
        # y.extend(data123[(data123['时间']>=startTime)&(data123['时间']<=endTime)]['GPS车速'])

    # print(tempDf)
    # tempDf=tempDf.sort_values(by='时间')
    #图
    plt.plot(list(tempDf['GPS车速']))
    plt.xlabel('时间(s)')
    plt.ylabel('速度(km/h)')
    plt.title('汽车行驶工况')
    plt.savefig('汽车行驶工况.png')
    plt.show()

'''评估'''
def PG(QuXian_YDXPD):
    # print(df1)
    print('=========')
    # print(QuXian_YDXPD)
    #工况曲线 的运动学片段 特征们
    QuXian_YDXPD_Feature=pd.DataFrame()
    for i, value in QuXian_YDXPD.iterrows():
        # print(df1[df1['起始时刻']==value['起始时刻']])
        QuXian_YDXPD_Feature=pd.concat((QuXian_YDXPD_Feature,df1[df1['起始时刻']==value['起始时刻']]))
    # print(QuXian_YDXPD_Feature)
    #曲线的特征。Series格式
    QuXian_YDXPD_Feature_Seri=QuXian_YDXPD_Feature.loc[:,['运行时间','加速时间','减速时间','怠速时间','匀速时间','运行距离','最大速度','平均速度','行驶速度','速度标准偏差','最大加速度','加速度段平均加速度','最小减速度','减速段平均减速度','加速度标准偏差','怠速时间比','加速时间比','匀速时间比','减速时间比']].mean()
    # print(QuXian_YDXPD_Feature_Seri)
    #实际数据的特征。series格式
    df1_Feature_Seri = df1.loc[:,['运行时间', '加速时间', '减速时间', '怠速时间', '匀速时间', '运行距离', '最大速度', '平均速度', '行驶速度', '速度标准偏差', '最大加速度','加速度段平均加速度', '最小减速度', '减速段平均减速度', '加速度标准偏差', '怠速时间比', '加速时间比', '匀速时间比', '减速时间比']].mean()
    # print(df1_Feature_Seri)
    #误差率=（实际数据的特征-曲线的特征）/实际数据的特征
    print((df1_Feature_Seri-QuXian_YDXPD_Feature_Seri)/df1_Feature_Seri)

if __name__ == '__main__':
    '''1.读入文件'''
    df1 = ReadFile()
    # print(df1)
    # print('======')

    '''2.所有特征无量纲化， 即标准化'''
    df1_BZH = BiaoZhunHua(df1.loc[:,['运行时间','加速时间','减速时间','怠速时间','匀速时间','运行距离','最大速度','平均速度','行驶速度','速度标准偏差','最大加速度','加速度段平均加速度','最小减速度','减速段平均减速度','加速度标准偏差','怠速时间比','加速时间比','匀速时间比','减速时间比']])
    # print(df1.loc[:,['运行时间','加速时间','减速时间','怠速时间','匀速时间','运行距离','最大速度','平均速度','行驶速度','速度标准偏差','最大加速度','加速度段平均加速度','最小减速度','减速段平均减速度','加速度标准偏差','怠速时间比','加速时间比','匀速时间比','减速时间比']])
    # print(df1_BZH.loc[:,['S运行时间','S加速时间','S减速时间','S怠速时间','S匀速时间','S运行距离','S最大速度','S平均速度','S行驶速度','S速度标准偏差','S最大加速度','S加速度段平均加速度','S最小减速度','S减速段平均减速度','S加速度标准偏差','S怠速时间比','S加速时间比','S匀速时间比','S减速时间比']])

    '''3.求（所有簇）每个运动学片段的特征之和，放在DF后面。'''
    df1_BZH['每个YDX片段的特征和']=EveryMotionSeries_FeatureSum(df1_BZH.loc[:,['S运行时间','S加速时间','S减速时间','S怠速时间','S匀速时间','S运行距离','S最大速度','S平均速度','S行驶速度','S速度标准偏差','S最大加速度','S加速度段平均加速度','S最小减速度','S减速段平均减速度','S加速度标准偏差','S怠速时间比','S加速时间比','S匀速时间比','S减速时间比']])
    # print(df1_BZH)

    '''4.分别对每个簇内部，求每个特征的均值，无量纲化后，加起来'''
    EveryCu_Feature_Sum = EveryCu_Feature_AVG_Standard_Sum(df1_BZH.loc[:,['S运行时间','S加速时间','S减速时间','S怠速时间','S匀速时间','S运行距离','S最大速度','S平均速度','S行驶速度','S速度标准偏差','S最大加速度','S加速度段平均加速度','S最小减速度','S减速段平均减速度','S加速度标准偏差','S怠速时间比','S加速时间比','S匀速时间比','S减速时间比','簇号']])
    # print(EveryCu_Feature_Sum)    #顺序与簇号 一样0、1、2、3。每个簇的 特征求和

    '''5.计算每个簇 在曲线中 所占时间的比例，乘以1200s'''
    EveryCuTime = EveryCu_Time(df1_BZH.loc[:,['簇号','运行时间']])
    # print(EveryCuTime)  #顺序与簇号 一样0、1、2、3。每个簇 在曲线中所占时间长度

    '''6.每个运动学片段的特征之和 - 它自己所在簇的特征计算值        再求绝对值，依然放在DF后面。然后从小到大排序作为备选工况，	直到时间大于等于5中计算的   该簇在曲线中的时长'''
    #计算 每个运动学片段的特征之和 - 它自己所在簇的特征计算值  的绝对值
    df1_BZH['片段特征减去簇特征的绝对值'] = JueDuiZhi(df1_BZH.loc[:,['簇号','每个YDX片段的特征和']],EveryCu_Feature_Sum)
    # print(df1_BZH)
    #曲线由哪些运动学片段组成，返回[(簇号,起始时刻,结束时刻,时间,片段特征减去簇特征的绝对值),(),()...]
    QuXianZuCheng_index = Choose_YDXSeries(df1_BZH.loc[:,['簇号','起始时刻','结束时刻','运行时间','片段特征减去簇特征的绝对值']],EveryCuTime)
    # print(EveryCuTime)
    # print('曲线组成：');print(QuXianZuCheng_index)
    # print(QuXianZuCheng_index['运行时间'].sum())

    '''7.画速度曲线图'''
    # DrawVT(QuXianZuCheng_index)


    '''评估'''
    PG(QuXianZuCheng_index)


