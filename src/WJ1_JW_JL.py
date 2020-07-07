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
JW_result=''
filepath='./3MidData_csv/MotionSeriesFeature_1103_(80%).csv'
filepath2='./3MidData_csv2/MotionSeriesFeature_822_(80%).csv'
filepath3='./3MidData_csv3/MotionSeriesFeature_786_(80%).csv'

#每个簇的特征。
#['总运行时间','平均运行时间','总加速时间','平均加速时间','总减速时间','平均减速时间','总怠速时间','平均怠速','总匀速时间','平均匀速时间','总运行距离','平均运行距离','最大速度','平均速度','行驶速度','速度标准偏差','最大加速度','加速度段平均加速度','最小减速度','减速段平均减速度','加速度标准偏差','最大怠速时间比','平均怠速时间比','最小怠速时间比','最大加速时间比','平均加速时间比','最小加速时间比','最大匀速时间比','平均匀速时间比','最小匀速时间比','最大减速时间比','平均减速时间比','最小减速时间比']
EveryCu_Feat=[]

'''读入文件'''
def ReadFile():
    df1 = pd.read_csv(filepath)
    f = lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")  # 把时间字符串变为时间对象
    df1["起始时刻"] = df1["起始时刻"].apply(f)
    df1["结束时刻"] = df1["结束时刻"].apply(f)

    df2 = pd.read_csv(filepath2)
    # f = lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")  # 把时间字符串变为时间对象
    df2["起始时刻"] = df2["起始时刻"].apply(f)
    df2["结束时刻"] = df2["结束时刻"].apply(f)

    df3 = pd.read_csv(filepath3)
    # f = lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")  # 把时间字符串变为时间对象
    df3["起始时刻"] = df3["起始时刻"].apply(f)
    df3["结束时刻"] = df3["结束时刻"].apply(f)

    df1 = pd.concat((df1, df2, df3))#串联
    df1.index=range(len(df1))#串联时，把索引也跟着串联了，所以重建索引
    return df1

'''降维'''
def JW():
    global JW_result
    data_standardized = preprocessing.scale(df1)#标准化
    pca = PCA(n_components=0.85)  # 加载PCA算法，若n_components设置为整数则设置降维后主成分数个数。若设置小数则保证降维后的数据保持90%的信息
    JW_result = pca.fit_transform(data_standardized)  # 对原始数据降维，保存在result中
    print('降维结果形状 ',JW_result.shape)

'''聚类'''
def JL():
    from sklearn import preprocessing
    # 正则化
    min_max_scaler = preprocessing.MinMaxScaler()
    X_norm = min_max_scaler.fit_transform(JW_result)

    km=KMeans(n_clusters=4,random_state=0)#
    label = km.fit_predict(X_norm)   #label聚类后各数据所属的标签。fit_predict计算簇中心以及簇分配序号
    # print(type(label))#numpy.ndarray
    label=pd.Series(label)
    # print(label)
    # print("所有样本距离所属簇中心点的总距离和为:%.5f" % km.inertia_)  # 越小越好

    # print("所有的中心点聚类中心坐标:")
    # cluter_centers = km.cluster_centers_  # 所有的中心点聚类中心坐标:
    # print(cluter_centers)


    # print('越大越好1',metrics.silhouette_score(X_norm, km.labels_, metric='euclidean'))
    # print('越大越好2',metrics.calinski_harabaz_score(X_norm, km.labels_))
    return label

'''聚类分析'''
def JL_FenXi():
    from sklearn import preprocessing
    # 正则化
    min_max_scaler = preprocessing.MinMaxScaler()
    X_norm = min_max_scaler.fit_transform(JW_result)

    # kmeans聚类
    # inertia样本到最近的聚类中心的距离总和
    # 肘部法则
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    distortions = []
    for i in range(1, 40):
        km = KMeans(n_clusters=i,
                    init='k-means++',
                    n_init=10,
                    max_iter=300,
                    random_state=0)
        km.fit(X_norm)
        distortions.append(km.inertia_)
        print('分成',i,'类，对应的距离总和（越小越好）：',km.inertia_)
    #画图
    plt.plot(range(1, 40), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()#inertia样本到最近的聚类中心的距离总和。越小越好


    print('-----------------------------------')
    #轮廓系数
    from sklearn import metrics
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    scores = []
    for i in range(2, 100):
        km = KMeans(        n_clusters=i,
                            init='k-means++',
                            n_init=10,
                            max_iter=300,
                            random_state=0      )
        km.fit(X_norm)
        scores.append(metrics.silhouette_score(X_norm, km.labels_ , metric='euclidean'))
        print('分成', i, '类，轮廓系数（越大越好）：', metrics.silhouette_score(X_norm, km.labels_ , metric='euclidean'))
    plt.plot(range(2,100), scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('silhouette_score')
    plt.show()  #越大越好。[-1,1]

    
    print('-----------------------------------')
    #Calinski-Harabaz Index。这种评估也同时考虑了族内族外的因素。类别内部数据的协方差越小越好，类别之间的协方差越大越好
    from sklearn import metrics
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    ch_scores = []
    for i in range(2, 100):
        km = KMeans(n_clusters=i,
                    init='k-means++',
                    n_init=10,
                    max_iter=300,
                    random_state=0)
        km.fit(X_norm)
        ch_scores.append(metrics.calinski_harabaz_score(X_norm, km.labels_))
        print('分成', i, '类，Calinski-Harabaz Index（不知道大好还是小好）：', metrics.calinski_harabaz_score(X_norm, km.labels_))
    plt.plot(range(2, 100), ch_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('calinski_harabaz_score')
    plt.show()


'''求每个簇的运动片段合体的特征'''
#['总运行时间','平均运行时间','总加速时间','平均加速时间','总减速时间','平均减速时间','总怠速时间','平均怠速','总匀速时间','平均匀速时间','总运行距离','平均运行距离','最大速度','平均速度','行驶速度','速度标准偏差','最大加速度','加速度段平均加速度','最小减速度','减速段平均减速度','加速度标准偏差','最大怠速时间比','平均怠速时间比','最小怠速时间比','最大加速时间比','平均加速时间比','最小加速时间比','最大匀速时间比','平均匀速时间比','最小匀速时间比','最大减速时间比','平均减速时间比','最小减速时间比']
def EveryCu_Feature():
    global EveryCu_Feat
    Cu_Num=len(df1['簇号'].value_counts())    #簇的个数
    for i in range(Cu_Num):#遍历每个簇。
        df1_temp=df1[df1['簇号']==i]#有这些：[起始时刻 结束时刻 原始开始索引 原始结束索引 运行时间 加速时间 减速时间 怠速时间 匀速时间 运行距离 最大速度 平均速度 行驶速度 速度标准偏差 最大加速度 加速度段平均加速度 最小减速度 减速段平均减速度 加速度标准偏差 怠速时间比 加速时间比 匀速时间比 减速时间比 簇号]
        # print(df1_temp.shape)
        current_Cu_Feature=list(range(0,33))  #[]33个数字占位#当前簇的特征
        current_Cu_Feature[0] =df1_temp['运行时间'].sum()#总运行时间
        current_Cu_Feature[1] =df1_temp['运行时间'].mean()#平均运行时间
        current_Cu_Feature[2] =df1_temp['加速时间'].sum()#总加速时间
        current_Cu_Feature[3] =df1_temp['加速时间'].mean()#平均加速时间
        current_Cu_Feature[4] =df1_temp['减速时间'].sum()#总减速时间
        current_Cu_Feature[5] =df1_temp['减速时间'].mean()#平均减速时间
        current_Cu_Feature[6] =df1_temp['怠速时间'].sum()#总怠速时间
        current_Cu_Feature[7] =df1_temp['怠速时间'].mean()#平均怠速
        current_Cu_Feature[8] =df1_temp['匀速时间'].sum()#总匀速时间
        current_Cu_Feature[9] =df1_temp['匀速时间'].mean()#平均匀速时间
        current_Cu_Feature[10] =df1_temp['运行距离'].sum()#总运行距离
        current_Cu_Feature[11] =df1_temp['运行距离'].mean()#平均运行距离
        current_Cu_Feature[12] =df1_temp['最大速度'].max()#最大速度
        current_Cu_Feature[13] =df1_temp['平均速度'].mean()#平均速度
        current_Cu_Feature[14] =df1_temp['行驶速度'].mean()#行驶速度
        current_Cu_Feature[15] =df1_temp['速度标准偏差'].mean()#速度标准偏差
        current_Cu_Feature[16] =df1_temp['最大加速度'].max()#最大加速度
        current_Cu_Feature[17] =df1_temp['加速度段平均加速度'].mean()#加速度段平均加速度
        current_Cu_Feature[18] =df1_temp['最小减速度'].min()#最小减速度
        current_Cu_Feature[19] =df1_temp['减速段平均减速度'].mean()#减速段平均减速度
        current_Cu_Feature[20] =df1_temp['加速度标准偏差'].mean()#加速度标准偏差
        current_Cu_Feature[21] =df1_temp['怠速时间比'].max()#最大怠速时间比
        current_Cu_Feature[22] =df1_temp['怠速时间比'].mean()#平均怠速时间比
        current_Cu_Feature[23] =df1_temp['怠速时间比'].min()#最小怠速时间比
        current_Cu_Feature[24] =df1_temp['加速时间比'].max()#最大加速时间比
        current_Cu_Feature[25] =df1_temp['加速时间比'].mean()#平均加速时间比
        current_Cu_Feature[26] =df1_temp['加速时间比'].min()#最小加速时间比
        current_Cu_Feature[27] =df1_temp['匀速时间比'].max()#最大匀速时间比
        current_Cu_Feature[28] =df1_temp['匀速时间比'].mean()#平均匀速时间比
        current_Cu_Feature[29] =df1_temp['匀速时间比'].min()#最小匀速时间比
        current_Cu_Feature[30] =df1_temp['减速时间比'].max()#最大减速时间比
        current_Cu_Feature[31] =df1_temp['减速时间比'].mean()#平均减速时间比
        current_Cu_Feature[32] =df1_temp['减速时间比'].min()#最小减速时间比
        EveryCu_Feat.append(current_Cu_Feature)
    EveryCu_Feat=pd.DataFrame(EveryCu_Feat,columns=['总运行时间','平均运行时间','总加速时间','平均加速时间','总减速时间','平均减速时间','总怠速时间','平均怠速','总匀速时间','平均匀速时间','总运行距离','平均运行距离','最大速度','平均速度','行驶速度','速度标准偏差','最大加速度','加速度段平均加速度','最小减速度','减速段平均减速度','加速度标准偏差','最大怠速时间比','平均怠速时间比','最小怠速时间比','最大加速时间比','平均加速时间比','最小加速时间比','最大匀速时间比','平均匀速时间比','最小匀速时间比','最大减速时间比','平均减速时间比','最小减速时间比'])
    # 排序是为了打印查看。
    # EveryCu_Feat = EveryCu_Feat.sort_values(by='总运行时间', ascending=False)
    # print(MotionSeriesFeature_List)







if __name__ == '__main__':
    '''读入文件'''
    df1 = ReadFile()
    # print(df1)
    print('合并3个文件的运动学片段特征.csv', df1.shape)
    df1.to_csv('合并3个文件的运动学片段特征.csv', index=False)

    '''降维'''
    # ['起始时刻','结束时刻','原始开始索引','原始结束索引','运行时间','加速时间','减速时间','怠速时间','匀速时间','运行距离','最大速度','平均速度','行驶速度','速度标准偏差','最大加速度','加速度段平均加速度','最小减速度','减速段平均减速度','加速度标准偏差','怠速时间比','加速时间比','匀速时间比','减速时间比']
    df1_copy=df1
    df1=df1.iloc[:,4:18]#只要后面的特征[:,4:18]
    JW()#降维
    df1=df1_copy


    '''聚类分析'''
    # JL_FenXi()#用for循环 分出很多类，看每一类的得分，确定最终分为几类。

    '''聚类'''
    label = JL()    #返回series。每行记录的簇号。
    df1['簇号']=label
    # print(label.value_counts())
    # print(df1['簇号'].value_counts())

    # print(df1.shape)
    df1.to_csv('HeBing3GeWenJianDeYunDongXuePianDuanTeZheng(BaoHanCuHao).csv', index=False)

    '''计算每个簇的特征'''
    #['运行时间','加速时间','减速时间','怠速时间','匀速时间','运行距离','最大速度','平均速度','行驶速度','速度标准偏差','最大加速度','加速度段平均加速度','最小减速度','减速段平均减速度','加速度标准偏差','怠速时间比','加速时间比','匀速时间比','减速时间比']
    #['总运行时间','平均运行时间','总加速时间','平均加速时间','总减速时间','平均减速时间','总怠速时间','平均怠速','总匀速时间','平均匀速时间','总运行距离','平均运行距离','最大速度','平均速度','行驶速度','速度标准偏差','最大加速度','加速度段平均加速度','最小减速度','减速段平均减速度','加速度标准偏差','最大怠速时间比','平均怠速时间比','最小怠速时间比','最大加速时间比','平均加速时间比','最小加速时间比','最大匀速时间比','平均匀速时间比','最小匀速时间比','最大减速时间比','平均减速时间比','最小减速时间比']
    EveryCu_Feature()
    print(len(EveryCu_Feat),'个簇的特征：')
    print(EveryCu_Feat)
    EveryCu_Feat.to_csv(str(len(EveryCu_Feat))+'个簇的特征.csv',index=False)

