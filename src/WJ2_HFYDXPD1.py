# -*- coding:utf-8 -*-
import datetime
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 1000)#pd的输出最大多少列
pd.set_option('display.max_rows', 1000)#pd的输出最大多少行
pd.set_option('display.width', 1000)#pd的输出宽度
pd.set_option('display.max_colwidth', 1000)#pd的输出最大列宽
import matplotlib.pyplot as plt
import math

df1=''
LianXuMotionDuan_List=[]#运动学片段们的 开始、结束、记录数等等。[[运动学片段1]，[运动学片段2]，...]
MotionSeriesFeature_List=[]#运动学片段们的特征。[[1号运动学片段特征],[2号运动学片段特征],...]

'''读入文件'''
def ReadFile():
    df1 = pd.read_csv('./3MidData_csv2/data2_(DropGPSV10)_138599.csv')
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


#划分运动学片段。
# 返回每个运动学片段的['起始时刻', '结束时刻', '记录数', '原始开始索引', '原始结束索引','总秒数' ,'怠速秒数','加速秒数', '匀速秒数' ,'减速秒数']
def HFMotionSeries():
    global df1
    global LianXuMotionDuan_List
    currentLXD = ['start', 'end', 0, 0, 0, 0, 0, 0, 0, 0]  # ['起始时刻', '结束时刻', '记录数', '原始开始索引', '原始结束索引','总秒数' ,'怠速秒数','加速秒数', '匀速秒数' ,'减速秒数']
    StartLingTime_List = []#记录每个前导0的[时刻，原始开始索引]。
    yudaoFeiLingFlag=False
    for (index, se) in df1.iterrows():
        # print('正在执行'+str(index))
        curr_Time=df1['时间'][index]       # 当前行的时间
        curr_GPSV=df1['GPS车速'][index]   # 当前行的车速
        # curr_A = df1['瞬时加速度'][index]  # 当前行的瞬时加速度
        if curr_GPSV==0:#车速为0。若是开头的0，则记录前导0；若是结尾的0则收尾、开启新的片段。
            if yudaoFeiLingFlag == False:  # 是开头的0
                #记录前导0的[时刻，原始开始索引]
                temp = [curr_Time,index]
                StartLingTime_List.append(temp)
            else:  # 是收尾的0
                currentLXD[1] = curr_Time
                currentLXD[2] += 1
                currentLXD[4] = index
                currentLXD[5] += 1
                t = PDState(se)
                currentLXD[t] += 1
                LianXuMotionDuan_List.append(currentLXD)
                #开启新片段
                yudaoFeiLingFlag = False
                StartLingTime_List=[]
                currentLXD = ['start', 'end', 0, 0, 0, 0, 0, 0, 0, 0]
                temp = [curr_Time, index]
                StartLingTime_List.append(temp)
        else:  # 车速不为0
            # 若是从前导0刚转入车速不为0 。则确定前导0的开始、结束
            # 然后登记当前行。
            if yudaoFeiLingFlag==False:
                yudaoFeiLingFlag = True
                for i in range(len(StartLingTime_List)):
                    if (StartLingTime_List[-1][0]-StartLingTime_List[i][0]).total_seconds()<180:
                        # print((StartLingTime_List[-1][0]-StartLingTime_List[i][0]).total_seconds())
                        currentLXD[0]=StartLingTime_List[i][0]#前导0的 起始时刻。
                        # currentLXD[1]=StartLingTime_List[-1][0]#结束时刻。
                        currentLXD[2]+=len(StartLingTime_List)-i    #记录数
                        currentLXD[3]=StartLingTime_List[i][1]#前导0的 原始开始索引。
                        # currentLXD[4] = StartLingTime_List[-1][1]  #原始结束索引。
                        currentLXD[5]+=len(StartLingTime_List)-i    #总秒数
                        currentLXD[6]+=len(StartLingTime_List)-i-1  #前导0 不包含末尾0。
                        t = PDState(df1.iloc[StartLingTime_List[-1][1]])#判断前导0的最后一个是否 是怠速 还是加速 状态。
                        currentLXD[t] += 1
                        break
            currentLXD[1]=curr_Time
            currentLXD[2]+=1
            currentLXD[4] = index
            currentLXD[5] += 1
            t = PDState(se)
            currentLXD[t] += 1

    LianXuMotionDuan_List = pd.DataFrame(LianXuMotionDuan_List,columns=['起始时刻', '结束时刻', '记录数', '原始开始索引', '原始结束索引', '总秒数', '怠速秒数', '加速秒数', '匀速秒数', '减速秒数'])
    # 排序是为了打印查看。
    # LianXuMotionDuan_List = LianXuMotionDuan_List.sort_values(by='总秒数', ascending=False)
    # print(LianXuMotionDuan_List)



#筛选运动学片段。
# 删除掉不符合要求的运动学片段。如总时间小于20s；运动学片段中包含300s以上的间隔时间。
def ChooseMotionSeries():
    global LianXuMotionDuan_List
    # print((LianXuMotionDuan_List['结束时刻']-LianXuMotionDuan_List['起始时刻']).map(lambda x:x.total_seconds()))
    # print(((LianXuMotionDuan_List['结束时刻']-LianXuMotionDuan_List['起始时刻']).map(lambda x:x.total_seconds()+1)-LianXuMotionDuan_List['总秒数'])<300)
    f1=((LianXuMotionDuan_List['结束时刻']-LianXuMotionDuan_List['起始时刻']).map(lambda x:x.total_seconds()+1)-LianXuMotionDuan_List['总秒数'])<300
    # print(f1.value_counts())
    f2=LianXuMotionDuan_List['总秒数']>20
    # print(f2.value_counts())
    f3=(LianXuMotionDuan_List['怠速秒数']/LianXuMotionDuan_List['总秒数'])<=0.8#怠速占总时间 不超过80%
    # print(LianXuMotionDuan_List[f1&f2])
    LianXuMotionDuan_List=LianXuMotionDuan_List[f1 & f2 & f3]
    LianXuMotionDuan_List.index = range(len(LianXuMotionDuan_List))  # 从0开始顺序建索引

#计算两个经纬度间的距离
#(纬度1，经度1，纬度2，经度2)
def cal_dis(latitude1, longitude1,latitude2, longitude2):
	latitude1 = (math.pi/180.0)*latitude1
	latitude2 = (math.pi/180.0)*latitude2
	longitude1 = (math.pi/180.0)*longitude1
	longitude2= (math.pi/180.0)*longitude2
	#因此AB两点的球面距离为:{arccos[sina*sinx+cosb*cosx*cos(b-y)]}*R  (a,b,x,y)
	#地球半径
	R = 6378.1
	temp=math.sin(latitude1)*math.sin(latitude2)+\
		 math.cos(latitude1)*math.cos(latitude2)*math.cos(longitude2-longitude1)
	if float(repr(temp))>1.0:
		 temp = 1.0
	d = math.acos(temp)*R
	return d;#单位是km

#计算运动学片段的行驶距离
def XSJL(WJD):#纬度、经度
    WJD.index=range(len(WJD))#从0建索引
    JL=0
    WJD1=[] #纬经度
    WJD2=[] #纬经度
    for (index, se) in WJD.iterrows():
        if index==0:
            WJD1=[se['纬度'],se['经度']]
            continue
        WJD2 = [se['纬度'],se['经度']]
        # print('WJD1：', WJD1,'len:',len(WJD1))
        # print('WJD2：', WJD2, 'len:',len(WJD2))
        JL+=cal_dis(WJD1[0],WJD1[1],WJD2[0],WJD2[1])#计算距离。单位km.
        WJD1=WJD2
    # print('距离：',round(JL*1000,4))
    return round(JL*1000,4)#返回单位米m，4位小数



#求运动学片段的特征们。['起始时刻', '结束时刻', '记录数', '原始开始索引', '原始结束索引','总秒数' ,'怠速秒数','加速秒数', '匀速秒数' ,'减速秒数']
# 返回每个运动学片段的['起始时刻','结束时刻','原始开始索引','原始结束索引','运行时间','加速时间','减速时间','怠速时间','匀速时间','运行距离','最大速度','平均速度','行驶速度','速度标准偏差','最大加速度','加速度段平均加速度','最小减速度','减速段平均减速度','加速度标准偏差','怠速时间比','加速时间比','匀速时间比','减速时间比']
def MotionSeriesFeature():
    global MotionSeriesFeature_List
    # print(LianXuMotionDuan_List)
    for (index, se) in LianXuMotionDuan_List.iterrows():
        i=se['原始开始索引']
        j=se['原始结束索引']
        GPS_V = df1.loc[i:j,'GPS车速']
        SS_A = df1.loc[i:j,'瞬时加速度']
        curr_Feature=list(range(0,23))  #[]23个数字占位。
        curr_Feature[9]='flag'
        curr_Feature[0] = se['起始时刻']
        curr_Feature[1] = se['结束时刻']
        curr_Feature[2] = se['原始开始索引']
        curr_Feature[3] = se['原始结束索引']
        curr_Feature[4] = se['总秒数']
        curr_Feature[5] = se['加速秒数']
        curr_Feature[6] = se['减速秒数']
        curr_Feature[7] = se['怠速秒数']
        curr_Feature[8] = se['匀速秒数']
        # curr_Feature[9] = round(GPS_V.sum()*(5/18),1) #运行距离。单位是m。这是按速度累加得到的运行距离。
        if curr_Feature[9]=='flag':  #9号位为'flag'表示还没计算过距离。
            curr_Feature[9] = XSJL(df1.loc[i:j,['纬度','经度']])#这是按经纬度计算得到的。单位m
        curr_Feature[10] = GPS_V.max()# 最大速度。单位是km/h
        curr_Feature[11] = GPS_V.mean() # 平均速度(包含0)。单位是km/h
        curr_Feature[12] = GPS_V.where(cond=GPS_V!=0).dropna().mean() # 行驶速度(不包含0)。单位是km/h
        curr_Feature[13] = GPS_V.std() # 速度标准偏差。一段时间周期内，汽车速度的标准差，即包括怠速状态。
        curr_Feature[14] = SS_A.max() # 最大加速度
        k1 = SS_A.where(cond=SS_A >= 0.1).dropna().mean()
        curr_Feature[15] = k1 if math.isnan(k1)==False else 0 # 加速度段平均加速度
        curr_Feature[16] = SS_A.min()# 最小减速度
        k2=SS_A.where(cond=SS_A<=-0.1).dropna().mean()
        curr_Feature[17] = k2 if math.isnan(k2)==False else 0 # 减速段平均减速度
        k3=SS_A.where(cond=SS_A >= 0.1).dropna().std()
        curr_Feature[18] = k3 if math.isnan(k3)==False else 0 # 加速度标准偏差。一段时间周期内，处在加速状态的汽车加速度的标准差。也就是只算瞬时加速度大于0.1m/s2的标准差。
        curr_Feature[19] = se['怠速秒数']/se['总秒数']#怠速时间比
        curr_Feature[20] = se['加速秒数']/se['总秒数'] # 加速时间比
        curr_Feature[21] = se['匀速秒数']/se['总秒数'] # 匀速时间比
        curr_Feature[22] = se['减速秒数']/se['总秒数'] # 减速时间比
        MotionSeriesFeature_List.append(curr_Feature)
    MotionSeriesFeature_List = pd.DataFrame(MotionSeriesFeature_List, columns=['起始时刻','结束时刻','原始开始索引','原始结束索引','运行时间','加速时间','减速时间','怠速时间','匀速时间','运行距离','最大速度','平均速度','行驶速度','速度标准偏差','最大加速度','加速度段平均加速度','最小减速度','减速段平均减速度','加速度标准偏差','怠速时间比','加速时间比','匀速时间比','减速时间比'])
    # 排序是为了打印查看。
    # MotionSeriesFeature_List = MotionSeriesFeature_List.sort_values(by='运行时间', ascending=False)
    # print(MotionSeriesFeature_List)

#判断当前时刻是怠速6、加速7、匀速8、减速9
def PDState(se):
    # print(se["Y轴加速度"])
    if (se["GPS车速"]==0 and se["瞬时加速度"]==0) :# 怠速。车速为0且加速度也为0。   不考虑车速小于10，因为在第一问已经处理。
        return 6
    elif (se["瞬时加速度"]>=0.1):# 加速。瞬时加速度大于0.1m/s2。
        return 7
    elif (se["GPS车速"]!=0 and abs(se["瞬时加速度"])<0.1):# 匀速。车速不为0，且瞬时加速度绝对值小于0.1m/s2。
        return 8
    elif (se["瞬时加速度"]<=-0.1):  # 减速。瞬时加速度小于-0.1m/s2
        return 9
    else:#当GPS车速为0，瞬时加速度 大于0 小于0.1时，不会进入上面任何一个。
        # print('不应该出现这条语句');print(se)
        return 6


if __name__ == '__main__':
    '''下面的代码是在data2_(DropGPSV10)_138599.csv上执行的'''

    '''读入文件'''
    df1 = ReadFile()
    # print(df1)

    #划分运动学片段。
    HFMotionSeries()
    LianXuMotionDuan_List.to_csv('./3MidData_csv2/未筛选的运动学片段_'+str(len(LianXuMotionDuan_List))+'_(80%).csv',index=False)

    # 筛选运动学片段。
    ChooseMotionSeries()
    LianXuMotionDuan_List.to_csv('./3MidData_csv2/筛选后的运动学片段_' + str(len(LianXuMotionDuan_List)) + '_(80%).csv', index=False)

    # 运动学片段的特征
    MotionSeriesFeature()
    MotionSeriesFeature_List.to_csv('./3MidData_csv2/MotionSeriesFeature_'+str(len(MotionSeriesFeature_List))+'_(80%).csv',index=False)


