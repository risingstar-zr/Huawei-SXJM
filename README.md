# Huawei-SXJM
2019 --“华为杯”第十六届中国研究生数学建模竞赛--D类赛题 汽车行驶工况构建 项目开源  
本人所在的团队参与D类赛题“汽车行驶工况构建”数据建模研究，最终获得二等奖。非常感谢我的两个队友（室友）， 队友都很强，大家为了这个比赛奋斗了五个日夜，最终结果还算满意。  
D类赛题的要求及相关文件代码：  
详见百度云盘：链接：https://pan.baidu.com/s/1dPreZFXKvUmk0-bMtO0y1A 提取码：wiro  

程序是使用pycharm开发工具编写的python程序。  
下面是项目中代码文件名的使用说明：  

`1. WJ1_dataClean2.py	1Data_csv/data1.csv	data1.csv是官方提供的文件1.xlsx。该脚本是对“文件1”数据集进行的第一步数据预处理，剔除了加速度不在合理范围的异常数据。  `  
 `2. WJ1_dataClean3.py	3MidData_csv/data1_(Max_A_D)_NOnan_184971.csv该脚本是对“文件1”数据集进行的第二步数据预处理，根据停车规律剔除了长期停车（GPS车速为0的连续时间大于180）的异常数据。  `  
`3. WJ1_dataClean4.py	3MidData_csv/data1_(DropGPSV0)_178588.csv	该脚本是对“文件1”数据集进行的第三步数据预处理，GPS车速大于等于0且小于10的连续时间段大于180的时间段的异常数据。 `   
`4. WJ1_HFYDXPD1.py	3MidData_csv/data1_(DropGPSV10)_177603.csv	该脚本是对“文件1”数据集进行运动学片段的划分、筛选有效的运动学片段、计算运动学片段特征。`  
`5. WJ2_dataClean2.py	1Data_csv/data2.csv	data2.csv就是官方提供的文件2.xlsx。  该脚本是对“文件2”数据集进行的第一步数据预处理，剔除了加速度不在合理范围的异常数据。  `  
`6. WJ2_dataClean3.py	3MidData_csv2/data2_(Max_A_D)_NOnan_143054.csv	该脚本是对“文件2”数据集进行的第二步数据预处理，根据停车规律剔除了长期停车（GPS车速为0的连续时间大于180）的异常数据。`  
`7. WJ2_dataClean4.py	3MidData_csv2/data2_(DropGPSV0)_143054.csv	该脚本是对“文件2”数据集进行的第三步数据预处理，GPS车速大于等于0且小于10的连续时间段大于180的时间段的异常数据。  `  
`8. WJ2_HFYDXPD1.py		3MidData_csv2/data2_(DropGPSV10)_138599.csv	该脚本是对“文件2”数据集进行运动学片段的划分、筛选有效的运动学片段、计算运动学片段特征。  `  
`9. WJ3_dataClean2.py	1Data_csv/data3.csv	data3.csv就是官方提供的文件3.xlsx。  该脚本是对“文件3”数据集进行的第一步数据预处理，剔除了加速度不在合理范围的异常数据。  `  
`10.  WJ3_dataClean3.py	该脚本是对“文件3”数据集进行的第二步数据预处理，根据停车规律剔除了长期停车（GPS车速为0的连续时间大于180）的异常数据。
3MidData_csv3/data3_(Max_A_D)_NOnan_163668.csv`  
`11. WJ3_dataClean4.py	3MidData_csv3/data3_(DropGPSV0)_163668.csv	该脚本是对“文件3”数据集进行的第三步数据预处理，GPS车速大于等于0且小于10的连续时间段大于180的时间段的异常数据。  `  
`12. WJ3_HFYDXPD1.py	./3MidData_csv3/data3_(DropGPSV10)_157846.csv	该脚本是对“文件3”数据集进行运动学片段的划分、筛选有效的运动学片段、计算运动学片段特征。`  
`13. WJ1_JW_JL.py  对“文件1”、“文件2”、“文件3”划分出的有效运动学片段进行标准化、降维、聚类、计算每个簇的特征。  
3MidData_csv/MotionSeriesFeature_1103_(80%).csv  
3MidData_csv2/MotionSeriesFeature_822_(80%).csv  
3MidData_csv3/MotionSeriesFeature_786_(80%).csv	  `  
`14. WJ1_xuanpianduan.py  从“文件1”、“文件2”、“文件3”划分出的有效运动学片段中选出1200-1300s时长的运动学片段、用特征评价体系进行评估。  
HeBing3GeWenJianDeYunDongXuePianDuanTeZheng(BaoHanCuHao).csv`  
`15. data123_(DropGPSV10)_474048.csv	画速度曲线图。`  

建模思路：  
要求：1、预处理（加减速异常、长时间停车、怠速等情况）；2、运动学片段的提取；3、汽车行驶工况的构建；  
解决思路：为了解决上述问题，我们对模型提出假设。根据运动学片段的定义，找出所有的运动学片段，然后个根据运动学片段的运动学特征参数，进行数学建模，对选取出的有代表性的运动学片段，由于每个片段的特征参数量纲不同，需将原始数据标准化（（原始数据-样本均值）/样本的标准差），求矩阵的相关系数，计算矩阵的特征值和特征向量，最后计算主成分方差贡献率以及累积方差贡献率，找主要的特征参数指标。
整个数学建模过程，都是基于Python3.6语言实现。  

觉得有用的话，麻烦给个star~
  

