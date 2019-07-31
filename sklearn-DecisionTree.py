#调用sklearn库实现决策树
#这里的lenses.txt文件为隐形眼镜数据集，它包含了很多患者眼部状况的观察条件以及医生推荐的隐形眼镜类型。

from sklearn.preprocessing import LabelEncoder,OneHotEncoder#其中LabelEncoder是用来对分类型特征值进行编码，即对不连续的数值或文本进行编码。OneHotEncoder用于将表示分类的数据扩维，它可以实现将分类特征的每个元素转化为一个可以用来计算的值。
import numpy as np 
import pandas as pd 
from sklearn import tree#导入决策树库
import os

os.chdir("D://Python")#指定目录
with open("lenses.txt") as fr:#打开数据文件lenses.txt
    lenses=[inst.strip().split('\t') for inst in fr.readlines()]#读取数据文件，解析以Tab分隔的数据行
lenses_target=[]#用来存放每组数据的类别
for each in lenses:
    lenses_target.append([each[-1]])
lensesLabels = ['age','prescript','astigmatic','tearRate']#特征标签，依次为年龄，症状，是否散光，眼泪数量，
lenses_list = []#用来临时存放lenses数据
lenses_dict = {}#用来存放lenses数据的字典，用于生成pandas
for each_label in lensesLabels:#一下几句用于提取信息，生产字典
    for each in lenses:
        lenses_list.append(each[lensesLabels.index(each_label)])#index方法用于从列表里面找出某一个值的第一个匹配项的索引位置
    lenses_dict[each_label] = lenses_list
    lenses_list = []
print(lenses_dict)#打印数据字典
lenses_pd = pd.DataFrame(lenses_dict)#生成DataFrame对象，DataFrame是一个以命名列方式组织的分布式数据集。
print(lenses_pd)#打印DataFrame对象
le = LabelEncoder()#创建LabelEncoder()对象，用于序列化，将object类型转化为数值。
for col in lenses_pd.columns:#将每一列进行序列化
    lenses_pd[col] = le.fit_transform(lenses_pd[col])#fit_transform()做两件事情：fit找到数据转换规则，并将数据进行标准化处理(将数字变成矩阵，哑变量处理)
print(lenses_pd)#打印编码后的数据，可与前面的lenses_pd进行对比

clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=4)#创建决策树类；其中criterion='entropy'表示使用信息熵作为划分标准，max_depth=4表示决策树最大深度为4
clf = clf.fit(lenses_pd.values.tolist(),lenses_target)#构建决策树
clf = clf.predict([[1,1,1,0]])#使用决策树进行预测任意数据
print(clf)#输出预测结果
