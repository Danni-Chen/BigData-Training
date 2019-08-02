#sklearn库的练习
from sklearn.datasets import load_breast_cancer#导入sklearn库datasets模块下的乳腺癌数据集（用于分类，良性和恶性） 
from sklearn.model_selection import train_test_split#train_test_split函数用于将矩阵随机划分为训练子集和测试子集，并返回划分好的训练集测试集样本和训练集测试集标签
from sklearn.preprocessing import MinMaxScaler#MinMaxScaler用于对原始数据进行线性变换，变换到[0,1]区间（也可以是其他固定最小最大值的区间）
from sklearn.preprocessing import StandardScaler#StandardScaler用于计算训练集的平均值和标准差，以便测试数据及使用相同的变换
from sklearn.decomposition import PCA#主成分分析PCA通过线性变换将原始数据变换为一组各维度线性无关的表示，可用于提取数据的主要特征分量，常用于高维数据的降维
from sklearn.svm import SVC#全称是C-Support Vector Classification，是一种基于libsvm的支持向量机，能够在指定的数据集上进行多类分类任务
#sklearn库的metric模块提供了一些函数，用来计算真实值与预测值之间的预测误差。其中以_score结尾的函数，返回一个最大值，越高越好；以_error结尾的函数，返回一个最小值，越小越好
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score
#accuracy_score函数计算正确率, 也就是计算正确预测的比例或数量；
#precision_score函数计算精度，也就是预测正确的正例数据占预测为正例数据的比例；
#recall_score函数计算召回率=提取出的正确信息条数/样本中的信息条数，通俗地说，就是所有准确的条目有多少被检索出来了；
#f1_score函数计算F1值，是精度和召回率的调和平均值；
#cohen_kappa_score函数计算Cohen’s kappa统计，这个度量指标旨在比较由不同的人类标注者给出的标签，而不是去比较分类器预测和真值。

from sklearn.metrics import classification_report#classification_report函数用于构建文本报告，用于展示主要的分类指标metrics
from sklearn.metrics import roc_curve#roc_curve函数计算接收机操作特性曲线(receiver operating characteristic curve, or ROC curve）
import matplotlib.pyplot as plt#导入画图工具
import numpy as np#导入科学计算库

cancer = load_breast_cancer()#从sklearn.datasets下载良/恶性肿瘤预测数据
#print(len(cancer))#打印数据集的长度
#print(type(cancer))#打印数据集的类型

cancer_data = cancer['data']#获取数据集的特征变量
#print(cancer_data)

cancer_target = cancer['target']#获取数据集的标签
#print(cancer_target)

cancer_names = cancer['feature_names']#获取数据集的特征变量名
#print(cancer_names)

cancer_desc = cancer['DESCR']#获取数据集的描述信息
#print(cancer_desc)

#print(cancer_data.shape)
#print(cancer_target.shape)

#train_test_split函数的格式：X_train,X_test, y_train, y_test = cross_validation.train_test_split(train_data,train_target,test_size=0.3, random_state=0)
#其中的train_data表示被划分的样本特征集。train_target表示被划分的样本标签。test_size表示测试集的大小，如果是浮点数在0-1之间，表示样本占比；如果是整数的话就是样本的数量。random_state代表是随机数的种子。
cancer_data_train,cancer_data_test,cancer_target_train,cancer_target_test = train_test_split(cancer_data,cancer_target,test_size=0.2,random_state=22)
#print(cancer_data_train.shape)
#print(cancer_data_test.shape)
#print(cancer_target_train.shape)
#print(cancer_target_test.shape)

Scaler = MinMaxScaler().fit(cancer_data_train)#构建规则。离差标准化（区间缩放，返回值为缩放到[0, 1]区间的数据）。如果需要使用标准差标准化是，只需要将MinMaxScaler换成StandardScaler即可
cancer_trainScaler = Scaler.transform(cancer_data_train)#应用规则。将刚刚构建的规则应用到乳腺癌的训练数据集上。
cancer_testScaler = Scaler.transform(cancer_data_test)#应用规则。将刚刚构建的规则应用到乳腺癌的测试数据集上。
#print(np.min(cancer_data_train))#打印出离差标准化前训练集数据的最小值
#print(np.min(cancer_testScaler))#打印出离差标准化后测试集数据的最小值

pca_model = PCA(n_components=10).fit(cancer_trainScaler)#构建PCA降维模型。n_components参数表示PCA算法中所要保留的主成分个数n，也即保留下来的特征个数n。
cancer_trainPca = pca_model.transform(cancer_trainScaler)#将降维模型应用于标准化之后的训练集数据
cancer_testPca = pca_model.transform(cancer_testScaler)#将降维模型应用于标准化之后的测试集数据
#print(cancer_trainScaler.shape)#打印出降维前的训练数据的形状
#print(cancer_trainPca.shape)#打印出降维后的训练数据的形状


stdScaler = StandardScaler().fit(cancer_data_train)#构建规则。标准差标准化
cancer_trainStd = stdScaler.transform(cancer_data_train)#应用规则
cancer_testStd = stdScaler.transform(cancer_data_test)#应用规则
svm = SVC().fit(cancer_trainStd,cancer_target_train)#建立SVC模型
#print(svm)#打印svc模型
cancer_target_pred = svm.predict(cancer_testStd)#预测测试集结果
true = np.sum(cancer_target_pred == cancer_target_test)#计算预测结果和真实情况一样的数量
print(true)#打印预测正确的数量
print(cancer_target_test.shape[0]-true)#打印预测错误的数量
print(true/cancer_target_test.shape[0])#打印预测结果的准确率

print(accuracy_score(cancer_target_test,cancer_target_pred))#打印准确率
print(precision_score(cancer_target_test,cancer_target_pred))#打印精确率
print(recall_score(cancer_target_test,cancer_target_pred))#打印召回率
print(f1_score(cancer_target_test,cancer_target_pred))#打印F1值
print(cohen_kappa_score(cancer_target_test,cancer_target_pred))#打印Cohen's Kappa系数

print(classification_report(cancer_target_test,cancer_target_pred))#打印乳腺癌数据的分类报告

fpr,tpr,thresholds=roc_curve(cancer_target_test,cancer_target_pred)#求ROC曲线的x轴和y轴，fps和tps就是混淆矩阵中的FP和TP的值，thresholds就是cancer_target_pred逆序排列后的结果
plt.figure(figsize=(10,6))#创建画布
plt.xlim(0,1)#x轴范围
plt.ylim(0.0,1.1)#y轴范围
plt.xlabel("x")#x轴标签
plt.ylabel("y")#y轴标签
plt.plot(fpr,tpr,linewidth=2,linestyle='-',color='red')#绘图数据和绘图样式
plt.show()#显示画布