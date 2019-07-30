#sklearn库是在numpy、scipy、matplotlib的基础上开发而成的，在安装sklearn之前需要先安装这些依赖库
#Numpy（Numerical python的缩写）是一个开源的Python科学计算库
#Scipy库是sklearn库的基础，它是基于Numpy的一个集成了多种数学算法和函数的Python模块
#matplotlib是基于Numpy的一套Python工具包，它提供了大量的数据绘图工具

from sklearn.datasets import load_iris #导入sklearn库中自带的鸢尾花数据集（适用于分类问题），
#这个数据集包含了150个鸢尾花样本，对应3种鸢尾花，各50个样本，以及它们各自对应的4种关于花外形的数据，
#它的目标是为了根据花萼长度、花萼宽度、花瓣长度、花瓣宽度这四个特征来识别出鸢尾花属于山鸢尾、变色鸢尾和维吉尼亚鸢尾中的哪一种。
from sklearn.preprocessing import MinMaxScaler#sklearn的preprocessing库用来对数据预处理，包括无量纲化，特征二值化，定性数据量化等；这里导入MinMaxScaler类是基于最大最小值，将特征值转换到[0, 1]区间上
from sklearn.cluster import KMeans#导入cluster库中的kmeans类，关于kmeans的聚类思想自行网上了解
import matplotlib.pyplot as plt#导入画图工具，matplotlib.pyplot是一些命令行风格函数的集合，使matplotlib以类似于MATLAB的方式工作

iris = load_iris()#导入数据
iris_data = iris['data']#获取特征变量
iris_target = iris['target']#获取标签
iris_names = iris['feature_names']#获取特征变量名
#print(iris_target.shape)
scale = MinMaxScaler().fit(iris_data)#区间缩放，返回值为缩放到[0, 1]区间的数据。构建规则
iris_dataScale = scale.transform(iris_data)#将刚刚构建的规则应用到鸢尾花的数据集上。应用规则
kmeans = KMeans(n_clusters=3,random_state=123).fit(iris_dataScale)#构建并训练模型，其中n-cluster定义分类簇的数量，random_state定义随机数生成器的种子
print(kmeans)#打印出刚刚构建的Kmeans模型
result = kmeans.predict([[1.5,1.5,1.5,1.5]])#随意输入一组数据进行预测
print(result[0])#将预测结果输出

P = plt.figure(figsize=(12,12))#创建画布
label_pred = kmeans.labels_#获取聚类标签
centroids = kmeans.cluster_centers_#获取聚类中心
inertia = kmeans.inertia_#获取聚类准则的总和
mark = ['or','ob','og','ok','^r']#这里o代表画圆圈，r代表颜色为红色，之后依次类推
j = 0
for i in label_pred:
    plt.plot([iris_data[j:j+1,0]],[iris_data[j:j+1,1]],mark[i],markersize = 5)#第一个参数为x轴数据，第二个参数为y轴数据，第三个参数表示根据聚类标签的不同，显示对应的样式（形状+颜色），第四个参数表示大小。
    j +=1
plt.show()#显示画布
