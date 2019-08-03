#不调用sklearn库，实现决策树算法（使用是否放贷的案例）

'''
本案例为：根据个人的年龄、是否有工作、是否有自己的房子、信贷情况这四个信息决定给不给这个人贷款
年龄：0代表青年，1代表中年，2代表老年；
有工作：0代表否，1代表是；
有自己的房子：0代表否，1代表是；
信贷情况：0代表一般，1代表好，2代表非常好；
类别(是否给贷款)：no代表否，yes代表是。
'''
from matplotlib.font_manager import FontProperties#解决用matplotlib绘图时，常出现不显中文或乱码的问题
import matplotlib.pyplot as plt#画图工具
from math import log#导入对数函数
import pickle#pickle包可以将决策树保存下来，方便进行调用
import operator#Operator模块提供了一系列与Python自带操作一样有效的函数

'''
函数说明：创建测试数据集
参数：无
返回：
dataSet - 数据集
labels - 分类属性
'''
def createDataSet():
    dataSet = [[0,0,0,0,'no'],[0,0,0,1,'no'],[0,1,0,1,'yes'],[0,1,1,0,'yes'],[0,0,0,0,'no'],[1,0,0,0,'no'],[1,0,0,1,'no'],[1,1,1,1,'yes'],[1,0,1,2,'yes'],[1,0,1,2,'yes'],[2,0,1,2,'yes'],[2,0,1,1,'yes'],[2,1,0,1,'yes'],[2,1,0,2,'yes'],[2,0,0,0,'no']]
    labels = ['年龄','有工作','有自己的房子','信贷情况']
    return dataSet,labels

'''
函数说明：计算给定数据集经验熵(香农熵)
'''
def calcShannonEnt(dataSet):
    #返回数据集行数
    numEntires = len(dataSet)
    #保存标签出现次数
    labelCounts={}
    #对每组特征向量进行统计
    for featVec in dataSet:
        currentLabel = featVec[-1]
        #如果标签没有放入统计次数的字典里面，需要添加进去
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        #对标签进行计数
        labelCounts[currentLabel]+=1
    #定义经验熵
    shannonEnt = 0.0
    #对经验熵进行计算
    for key in labelCounts:
        #选择该标签的概率
        prob = float(labelCounts[key])/numEntires
        shannonEnt -= prob*log(prob,2)
    return shannonEnt

'''
函数说明：按照给定的特征划分相应的数据集
参数：
dataSet - 待划分的数据集
axis - 划分数据集特征
value - 需要返回的特征的值
返回：被划分的数据集
'''
def splitDataSet(dataSet,axis,value):
    #创建返回的数据集列表
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis] == value:
            #去掉axis特征
            reducedFeatVec = featVec[:axis]
            #将符合条件的田间道返回数据集
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

'''
函数说明：选择最优特征
Gain(D,g)=Ent(D)-SUM(|Dv|/|D|)*Ent(Dv)
参数：
dataSet - 数据集
返回：
bestFeature - 信息增益最大的特征的索引值
'''
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) -1 #特征数量
    #计算香农熵
    baseEntropy = calcShannonEnt(dataSet)
    #信息增益
    bestInfoGain = 0.0
    #最优特征的索引值
    bestFeature = -1
    for i in range(numFeatures):
        #获取dataSet的第i个所有特征存在featList列表中
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        #定义经验条件熵
        newEntropy = 0.0
        #求经验条件熵
        for value in uniqueVals:
            #subDataSet划分后的子集
            subDataSet=splitDataSet(dataSet,i,value)
            #计算子集的概率
            prob = len(subDataSet)/float(len(dataSet))
            #根据公式计算经验条件熵
            newEntropy += prob * calcShannonEnt(subDataSet)
        #求信息增益
        infoGain = baseEntropy - newEntropy
        print("第%d个特征的增益为%.3f" %(i,infoGain))
        if(infoGain>bestInfoGain):
            #更新信息增益,找到最大的信息增益
            bestInfoGain = infoGain
            #记录信息增益最大的特征的索引值
            bestFeature = i
    return bestFeature

'''
函数说明：统计classList中出现最多的元素（类标签）
参数：
classList - 类标签列表
返回：
sortedClassCount[0][0] - 出现次数最多的元素(类标签)
'''
def majorityCnt(classList):
    classCount={}
    #统计classList中的每个元素出现的次数
    for vote in classList:
        if vote not in classList.keys():
            classCount[vote]=0
        classCount[vote] += 1
    #根据字典的值进行降序排序
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

'''
函数说明：创建决策树（ID3算法）
递归有终止条件：
1、所有的类标签完全相同，直接返回类标签
2、用完所有标签但得不到唯一类别的分组，就是说特征不够用，挑选出现数量最多的类别来进行返回
参数：
dataSet - 训练集
labels - 分类属性标签
featLabels - 存储选择的最优特征标签
返回：
myTree - 决策树
'''
def createTree(dataSet,labels,featLabels):
    #获取分类标签（是否放贷：yes or no）
    classList = [example[-1] for example in dataSet]
    #如果类别完全相同停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #遍历完所有特征时返回出现次数最多的类标签
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    #选择最优特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    #最优特征取标签
    bestFeatLabel = labels[bestFeat]
    featLabels.append(bestFeatLabel)
    #根据最优特征的标签生成决策树
    myTree = {bestFeatLabel:{}}
    #删除已经使用过的特征标签
    del(labels[bestFeat])
    #获取训练集中所有最优解特征属性值
    featValues = [example[bestFeat] for example in dataSet]
    #去除重复的属性值
    uniqueVals = set(featValues)
    #对特征进行遍历，生成决策树
    for value in uniqueVals:
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),labels,featLabels)
    return myTree

'''
函数说明：获取决策树的叶子节点数目
参数：
myTree - 决策树
返回：
numLeafs - 决策树1的叶子节点的数目
'''
def getNumLeafs(myTree):
    #初始化叶子
    numLeafs=0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        #测试该节点是不是为字典，如果不是字典，代表此节点为叶子节点
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

'''
函数说明：获取决策树层数
参数：
myTree - 决策树
返回：
maxDepth - 决策树层数
'''
def getTreeDepth(myTree):
    #初始化决策树深度
    maxDepth = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        #测试该节点是不是为字典，如果不是字典，代表此节点为叶子节点
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth += getNumLeafs(secondDict[key])
        else:
            thisDepth += 1
        if thisDepth>maxDepth:
            maxDepth = thisDepth;
    return maxDepth


'''
函数说明：使用决策树分类
参数：
inputTree - 已经生成的决策树
featLabels - 存储选择的最优特征标签
testVec - 测试数据列表
返回：
classLabel - 分类结果
'''
def classify(inputTree,featLabels,testVec):
    #获取决策树节点
    firstStr = next(iter(inputTree))
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def main():
    dataSet,features = createDataSet()
    featLabels = []
    myTree = createTree(dataSet,features,featLabels)
    #定义学习集
    testVec=[1,1,0,1]
    result = classify(myTree,featLabels,testVec)
    if result == 'yes':
        print('放贷')
    else:
        print('不放贷')
    #print(myTree)
    #print("最优特征索引值："+str(chooseBestFeatureToSplit(dataSet)))

if __name__ == '__main__':
    main()