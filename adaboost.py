#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i,-1] == 0:
            data[i,-1] = -1
    # print(data)
    return data[:,:2], data[:,-1]

from numpy import *

#通过比较阈值进行分类
#threshVal是阈值 threshIneq决定了不等号是大于还是小于
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):

    retArray = ones((shape(dataMatrix)[0],1)) #先全部设为1
    if threshIneq == 'lt':  #然后根据阈值和不等号将满足要求的都设为-1
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

#在加权数据集里面寻找最低错误率的单层决策树
#D是指数据集权重 用于计算加权错误率
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)  #m为行数 n为列数
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    minError = inf #最小误差率初值设为无穷大
    for i in range(n):  #第一层循环 对数据集中的每一个特征 n为特征总数
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1): #第二层循环 对每个步长
            for inequal in ['lt','gt']: #第三层循环 对每个不等号
                threshVal = rangeMin + float(j) * stepSize#计算阈值
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#根据阈值和不等号进行预测
                errArr = mat(ones((m,1)))#先假设所有的结果都是错的（标记为1）
                errArr[predictedVals == labelMat] = 0#然后把预测结果正确的标记为0
                weightedError = D.T*errArr#计算加权错误率 
                #print 'split: dim %d, thresh %.2f, thresh inequal: %s, \
                #        the weightederror is %.3f' % (i,threshVal,inequal,weightedError)
                if weightedError < minError:    #将加权错误率最小的结果保存下来
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


#基于单层决策树的AdaBoost训练函数
#numIt指迭代次数 默认为40 当训练错误率达到0就会提前结束训练
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []   #用于存储每次训练得到的弱分类器以及其输出结果的权重
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)  #数据集权重初始化为1/m
    aggClassEst = mat(zeros((m,1))) #记录每个数据点的类别估计累计值
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#在加权数据集里面寻找最低错误率的单层决策树
        #print "D: ",D.T
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#根据错误率计算出本次单层决策树输出结果的权重 max(error,1e-16)则是为了确保error为0时不会出现除0溢出
        bestStump['alpha'] = alpha#记录权重
        weakClassArr.append(bestStump)
        #print 'classEst: ',classEst.T
        #计算下一次迭代中的权重向量D
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)#计算指数
        D = multiply(D,exp(expon))
        D = D/D.sum()#归一化
        #错误率累加计算
        aggClassEst += alpha*classEst
        #print 'aggClassEst: ',aggClassEst.T
        #aggErrors = multiply(sign(aggClassEst)!=mat(classLabels).T,ones((m,1)))
        #errorRate = aggErrors.sum()/m
        errorRate = 1.0*sum(sign(aggClassEst)!=mat(classLabels).T)/m#sign(aggClassEst)表示根据aggClassEst的正负号分别标记为1 -1
        print ('total error: ',errorRate)
        if errorRate == 0.0:#如果错误率为0那就提前结束for循环
            break
    return weakClassArr

#基于AdaBoost的分类函数
#dataToClass是待分类样例 classifierArr是adaBoostTrainDS函数训练出来的弱分类器数组
def adaClassify(dataToClass,classifierArr):
    dataMatrix = mat(dataToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)): #遍历所有的弱分类器
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        #print aggClassEst
    return sign(aggClassEst)

X,Y = create_data()
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state=42)

classifierArray = adaBoostTrainDS(X_train,Y_train,10)
prediction = adaClassify(X_test,classifierArray)
print ('错误率：',1.0*sum(prediction!=mat(Y_test).T)/len(prediction))

prediction = prediction.getA()
prediction = prediction.reshape(-1)

figure1 = plt.figure()
plt.scatter(X_test[:,0],X_test[:,1],c=Y_test)#绘制测试集散点图
plt.scatter(X_test[:,0], X_test[:,1],c=prediction,cmap='Oranges',marker='+')#绘制测试集预测散点图
plt.title('测试集预测结果与标签对比图')
plt.show()