'''
说明：本代码应用于强化学习（第2版）的2.5节
    10 臂老虎机
    对2.3代码进行了修改，删除了qModel中计算qSum的矩阵，并改为了增量更新公式。
    对2.4代码进行了修改，可以自由改步长
    加入了UCB更新
作者：张天昊
版本：0.0
时间：12/06/19
'''

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class laoHuJi:
    '''
    建立k臂老虎机的类，输出服从(0,1)分布的价值函数
    '''
    def __init__(self,k):
        self.num = k

    def reset(self):
        self.q = np.random.normal(0,1,self.num)
        return np.argmax(self.q)
        
    def get_reward(self,a):
        r = np.random.normal(self.q[a],1)
        return r

    def updateValue(self):
        self.q += np.random.normal(0,0.01,self.num)
        return np.argmax(self.q)


class qModel:
    '''
    生成动作选择模型
    基于e-贪婪生成动作
    Q估计为R的均值
    R服从期望价值方差1的高斯分布
    '''
    def __init__(self,kArm,epsilon,c):
        self.kArm = kArm
        self.epsilon = epsilon
        self.qGuJi = np.zeros(kArm)
        self.aNum = np.zeros(kArm)
        self.c = c

    def getAction(self,t):
        if np.any(self.aNum == 0):
            index = np.where(self.aNum==0)
            return index[0][0]
        else:
            return np.argmax(self.qGuJi + self.c * np.sqrt(np.true_divide(np.log(t),self.aNum)))

    def updateQ(self,a,r,learningRate):
        self.aNum[a] += 1
        if learningRate == 2:
            learningRate = 1/self.aNum[a]
        self.qGuJi[a] = self.qGuJi[a] + learningRate * (r - self.qGuJi[a])
    
    def reset(self):
        self.qGuJi = np.zeros(kArm)
        self.aNum = np.zeros(kArm)


class recordR:
    '''
    用于记录训练过程中的数据
    self.num 训练第t步时的最优动作的次数
    '''
    def __init__(self,maxStep):
        self.num = np.zeros(maxStep)    #训练步数时的平均
        self.trackingReward = np.zeros(maxStep) #训练步数的平均收益
    
    def updateRecord(self,t,aFlag,reward):
        self.num[t] += aFlag
        self.trackingReward[t] += r
    
    def getRecord(self):
        return self.num, self.trackingReward

if __name__ == '__main__':

    kArm = 10  # 老虎机臂数
    epsilon = 0.1 # 贪心随机数
    maxStep = 1000  # 每个老虎机最大训练次数
    maxEpoch = 1000 # 最多训练2000个老虎机
    learningRate= 2
    c = 2

    kLaoHuJi = laoHuJi(kArm)
    model = qModel(kArm,epsilon,c)
    record = recordR(maxStep)
    for i in tqdm(range(maxEpoch)):
        aFlag = kLaoHuJi.reset()
        model.reset()
        for j in range(maxStep):
            a = model.getAction(j)
            r = kLaoHuJi.get_reward(a)
            model.updateQ(a,r,learningRate)
            record.updateRecord(j,a==aFlag,r)

    num,trackingReward = record.getRecord()
    num = num/maxEpoch
    trackingReward = trackingReward/maxEpoch
    plt.figure()
    plt.plot(trackingReward)
    plt.draw()
    plt.pause(100)
    plt.close()