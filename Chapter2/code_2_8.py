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

def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

class laoHuJi:
    '''
    建立k臂老虎机的类，输出服从(0,1)分布的价值函数
    '''
    def __init__(self,k):
        self.num = k

    def reset(self):
        self.q = np.random.normal(4,1,self.num)
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
    def __init__(self,kArm,learningRate,rArrage):
        self.kArm = kArm
        self.H = np.zeros(kArm)
        self.rArrage = rArrage
        self.learningRate = learningRate

    def getAction(self):
        probs = softmax(self.H)
        a = np.random.choice(np.arange(probs.shape[0]), p=probs.ravel()),probs.ravel()
        return a[0]

    def updateH(self,a,r):
        for i in range(self.kArm):
            if i == a:
                self.H[i] = self.H[i] + self.learningRate * (r  - self.rArrage) * (1 - softmax(self.H)[i])
            else:
                self.H[i] = self.H[i] - self.learningRate * (r  - self.rArrage) * softmax(self.H)[i]
    
    def reset(self):
        self.H = np.zeros(kArm)

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
    maxStep = 1000  # 每个老虎机最大训练次数
    maxEpoch = 1000 # 最多训练2000个老虎机
    learningRate= 0.1
    rArrage = 4

    kLaoHuJi = laoHuJi(kArm)
    model = qModel(kArm,learningRate,rArrage)
    record = recordR(maxStep)
    for i in tqdm(range(maxEpoch)):
        aFlag = kLaoHuJi.reset()
        model.reset()
        for j in range(maxStep):
            a = model.getAction()
            r = kLaoHuJi.get_reward(a)
            model.updateH(a,r)
            record.updateRecord(j,a==aFlag,r)

    num,trackingReward = record.getRecord()
    num = num/maxEpoch
    trackingReward = trackingReward/maxEpoch
    plt.figure()
    plt.plot(num)
    plt.draw()
    plt.pause(1000)
    plt.close()