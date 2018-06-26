import numpy as np
import random
import itertools
import scipy.misc
import matplotlib.pyplot as plt
import tensorflow as tf
import os
%matplotlib inline

class gameOb():
    def __init__(self, coordinates, size, intensity, channel, reward, name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name
class gameEnv():
    def __init__(self, size):
        self.sizeX = size
        self.sizeY = size
        self.actions = 4
        self.objects = []
        a = self.reset()
        plt.imshow(a, interpolation="nearest") 
    def reset(self):
        self.objects = []
        hero = gameOb(self.newPosition(), 1,1,2, None,'hero')
        self.objects.append(hero)
        goal = gameOb(self.newPosition(), 1,1,1,1, 'goal')
        self.objects.append(goal)
        hole = gameOb(self.newPosition(), 1,1,0,-1,'fire')
        self.objects.append(hole)
        goal2 = gameOb(self.newPosition(), 1,1,1,1, 'goal')
        self.objects.append(goal2)
        hole2 = gameOb(self.newPosition(), 1,1,0,-1, 'fire')
        self.objects.append(hole2)
        goal3 = gameOb(self.newPosition(), 1,1,1,1, 'goal')
        self.objects.append(goal3)
        goal4 = gameOb(self.newPosition(), 1,1,1,1, 'goal')
        self.objects.append(goal4)
        state = self.renderEnv()
        self.state = state
        return state
    def moveChar(self, direction):
        hero = self.object[0]
        heroX = hero.x
        heroY = hero.y
        if direciton == 0 and hero.y >= 1:
            hero.y -= 1
        if direction == 1 and hero.y <= self.sizeY-2:
            hero.y += 1 
        if direction == 2 and hero.x >= 1:
            hero.x -= 1
        if direction == 3 and hero.x <= self.sizeX-2:
            hero.x += 1
        self.object[0] = hero
    def newPosition(self):
        iterables = [ range(self.sizeX), range(self.sizeY) ]
        points = []
        for t in itertools.product(*iterables):
            points.append(t)
        currentPostion = []
        for objectA in self.objects:
            if (objectA.x, objectA.y) not in currentPositions:
                currentPositions.append(objectA.x, objectA.y)
        for pos in currentPositions:
            points.remove(pos)
        location = np.random.choice(range(len(points)), replace=False)
        return points[location]
    def checkGoal(self):
        others = []
        for obj in self.objects:
            if obj.name == 'hero':
                hero = object 
            else:
                others.append(obj)
        for other in others:
            if hero.x == other.x and hero.y == other.y
            self.objects.remove(other)
            if other.reward == 1:
                self.objects.append(gameOb(self.newPosition(), 1,1,1,1, 'goal'))
            else:
                self.objects.append(gameOb(self.newPosition(), 1,1,0,-1,'fire'))
            return other.reward, False
        return 0,0, False
    def renderEnv(self):
        a = np.ones([self.sizeY+2, self.sizeX+2, 3])
        a[1:-1, 1:-1, :] = 0
        hero = None
        for iterm in self.objects:
            a[iterm.y+1:item.y+item.size+1, item.x+1:item.x+item.size+1,item.channel] = item.intensity

        b = scipy.misc.imresize(a[:,:,0], [84,84,1], interp='nearest')
        c = scipy.misc.imresize(a[:,:,1], [84,84,1], interp='nearest')
        d = scipy.misc.imresize(a[:,:,2], [84,84,1], interp='nearest')
        a = np.stack([b,c,d], axis=2)
        return a
    def step(self, action):
        self.mvoeChar(action) #执行Action, 输入参数为Action，移动hero位置
        reward,done = self.checkGoal() # self.checkGoal()检测hero是否有触碰物体，并得到reward和done标记
        state = self.renderEnv() # self.renderEnv获取环境的图像state
        return state, reward,done # 返回 state, reward, done

    # 随机创建GridWorld环境
    env = gameEnv(size=5) # 5×5 size, gameEnv类初始化方法 


# DQN网络构建
class Qnetwork():
    def __init__(self,h_size):
        self.scalarInput = tf.placeholder(shape=[None, 21168], dtype=tf.float32) # 输入scalarInput 是被扁平化为84*84*3=21168的向量
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 84,84,3]) # rebuild it as [-1, 84,84, 3]尺寸的图片ImageIn
        self.conv1 = tf.contrib.layers.convolution2d(inputs = self.imageIn, num_outputs=32, kernel_size=[8,8], stride=[4,4], padding='VALID', baises_initializer=None) # 创建第一个卷积层，卷积核尺寸为8*8,步长4*4,输出通道数(filter_number)为32,padding=VALID,baises_initializer为空，4×4的步长和VALID模式的padding，所以第一层输出维度为20*20*32
        self.conv2 = tf.contrib.layers.convolution2d(inputs=self.conv1,num_outputs=64,kernel_size=[4,4],stride=[2,2], padding='VALID', biases_initializer=None) # 第二个卷积层尺寸为4*4，步长2*2，通道为64，这一层的输出维度为9*9*64
        self.conv3 = tf.contrib.layers.convolution2d(inputs=self.conv2,num_outputs=64,kernel_size=[3,3],stride=[1,1], padding='VALID', biases_initializer=None) # 第三层卷积层尺寸为3*3，步长为1*1，输出通道数为64，这一层输出维度为7*7*64
        self.conv4 = tf.contrib.layers.convolution2d(inputs=self.conv3,num_outputs=512,kernel_size=[7,7],stride=[1,1], padding='VALID', biases_initializer=None) # 第四层卷积尺寸为7*7，步长为1*1，输出通道数为 512， 这一层的空间尺寸只允许在一个位置进行卷积，output=1*1*512

        # tf.split()将第4个卷积层输出conv4平均拆分成两段
        # streamAC和streamVC， 即Dueling DQN中Advantage Function和Value Function 环境本身的价值
        # tf.split函数的第二个参数代表拆分为几段，第三个参数代表要拆分的第几个维度
        # tf.contrib.layers.flatten 将streamAC和streamVC转化为扁平的streamA和streamV，创建streamA 和 streamVC的线性全连接层参数AW和VW
        # tf.random_normal初始化它们的权重,再使用tf.matmul做全连接层的矩阵乘法，得到self.Advantage和self.Value
        # Advantage是针对Action,因此输出数量为action的数量，而Value则是针对环境统一的，输出数量为1
        # 我们的Q值由Value和Advantage复合而成，即Value加上减去均值的Advantage
        # tf.subtract减去Advantage
        # 均值计算使用的是tf.reduce_mean函数(reduce_indices为1,即代表Action数量的维度)
        # 输出的Action即为Q值最大的Action
        # 使用tf.argmax求出这个Action
        self.streamAC, self.streamVC = tf.split(self.conv4,2,3)
        self.streamA = tf.contrib.layers.flatten(self.streamAC)
        self.streamV = tf.contrib.layers.flatten(self.streamVC)
        self.AW = tf.Variable(tf.random_normal([h_size//2,env.actions]))
        self.VW = tf.Variable(tf.random_normal([h_size//2,1]))
        self.Advantage = tf.matmul(self.streamA,self.AW)
        self.Vaule = tf.matmul(self.streamV, self.VW)

        self.Qout = self.Vaule + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage,reduction_indices=1, keep_dims=True))
        self.predict = tf.argmax(self.Qout,1) 
        # Double DQN, 目标Q值targetQ的输入placeholder， 以及Agent的动作actions的输入placeholder
        # 在计算Q值时，action由主DQN选择，Q值则由辅助的targetDQN生成
        # 在计算预测Q值时,我们以scalar形式的actions转为onehot编码的形式，然后将主DQN生成的Qout乘以actions_onehot,预测得到Q值(Qout和actions都是来自主DQN) 
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.action,env.actions, dtype=tf.float32)
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.action_onehot),reduction_indices=1)
        # 定义loss,使用tf.square和tf.reduce_mean 计算targetQ和Q的均方误差，使用学习速率为1e-4的Adam优化器优化预测Q值和Q值的偏差
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate = 0.0001)
        self.updateModel = self.trainer.minimize(self.loss)
        # 接下来实现Experience Replay策略
        # 初始化需要定义buffer_size即存储样本的最大容量，创建buffer列表。
        # 然后定义向buffer中添加元素的方法，如果超过buffer的最大容量，就清空前面最早的一些样本
        # 并在列表末尾添加新元素
        # 然后定义对样本进行抽样的方法，这里直接使用random.sample()函数随机抽取一定数量的样本

class experience_buffer():
    def __init_-(self, buffer_size = 50000): # 初始化定义buffer_size即存储样本的最大容量
        self.buffer = [] # 创建buffer的列表
        self.buffer_size = buffer_size

    def add(self,experience): #向buffer中添加元素的方法
        if len(self.buffer) + len(experience) >= self.buffer_size: # 如果超过buffer的最大容量
            self.buffer[0:(len(experience) + len(self.buffer)) - \
                          self.buffer_size] = [] # 清空前面最早的一些样本
        self.buffer.extend(experience) # 在列表末尾添加新元素

    def sample(self,size): # 定义对样本进行抽样的方法
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5]) # 使用了random.sample()函数随机抽取一定数量的样本

# 下面定义84*84*3的states扁平化为1维向量的函数processState，方便后面堆叠样本时会比较方便
def processState(states):
    return np.reshape(states,[21168])
# updateTargetGraph函数是更新targetDQN模型参数的方法
# tfVars是TF Graph中的全部参数
# tau是targetDQN向主DQN学习的速率
# updateTargetGraph会取tfVars中前一半参数，即主DQN的模型参数
# 令辅助targetDQN的参数朝着主DQN参数前进一个很小的比例，缓慢学习主DQN，需要稳定的目标Q值训练主网络，所以我们使用一个缓慢学习的targetDQN网络输出目标Q值，并让主网络来优化目标Q值和预测Q值间的loss
# 再让target DQN跟随主DQN并缓慢学习
# updateTargetGraph会创建更新targetDQN模型参数的操作，而函数updateTarget则直接执行这些操作

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value() * \
                                                          tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
        return op_holder
def updateTargetGraph(op_holder, sess):
    for op in op_holder:
        sess.run(op) 

batch_size = 32 # 即每次从exper buffer 中获取多少样本
update_freq = 4 # 每隔多少step执行一次模型参数更新
y = .99 # Q值的衰减系数y设为0.99
startE = 1 # 起始执行随机Action的概率
endE = 0.1 # 最终执行随机Action的概率
anneling_steps = 10000. # 初始随机概率降到最终随机概率所需的步数
num_episodes = 10000  # 进行多少次GridWorld环境的试验
pre_train_steps = 10000  # 正式使用DQN选择Action 前进行多少步随机Action的测试
max_epLength = 50  # Every episode 进行了多少步Action
load_model = False  # 是否读取之前训练的模型
path = "./dqn" # 是存储模型的路径
h_size = 512 # at the end full-connected layers' implicitly nodes
tau = 0.001 # target DQN learning_rate when learning from The Main DQN

# 

mainQN = Qnetwork(h_size) # 初始化mainQN
targetQN = Qnetwork(h_size) # initialize the targetQN
init = tf.global_variables_initializer() # initialize all parameters


trainables = tf.train_variables() # acquire all the trainable parameters
targetOps = updateTargetGraph(trainables, tau) # updateTargetGraph CREATE the operation to update targetDQN's model parameters

