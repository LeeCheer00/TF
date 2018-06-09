# coding=utf-8
import tensorflow as tf
b=tf.Variable(tf.zeros([100]))
W=tf.Variable(tf.random_uniform([784,100],-1,1)) # 生成784x100 的随机矩阵W
x=tf.placeholder(name="x")                       # 输入的Placeholder
relu=tf.nn.relu(tf.matmul(W,x)+b)                # ReLu(Wx+b)
C=[]                                          # 根据ReLu函数的结果计算Cost
s=tf.Session()
for step in range(0,10):
        input=...construct 100-D input array...  # 为输入创建一个100维的向量
        result=s.run(C, feed_dict={x: input})    # 获取Cost, 供给输入X
        print(step, result)
