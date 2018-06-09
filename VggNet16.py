# encoding=utf-8
from datetime import datetime
import math
import time
import tensorflow as tf

# 创建conv_op
# 把本层参数存入参数列表
# kh = kernel height kw=kernel width
# tf.get_variable创建，shape就是[kh, kw, n_in, n_out]
# tf.contrib.layers.xavier_initializer_conv2d()做初始化

def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
        n_in = input_op.get_shape()[-1].value

        with tf.name_scope(name) as scope:
                kernel = tf.get_variable(scope+"w", shape=[kh, kw, n_in, n_out], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
                
                # 对input_op进行卷积处理
                # 将创建卷积层时用到的kernel和biases添加进参数列表
                # 并将卷积层的输出activation作为函数结果返回

                conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
                bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
                biases = tf.Variable(bias_init_val, trainable=True, name='b')
                z = tf.nn.bias_add(conv, biases)
                activation = tf.nn.relu(z, name=scope)
                p += [kernel, biases]
                return activation

# 全连接层fc_op (input_op的通道数 )同样参数初始化方法用xavier_initializer
# x_i 赋予一个较小的值0.1 避免dead neuron
# kernel、bias 
# ReLu activation 添加到参数p
# return activation

def fc_op(input_op, name, n_out, p):
        n_in = input_op.get_shape()[-1].value

        with tf.name_scope(name) as scope:
                kernel = tf.get_variable(scope+"w", shape= [n_in, n_out], dtype= tf.float32, initializer= tf.contrib.layers.xavier_initializer_conv2d())
                biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
                activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
                p += [kernel, biases]
                return activation

# mpool_op
# use da tf.nn.max_pool(input_op, kh*kw, stride dh*dw, padding SAME)

def mpool_op(input_op, name, kh, kw, dh, dw):
        return tf.nn.max_pool(input_op,ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding= 'SAME', name=name)

# inference_op
# conv_op、mpool_op 来构建224*224*3, 224*224*64, 112*112*64
def inference_op(input_op, keep_prob):
        p = []

        conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
        conv1_2 = conv_op(conv1_1, name="conv1_2", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
        pool1 = mpool_op(conv1_2, name="pool1", kh=2, kw=2, dw=2, dh=2)

# 第二段卷积网络, 2*conv, 1*mpool
# conv 3*3,output 128 channels
# mpool ouput 1/2 56*56*128

        conv2_1 = conv_op(pool1, name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
        conv2_2 = conv_op(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
        pool2 = mpool_op(conv2_2, name="pool2", kh=2, kw=2, dh=2, dw=2)

# 第三段convNet, 3*conv, 1*mpool
# 3*3, output 256 channels
# mpool output 1/2 28*28*256
        
        conv3_1 = conv_op(pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
        conv3_2 = conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out= 256, dh=1, dw=1, p=p)
        conv3_3 = conv_op(conv3_2, name="conv3_3", kh=3, kw=3, n_out= 256, dh=1, dw=1, p=p)
        pool3 = mpool_op(conv3_3, name="pool3", kh=2, kw=2, dh=2, dw=2) 

# 第四段ConvNet， 3*conv, 1*mpool
# 3*3 output 526 channels
# mpool output 1/2 14*14*526, S reduced 1/4, channels doubled, but tensor reduced 1/2
        conv4_1 = conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
        conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
        conv4_3 = conv_op(conv4_2, name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
        pool4 = mpool_op(conv4_3, name="pool4", kh=2,  kw=2, dh=2, dw=2)

# The 5th and the last ConvNet
# channels is 256 remains
# mpool 2*2, stride 2*2, the final output 7*7*512
        
        conv5_1 = conv_op(pool4, name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
        conv5_2 = conv_op(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
        conv5_3 = conv_op(conv5_2, name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
        pool5 = mpool_op(conv5_3, name="pool5", kh=2, kw=2, dw=2, dh=2)
        
# flatten
# tf.shape trans it into 1 demension
# 7*7*512=25088
        shp = pool5.get_shape()
        falttened_shape = shp[1].value * shp[2].value * shp[3].value
        resh1 = tf.reshape(pool5, [-1, falttened_shape], name="resh1")

# 然后连接一个4096的全连接层，激活函数为ReLU，然后连接一个Dropout层，训练保留率0.5, 预测为1.0
        fc6 = fc_op(resh1, name="fc6", n_out=4096, p=p)
        fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc6_drop")

# fc_op7 全连接层， 后连接一个dropout
        fc7 = fc_op(fc6_drop, name="fc7", n_out=4096, p=p)
        fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc7_drop")


# 全连接层
# 1000个输出结点, 并使用softmax输出概率
# tf.argmax 求输出概率最大的类别
# fc8, softmax, predictions & p returned together
        fc8 = fc_op(fc7_drop, name="fc8", n_out= 1000, p=p)
        softmax = tf.nn.softmax(fc8)
        predictions = tf.argmax(softmax, 1)
        return predictions, softmax, fc8, p
# sesseion.run(),adding feed_dict, 方便后面传入keep_prob 来控制dropout层的保留比率
def time_tensorflow_run(session, target, feed, info_string):
        num_steps_burn_in = 10
        total_duration = 0.0
        total_duration_squared = 0.0
        for i in range (num_batches + num_steps_burn_in):
                start_time = time.time()
                _ = session.run(target, feed_dict = feed)
                duration = time.time() - start_time
                if i >= num_steps_burn_in:
                        if not i % 10 :
                                print ('%s: step %d, duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))
                        total_duration += duration
                        total_duration_squared += duration * duration
        mn = total_duration / num_batches
        vr = total_duration_squared / num_batches - mn * mn
        sd = math.sqrt(vr)
        print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' % (datetime.now(), info_string, num_batches, mn, sd))

# run_benchmark(), 方法与AlexNet, 通过tf.random_normal 函数生成标准差为0.1 的正态分布的随机图片
def run_benchmark():
        with tf.Graph().as_default():
                image_size = 224
                images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3], dtype=tf.float32,stddev=1e-1))
                # 创建keep_prob 的placeholder
                # 调用inference_op函数构建VGGNet-16的网络结构，获得predictions、softmax、fc8和参数列表p
                keep_prob = tf.placeholder(tf.float32)
                predictions, softmax, fc8, p = inference_op(images, keep_prob)
                # 然后创建Session并初始化全局参数
                init = tf.global_variables_initializer()
                sess = tf.Session()
                sess.run(init)

                # keep_prob 设为1.0, 执行预测， 与之前训练时的dropout0.5形成了反差
                # fc8 的 l2 loss 
                # tf.gradients 求相对与这个loss的所有模型额参数的梯度
                # time_tensorflow_run评测backward运算时间,这里target为求解梯度的操作grad，keep_prob为0.5
                time_tensorflow_run(sess, predictions, {keep_prob:1.0}, "Forward")
                objective = tf.nn.l2_loss(fc8)
                grad = tf.gradients(objective ,p)
                time_tensorflow_run(sess, grad, {keep_prob:0.5}, "Forward-backward")

# batch-size 设置 32， VggNet-16比较大，显存不够用
# VGGNet-16 在Tensorflow 上的forward 和 backward耗时
batch_size = 8
num_batches = 100
run_benchmark() 
