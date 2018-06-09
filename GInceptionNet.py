# encoding = utf-8
import tensorflow as tf
slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

# inception_v3_arg_scope
# L2正则的weight_decay = 0.00004
# stddev = 0.1
# batch_norm_var_collection 默认值为moving_vars
######
# batch normalizaton的参数字典
# 衰减系数decay为0.9997
# epsilon 为0.001
# updates_collection 为 tf.GrapahKeys.UPDATE_OPS
# variables_collections 和beta 和gamma 设置 None
# moving_mean和moving_variance均设置为前面的batch_norm_var_collection
######
######
# slim.arg_scope is a useful tool, give the vaules to parameters automatically
# slim.arg_scope 不需要重复设置参数
# slim.arg_scope 对卷积层生成函数slim.conv2d的几个参数赋予默认值
# trunc_normal(stddev)设置成权重初始化器
# slim.batch_norm, 标准化器设置为前面的batch_norm_params
# 返回scope
######
######
# slim.conv2d默认参数, 定义一个卷积层变得非常方便
# 一行代码定义一个conv
#####
def  inception_v3_arg_scope(weight_decay=0.00004, stddev=0.1, batch_norm_var_collection='moving_vars'):
        batch_norm_params = {
                        'decay': 0.9997,
                        'epsilon': 0.001,
                        'updates_collections': tf.GraphKeys.UPDATE_OPS,
                        'variables_collections':{
                                'beta': None,
                                'gamma': None,
                                'moving_mean': [batch_norm_var_collections],
                                'moving_variance':[batch_norm_var_collection],
                                } 
                        }
        with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(weight_decay)):
                with slim.arg_scope([slim.conv2d], weights_initializer=tf.truncated_normal_initializer(stddev=stddev),activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, normalizer_paramas=batch_norm_params) as sc:
                        return sc
#######
###
# slim.conv2d的第一个参数输入的tensor,第二个参数为输出的通道数，3th is conv size, 4th is stride, 5th is padding
###
######

# 5 convs, 2 mpools


def inception_v3_base(inputs, scope=None):

        end_points = {}
        with tf.variable_scope(scope, 'InceptionV3', [inputs]):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='VALID'):
                        net = slim.conv2d(inputss, 32, [3, 3], stride=2, scope='Conv2d_1a_3x3')
                        net = slim.conv2d(net, 32, [3, 3], scope='Conv2d_2a_3x3')
                        net = slim.conv2d(net, 64, [3, 3], padding='SAME',scope='Conv2d_2b_3x3')
                        net = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_3a_3x3')
                        net = slim.conv2d(net, 80, [1, 1], scope='Conv2d_3b_1x1')
                        net = slim.conv2d(net, 192, [3, 3], scope='Conv2d_4a_3x3')
                        net = slim.max_pool2d(netm [3, 3], stride=2, scope='MaxPool_5a_3x3')

####
# 4 branches 0-3
# 0. 64* 1*1 conv 
# 1. 48* 1*1 then 64* 5*5 conv
# 2. 96* 3*3 conv
# 3. 3*3 mpool then 32* 1*1 conv
# tf.concat 合并输出
# 64+64+96+32=256
# output is 35*35*256
# Inception Module.1 输出的图片尺寸均为35*35，But Inception Module last 2 will change the channels 
####
                with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
                        with tf.variable_scope('Mixed_5d'):
                                with tf.variable_scope('Branch_0'):
                                        branch_0 = slim.conv2d(net, 64, [1, 1], scop='Conv2d_0a_1x1')
                                with tf.variable_scope('Branch_1'):
                                        branch_1 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
                                        branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope='Conv2d_0b_5x5')
                                with tf.variable_scope('Branch_2'):
                                        branch_2 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                                        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d 0b 3x3')
                                        branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                                with tf.variable_scope('Branch_3'):
                                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                                        branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
                                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)


#####
###
#Inception Modules 2.
###
#####
#
# 5 Inception Mudule included
#
# 1st Inception Module was named as Mixed_6a
# 3 branches
# 384* 3*3 conv
# stride 2, padding = VALID, size = 17*17
# 2ed, 3 branches 64* 1*1 and 2* 96* 3*3 conv
# 3rd, stride=2, padding=VALID, 17*17*96
# 
#
# THE 3rd branches is 3*3 mpool, stride=2, padding=VALID, so the size output is 17*17*256
# use the TF.CONCAT combine the 3 branches into 1, the final output is 17*17*(384+96+256)=17*17*768
# the photo's size and channel have no changes.
#
#
                        with tf.variable_scope('Mixed_6a'):
                                with tf.variable_scope('Branch_0'):
                                        branch_0 = slim.conv2d(net, 384, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_1x1')
                                with tf.variable_scope('Branch_1'):
                                        branch_1 = slim.conv2d('net', 64, [1, 1], scope='Conv2d_0a_1x1')
                                        branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
                                        branch_1 = slim.cnov2d(branch_1, 96, [3, 3], stride=2, scope='Conv2d_1a_1x1')
                                with tf.variable_scope('Branch_2'):
                                        branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
                                net = tf.concat([branche_0, branch_1, branch_2], 3) 


#######
# Inception Modules 2.
####
# 2ed Inception --Mixed_6b
# 4 branches
# 1. 192* 1*1 conv
# 2. consist of 3 conv, 1.. 128* 1*1  2.. 128* 1*7 3.. 192* 7*1 Ｆactorization into small convolutions
# 参数量减少
# add one activation function add the non-liner
# 3. 5 conv, consist of  1*1 * 128channels , 7*1 *128, 1*7 *128, 7*1 * 128, 1*7 *192
# 4. 3*3 mpool, 1*1 *192
# tf.concat
# tensor size= 17*17*(192+192+192+192)=17*17*768
#######
                        with tf.variable_scope('Mixed_6b'):
                                with tf.variable_scope('Branch_0'):
                                        branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                                with tf.variable_scope('Branch_1'):
                                        branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                                        branch_1 = slim.conv2d(branch_1, 128, [1, 7], scope='Conv2d_0b_1x7')
                                        branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                                with tf.variable_scope('Branche_2'):
                                        branch_2 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
                                        branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0b_7x1')
                                        branch_2 = slim.conv2d(branch_2, 128, [1, 7], scope='Conv2d_0c_1x7')
                                        branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope='Conv2d_0d_7x1')
                                        branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                                with tf.Variable_scope('Branch_3'):
                                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                                        branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                                net = tf.concat([branch_0,  branch_1, branch_2, branch_3], 3)
#####
# Inception Module -- Mixed_6c
#####
# Similar to mixed_6b
# 2ed and 3rd branch channels is different
# 128 to 160
# the final out is no change, 192
# the others is the same 
# Inception module 每经过一个， the tensor size is no change, the feature is refined once,  the variable conv and non-linear have great contribution to the performance.
#####
                        with tf.variable_scope('Mixed_6c'):
                                with tf.variable_scope('Branch_0'):
                                        branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                                with tf.variable_scope('Branch_1'):
                                        branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                                        branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                                        branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                                with tf.variable_scope('Branch_2'):
                                        branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                                        branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                                        branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                                        branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                                        branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                                with tf.variable_scope('Branch_3'):
                                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                                        branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
####
# Inception Module --Mixed_6d
###
# the same to the --Mixed_6c
####
                        with  tf.variable_scope('Mixed_6d'):
                                with tf.variable_scope('Branch_0'):
                                        branch_0 = sliml.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                                with tf.variable_scope('Branch_1'):
                                        branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv22d_0a_1x1')
                                        branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope='Conv2d_0b_1x7')
                                        branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                                with tf.variable_scope('Branch_2'):
                                        branch_2 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
                                        branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0b_7x1')
                                        branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope='Conv2d_0c_1x7')
                                        branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope='Conv2d_0d_7x1')
                                        branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                                with tf.variable_scope('Branch_3'): 
                                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                                        branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)


############
## Mix_6e
##########
## the same to the before 
## Inception module 2. We put the Mixed_6e into  end_points, as the Auxiliary Classifier as "Classifier"
###########
                        
                        with tf.variable_scope('Mixed_6e'):
                                with tf.variable_scope('Branch_0'):
                                        branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                                with tf.variable_scope('Branch_1'):
                                        branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                                        branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_0b_1x7')
                                        branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                                with tf.variable_scope('Branch_2'):
                                        branch_2 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                                        branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0b_7x1')
                                        branch_2 = slim.conv2d(branch_2 ,192, [1, 7], scope='Conv2d_0c_1x7')
                                        branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0d_7x1')
                                        branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope='Conv2d_0e_1x7')
                                with tf.variable_scope('Branch_3'):
                                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                                        branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                        end_points['Mixed_6e'] = net
###########################################
##
## Inception  Module 3.
# 3 Inception modules
# the later 2 moudles are similar
# the 1st was named as Mixed_7a, 3 branches were included
# 1..1*1 * 192 conv     3*3 * 320 conv stride=2, padding=VALID     2.. 1*1 * 192, 1*7 * 192, 7*1 * 192, 3*3 * 192, stride = 2, padding = VALID, the output is 8*8 * 192
# the 3rd branch is 3*3  mpool, stride = 2.padding=VALID output is  8*8 * 768
# tf.concat combine together The channels, 输出的tensor size 为8*8*(320+192+768)=8*8*1280
# Inception Module begins, the output size photoes is reduced , the channel is add up, tensor size is dropping.
##########################################

                        with tf.variable_scope('Mixed_7a'):
                                with tf.variable_scope('Branch_0'): 
                                        branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                                        branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                                with tf.variable_scope('Branch_1'):
                                        branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
                                        branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope='Conv2d_ob_1x7')
                                        branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope='Conv2d_0c_7x1')
                                        branch_1 = slim.conv2d(branch_1, 192, [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                                with tf.variable_socpe('Branch_2'):
                                        branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
                                net = tf.concat([branch_0, branch_1, branch_2], 3)


########################################
##
# Inception Module 3.
# 2 Module
# 4 branches
# 1..   1*1 *320 
# 2..   1*1 * 384 convs, and give 2 branches, 1*3 *384 and another is 3*1 * 384, then tf.concat combine the 2 branches, the outout size of tensor is 8*8 *(384+384)=8*8* 768
# 3..   1*1 * 448 convs, then 3*3 * 384, dvivided into 2 branches: 1*3 * 384 and 3*1 * 384, and the combined as 8*8*768
# 4..   3*3 mpool, 1*1 * 192convs,
# tf.concat the 4 branches, the tensor size is 8*8*(320+768+768+192)=8*8*2048
# the output is 2048 Channels
#######################################

                        with tf.variable_scope('Miexed_7b'):
                                with tf.variable_scope('Branch_0'):
                                        branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
                                with tf.variable_scope('Branch_1'):
                                        branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                                        branch_1 = tf.concat([slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'), slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0b_3x1')], 3)
                                with tf.variable_scope('Branch_2'):
                                        branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
                                        branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
                                        branch_2 = tf.concat([slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'), slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')], 3)
                                with tf.variable_scope('Branch_3'):
                                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                                        branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

#####################
# Mixed_7c is the last Inception Module
# But it's the copy version of the Mixed_7b, the output size is 8*8*2048
#####
# Inception Module is the result , and we return it as the inception_v3_base function's final result
#####################

                        with tf.varialbe_scope('Mixed_7c'):
                                with tf.variable_scope('Branch_0'):
                                        branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
                                with tf.variable_scope('Branch_1'):
                                        branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
                                        branch_1 = tf.concat([slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'), slim.conv2d(branch_1, 384, [3 ,1], scope='Conv2d_0c_3x1')], 3)
                                with tf.variale_scope('Branch_2'):
                                        branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
                                        branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv3d_0b_3x3')
                                        branch_2 = tf.concat([slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'), slim.conv2d(branch_2, 384, [3, 1], scope= 'Conv2d_0d_3x1')], 3)
                                with tf.variale_scope('Branch_3'):
                                        branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                                        branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')
                                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
                        return net, end_points


                                
