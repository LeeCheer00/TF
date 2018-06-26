#coding=utf-8
import math
import collections
from datetime import datetime
import time
import tensorflow as tf
slim = tf.contrib.slim

class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
        'A named tuple describing a ResNet block'   # tuple create the Block class.3 parameters include scope、unit_fn、args, bottleneck is a unit in Residual learning unit(depth, depth_bottleneck, stride)

def subsample(inputs, factor, scope=None):   # 降采样 输入、采样因子、scope, factor=1 return inputs straight, if it's not 1 use the slim.max_pool2d，1x1 pool, stride we could subsample
        if factor == 1:
                return inputs
        else:
                return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)
        
def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
        if stride == 1:
                return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, padding='SAME', scope=scope) # if the stirde=1, conv2d=SAME
        else: # stride!=1, pad zero 's total number = kernel-1
                pad_total = kernel_size - 1
                pad_beg = pad_total // 2 # pad_bag = pad/2
                pad_end = pad_total - pad_beg # pad_end is rest part
                inputs = tf.pad(inputs, [[0, 0], [pad_beg,  pad_end], [pad_beg, pad_end], [0, 0]]) # tf.pad add The ‘0’, zero padding, the  VALID padding mode build the conv
                return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride, padding='VALID', scope=scope)
@slim.add_arg_scope
def stack_blocks_dense(net, blocks, outputs_collections=None):    # stack the stack_block_dense function is begining
        for block in blocks: # block is the class list of the Block
                with tf.variable_scope(block.scope, 'block', [net]) as sc: # Res learning unit named block1/unit1
                        for i, unit in enumerate(block.args):  # We got every block's Residual unit 's args, and unfold it depth、depth_bottleneck、stride
                                with tf.variable_scope('unit_%d' %(i+1), values=[net]):
                                        unit_depth, unit_depth_bottleneck, unit_stride = unit
                                        net = block.unit_fn(net, depth=unit_depth, depth_bottleneck=unit_depth_bottleneck, stride=unit_stride) # unit_fn generator of ResNet
                        net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net) # at the last , we use  slim.collect_named_outputs add the Net into the COLLECTION
        return net
        

def resnet_arg_scope(is_training=True, weight_decay=0.0001, batch_norm_decay=0.997, batch_norm_epsilon=1e-5, batch_norm_scale=True): # arg_scope is used to define the paramters as default
        batch_norm_params={'is_training': is_training, 'decay':batch_norm_decay, 'epsilon':batch_norm_epsilon, 'scale':batch_norm_scale, 'updates_collections':tf.GraphKeys.UPDATE_OPS,
                        }
        with slim.arg_scope(
                        [slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=slim.variance_scaling_initializer(),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
                with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                        with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc: # maxpool's padding=SAME, it's good for alignment
                                return arg_sc
@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, outputs_collections=None, scope=None): #  The difference between the  V1 and V2: BN in every layer, 2. preactivation not in the conv
        with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
                depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4) # 先使用s.u.l._d 来获取输入的最后一个维度，输出通道数 min_rank表示限定最少为4个维度
                preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact') # s.b.n to BN, relu to Preactivate
                if depth == depth_in:
                        shortcut = subsample(inputs, stride, 'shortcut') # directed connect X
                else: # if the channels input and output is not equal , we use the stride 1x1 conv to change the channels number
                        shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride, normalizer_fn=None, activation_fn=None, scope='shortcut')

                residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1, scope='conv1') # residual have 3 layers. 1x1 stride=1, output = depth_bottleneck conv
                residual = conv2d_same(residual, depth_bottleneck, 3, stride, scope='conv2') # 3x3 size stride, channels=depth_bottleneck 
                residual = slim.conv2d(residual, depth, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='conv3') # 1x1, stride=1, output=depth conv, no l2 regualrization and no activation
                output = shortcut + residual #  shortcut + residual = output
                return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)   # s.u.c.n.o add into collection and return result



# ResNet 主函数
def resnet_v2(inputs, blocks, num_classes=None, global_pool=True, include_root_block=True, reuse=None, scope=None): # pre-define the res learning module, inputs is INPUT, blocks is the Block's list, num_classes is the output class, global_pool 
        with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc: #  reuse 标志 if it reuse, scope is the network's name ,'inlcude_root_block' shows that if add the 7x7 conv and mpool before the ResNet
                end_points_collection = sc.original_name_scope + '_end_points'
                with slim.arg_scope([slim.conv2d, bottleneck, stack_blocks_dense], outputs_collections=end_points_collection): # slim.conv2d, bottleneck, stack_b_d  3 paramters are set defaulted  as 'end_points_collection'
                        net = inputs
                        if include_root_block: # create the 64channels stride=2, 7x7 conv, and the stride 2, 3x3 mpool
                                with slim.arg_scope([slim.conv2d], activation_fn=None,normalizer_fn=None):
                                        net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
                                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1') # after 2 stride2 pool, size were reduced  to 1/4
                        net = stack_blocks_dense(net, blocks) # s_b_d to generator the resnet module,  add the max pool layer  according to  the average pool
                        net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
                        if global_pool:
                                net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims = True) # tf.reduce_mean  is better than the avg_pool
                        if num_classes is not None: # we add the  num_classes conv to add 1x1* num_classes (channels) 
                                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')
                        end_points = slim.utils.convert_collection_to_dict(end_points_collection) # slim.utils.convert_c_t_d to change collection into Python dict 
                        if num_classes is not None:
                                end_points['predictions'] = slim.softmax(net, scope='predictions') #add softmax outlet the result, 
                        return net, end_points
def resnet_v2_50(inputs, num_classes=None, global_pool=True, reuse=None, scope='resnet_v2_50'):
        block = [
                        Block('block1', bottleneck, [(256, 64, 1)]*2 + [256, 64, 2]),
                        Block('block2', bottleneck, [(512, 128, 1)]*3 + [512, 128, 2]),
                        Block('block3', bottleneck, [(1024, 256, 1)]*5 + [1024,256, 2]),
                        Block('block4', bottleneck, [(2048, 512, 1)] *3) ]
        return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)
def resnet_v2_101(inputs, num_classes=None, global_pool=True, reuse=None, scope='resnet_v2_101'):
        blocks =[
                        Block('block1', bottleneck, [(256, 64, 1)]*2+ [(256, 64, 2)]),
                        Block('block2', bottleneck, [(512, 128, 1)]*3 + [(512, 128, 2)]),
                        Block('block3', bottleneck, [(1024, 256, 1)]*22+ [(1024, 256, 2)]),
                        Block('block4', bottleneck, [(2048, 512, 1)]*3 ) ]
        return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)
'''
def resnet_v2_152(inputs, num_classes=None, global_pool=True, reuse=None, scope='resnet_v2_152'):
        blocks = [
                        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [256, 64, 2]),
                        Block('block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]), 
                        Block('block3', bottleneck, [(1024, 256, 1)] * 35+ [(1024, 256, 2)]), 
                        Block('block4', bottleneck, [(2048, 512, 1)] * 3) ]
        return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)
'''
def resnet_v2_152(inputs, num_classes=None, global_pool=True, reuse=None, scope='resnet_v2_152'):
    blocks = [
        Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
        Block('block2', bottleneck, [(512, 128, 1)] * 7 + [(512, 128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 35 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)
    ]
    return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True,
                     reuse=reuse, scope=scope)


def resnet_v2_200(inputs, num_classes=None, global_pool=True, reuse=None, scope='resnet_v2_200'):
        blocks = [
                        Block('blocks1', bottleneck, [(256, 64, 1)] * 2 + [256, 64, 2]),
                        Block('blocks2', bottleneck, [(512, 128, 1)] * 23 + [512, 128, 2]),
                        Block('blocks3', bottleneck, [(1024,256, 1)] * 35 + [1024, 256, 2]),
                        Block('blocks4', bottleneck, [(2048, 512, 1)] * 3)]
        return resnet_v2(inputs, blocks, num_classes, global_pool, include_root_block=True, reuse=reuse, scope=scope)

def time_tensorflow_run(session, target, info_string):
        num_steps_burn_in = 10
        total_duration = 0.0
        total_duration_squared = 0.0
        for i in range(num_batches  + num_steps_burn_in):
                #num_batchaes+num_steps_burn_in 次迭代以计算
                #time.time()记录时间，每10轮迭代后显示当前迭代所需要的时间
                #每轮将total_duration和total_duration_squared累加，为了后面计算每轮耗时的均值和标准差
                #

                start_time = time.time()
                _ = session.run(target)
                duration = time.time() - start_time
                if i >= num_steps_burn_in:
                        if not i % 10:
                                print('%s : step %d, duration = %.3f' % (datetime.now(), i -  num_steps_burn_in, duration ))
                        total_duration += duration
                        total_duration_squared += duration * duration

# 计算每轮迭代的平均耗时mn和标准差sd，显示
#""" 这样就完成了每轮迭代耗时的评测函数time_tensorflow_run """
        mn = total_duration / num_batches
        vr = total_duration_squared / num_batches - mn * mn
        sd = math.sqrt(vr)
        print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' % (datetime.now(), info_string, num_batches, mn, sd))


batch_size= 8
height, width =224, 224
inputs = tf.random_uniform((batch_size, height, width, 3))
#with slim.arg_scope(resnet_arg_scope(is_training=False)):
#        net, end_points = resnet_v2_152(inputs, 1000)
with slim.arg_scope(resnet_arg_scope(is_training=False)):
    net, end_points = resnet_v2_152(inputs, 1000)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
num_batches=1000
time_tensorflow_run(sess, net, "Forward")
