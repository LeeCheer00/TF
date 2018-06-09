# coding=utf-8
import numpy as np 
import sklearn.preprocessing as  prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
def xavier_init(fan_in, fan_out, constant = 1):
        low = -constant * np.sqrt(6.0/ (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random_uniform((fan_in, fan_out),minval = low, maxval = high, dtype= tf.float32)
class AdditiveGaussianNoiseAutoencoder(object):
        def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer = tf.train.AdamOptimizer(), scale=0.1):
                self.n_input = n_input
                self.n_hidden = n_hidden
                self.transfer = transfer_function
                self.scale = tf.placeholder(tf.float32)
                self.training_scale = scale
                network_weights = self._initialize_weights()
                self.weights = network_weights
                self.x = tf.placeholder(tf.float32, [None, self.n_input])
                self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)),self.weights['w1']), self.weights['b1']))
                self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
                # self.cost
                self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
                self.optimizer = optimizer.minimize(self.cost) 

                init = tf.global_variables_initializer()
                self.sess = tf.Session()
                self.sess.run(init)
        # dict

        def _initialize_weights(self):
                all_weights = dict()
                all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
                all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype = tf.float32))
                all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype = tf.float32))
                all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype = tf.float32))
                return all_weights
        # partial_fit

        def partial_fit(self, X):
                cost, opt = self.sess.run((self.cost, self.optimizer),feed_dict = {self.x: X, self.scale: self.training_scale})
                return cost
        # calc_total_cost, the same to partial_fit, but won't triger the train.

        def calc_total_cost(self, X):
                return self.sess.run(self.cost, feed_dict = {self.x: X, self.scale: self.training_scale})
        # transform, 返回自编码隐藏层的输出结果，它的目的是提供一个接口来获取抽象后的特征，自编码器的隐藏层的最主要功能就是学习出数据中的高阶特征

        def transform(self, X):
                return self.sess.run(self.hidden, feed_dict = {self.x: X, self.scale: self.training_scale})
        # generate, 将隐含层的输出结果作为输入，将高阶特征复原为原始数据的步骤

        def generate(self, hidden =None):
                if hidden is None:
                        hidden = np.random.normal(size = self.weights["b1"])
                return self.sess.run(self.reconstruction, feed_dict = {self.hidden: hidden})
        # reconstruction 输入数据是原数据，输出是复原后的数据

        def reconstruction(self, X):
                return self.sess.run(self.reconstruction, feed_dict = {self.x: X, self.scale: self.training_scale})
        # 作用是获取隐含层的权重w1

        def getWeight(self):
                return self.sess.run(self.weights['w1'])
        # getBiases 作用是获取隐含层的偏置系数b1

        def getBiases(self):
                return self.sess.run(self.weights['b1'])
        # 至此，去噪自编码器的class就全部定义完了
        # 载入TF提供的MNIST数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
# StandardScaler

def standard_scale(X_train, X_test):
        preprocessor = prep.StandardScaler().fit(X_train)
        X_train = preprocessor.transform(X_train)
        X_test = preprocessor.transform(X_test)
        return X_train, X_test
# block 数据获取, 不放回抽样，提高了数据的利用率
def get_random_block_from_data(data, batch_size):
        start_index = np.random.randint(0, len(data) - batch_size)
        return data[start_index:(start_index + batch_size)]
# 使用之前定义standard_scale函数对训练集、测试集进行标准化变换

X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
# 定义几个参数，训练总数，batch_size is 128, 最大训练的轮数,并设置每隔一轮（epoch)就显示一次损失cost

n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1
# AGN 自编码实例， n_input == 784, 自编码的隐含层节点数 n_hidden 为200， 隐含层的激活函数 transfer_function 为 softplus, 优化器 Optimizer 为 Adam， learning_rate is 0.001, 噪声系数scale 设为 0.01
autoencoder = AdditiveGaussianNoiseAutoencoder(n_input = 784, n_hidden = 200, transfer_function = tf.nn.softplus, optimizer = tf.train.AdamOptimizer(learning_rate = 0.001), scale = 0.01)
# 调整batch_size、epoch 、 优化器、 自编码器的隐含层数、 隐含节点数等， 来尝试获得更低的cost
for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):
                batch_xs = get_random_block_from_data(X_train, batch_size)


                cost = autoencoder.partial_fit(batch_xs)
                avg_cost += cost / n_samples * batch_size


        if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))
