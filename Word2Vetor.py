# --coding: utf-8--
import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib
import tensorflow as tf

#定义下载文本数据的函数，这里使用urllib.request.urlretrieve下载数据的压缩文件并核对文件尺寸。
url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
            if not os.path.exists(filename):
                            filename, _ = urllib.request.urlretrieve(url + filename, filename)
            statinfo = os.stat(filename)
            if statinfo.st_size == expected_bytes:
                    print('Found and verified', filename)
            else:
                    print(statinfo.st_size)
                    raise Exception( 'Failed to verify ' + filename + '. Can you get to it with a browser?')
            return filename

filename = maybe_download('text8.zip', 31344016)

#解压下载的压缩文件使用tf.compat.as_str将数据转成单词的列表。
def read_data(filename):
            with zipfile.ZipFile(filename) as f:
                            data = tf.compat.as_str(f.read(f.namelist()[0])).split()
            return data


words = read_data(filename)
print('Data size', len(words))
vocabulary_size = 50000


def build_dataset(words):#创建vocabulary词汇表
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    #用collections.Counter统计单词列表中单词的频数。用most_common取top50000频数的单词作为vocabulay放入dictionary中。
    dictionary = dict()
    for word, _ in count:
                    dictionary[word] = len(dictionary)#将单词转为编号（以频数排序的编号）
    data = list()
    unk_count = 0
    for word in words:#遍历单词列表
            if word in dictionary:#对每个单词先判断是否出现在dictionary中，如果是则转为编号
                        index = dictionary[word]
            else:#如果不是则转为编号0。
                            index = 0
                            unk_count += 1
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary    #返回转换后的编码,每个单词的频数统计，词汇表，及其反转形式
data, count, dictionary, reverse_dictionary = build_dataset(words)
del words
print('Most common words(+UNK)', count[:5])

print('Sample data', data[:10], [reverse_dictionary[i] for i in dat[:10]])
data_index = 0

def generatt_batch(batch_size, num_skips, skip_window):#生成训练用的batch数据，
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    #skip_window为单词最远可以联系的距离，设为1代表只能跟紧邻的两个单词生成样本，
    #num_skips为对每个单词生成多少个样本。
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    #用np.ndarray将batch和labels初始化为数组。
    span = 2 * skip_window + 1 #span为对某个单词创建相关样本时会使用到的单词数量，包括目标单词本身和它前后的单词
    buffer = collections.deque(maxlen=span) #创建一个最大容量为span的deque(双向队列）
    for _ in range(span):#从data_index开始，把span个单词顺序读入buffer作为初始值
                buffer.append(data[data_index])
                data_index = (data_index + 1) % len(data)
    #第一层循环，每次循环内对一个目标单词生成样本。
    for i in range(batch_size // num_skips):
        target = skip_window  #buffer中第skip_window个变量为目标单词。
        targets_to_avoid = [ skip_window ] #targets_to_avoid为生成样本时需要避免的单词列表
        #第二层循环，每次循环对一个语境单词生成样本，先产生随机数，，
        for j in range(num_skips):
                    while target in targets_to_avoid:  #直到随机数不在targets_to_avoid中代表可以使用的语境单词，然后产生一个样本
                            target = random.randint(0, span - 1)
                    targets_to_avoid.append(target)
                    batch[i * num_skips + j] = buffer[skip_window]
                    labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


#调用generatt_batch函数测试一下其功能，
batch, labels = generatt_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
            print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0],
                                      reverse_dictionary[labels[i, 0]])
            batch_size = 128
embedding_size = 128 #为将单词转为稠密向量的维度
skip_window = 1
num_skips = 2


valid_size = 16 #用来抽取的验证单词数
valid_window = 100 #验证单词只从频数最高的100个单词中抽取，
valid_examples = np.random.choice(valid_window, valid_size, replace=False)  #valid_examples为验证数据集
num_sampled = 64 #训练时用来做负样本的噪声单词的数量


graph = tf.Graph()
with graph.as_default():
            
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32) #将前面随即产生的valid_examples转化为tensorflow中的constant
    
    with tf.device('/cpu:0'):
                    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
                    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        
        
        #nec loss作为训练的优化目标
                    nce_weights = tf.Variable(
                                    #初始化nec loss中的权中参数nce_weights
                                                tf.truncated_normal([vocabulary_size, embedding_size],
                                                                                        stddev=1.0 / math.sqrt(embedding_size)))
                    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    #使用tf.nn.nce_loss计算学习出现的词向量embedding在训练数据上的loss，并在 tf.reduve_mean进行汇总。
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=train_labels,
                                                     inputs=embed, num_sampled=num_sampled, num_classes=vocabulary_size))
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
#接着计算嵌入向量 embeddings 的L2范数norm
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    
    #使用tf.nn.embedding_lookup查询验证单词的嵌入向量，并计算验证单词的嵌入向量与词汇表中所有单词的相似性
    valid_embeddings = tf.nn.embedding_lookup(
                            normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
                                valid_embeddings, normalized_embeddings, transpose_b=True)
            
    init = tf.global_variables_initializer()   #初始化所有模型参数


num_steps = 100001 #最大迭代次数为10万次

with tf.Session(graph=graph) as session:
    init.run()
    print("Initialized")
    
    average_loss = 0
    for step in range(num_steps):
                    batch_inputs, batch_labels = generate_batch(
                                                batch_size, num_skips, skip_window)
                    feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}
        
                    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
                    average_loss += loss_val

#之后每2000次循环，计算一下平均loss并显示出来
    if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    print("Average loss at step ", step, ":", average_loss)
                    average_loss = 0
#每10000次循环，计算一次验证单词与全部单词的相似度，并将与每个验证单词最相似的8个单词显示出来
    if step % 10000 == 0:
        sim = similarity.eval()
        for i in range(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 8
            nearest = (-sim[i, :]).argsort()[1: top_k+1]
            log_str = "Nearest to %s: " % valid_word
            for k in range(top_k):
                close_word = reverse_dictionary[nearest[k]]
                log_str = "%s %s," %(log_str, close_word)
            print(log_str)
        final_embeddings = normalized_embeddings.eval()

def plot_with_labels(low_dim_embs, labels, filename='tsnem.png'):
        assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
        plt.figure(figure=(18, 18))
        for i, label in enumerate(labels):
                x, y = low_dim_embs[i, :]
                plt.scatter(x, y)
                plt.annotate(label, xy=(x, y), xytex=(5, 2), textcoords='offset points', ha='right', va='bottom')
        plt.savefig(filename)


        # 降维度, 将原始的128维嵌入向量降到2维
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
tsne =  TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 100 # 这里显示词频最高的100个单词的可视化结果
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
labels = [reverse_dictionary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)

