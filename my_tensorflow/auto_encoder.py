import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    hight = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=hight, dtype=tf.float32)

class AdditiveGaussianNoiseAutoencoder(object):

    #  n_input: 输入变量数, n_hidden：隐藏层数, transfer_function=tf.nn.softplus：隐藏层激活函数,
    #             optimizer=tf.train.AdamOptimizer()：优化器, scale=0.1：高斯噪声系数
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        # 网络结构
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        # 隐藏层： 噪声化x —— 乘上隐藏层权重w1，加上偏置b1 —— transfer激活
        self.hidden = self.transfer(tf.add(tf.matmul(
            self.x + scale * tf.random_normal((n_input,)), self.weights['w1']), self.weights['b1']))
        # 重构: 不需要激活
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # 损失函数：平方误差 squared error
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    # 计算损失及执行一步的函数
    # 用一个batch数据进行训练并返回当前的损失cost
    def partial_fit(self, X):
        cost, optimizer = self.session.run((self.cost, self.optimizer),
                                           feed_dict={self.x: X, self.scale: self.training_scale})
        return cost

    # 只进行计算损失的函数，在编码完毕后测试时会用上
    def calc_total_cost(self, X):
        return self.session.run(self.cost, feed_dict={self.x: X, self.scale: self.training_scale})

    # transform: 返回自编码隐藏层的输出结果。提供一个接口来获取抽象后的特征，自编码的隐藏层最主要的功能就是学习
    # 出数据中的高阶特征
    def transform(self, X):
        return self.session.run(self.hidden, feed_dict={self.x: X, self.scale: self.training_scale})

    # generate: 将隐藏层输出作为输入，通过之后的重建层复原为原始数据
    def generate(self, hidden = None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.session.run(self.reconstruction, feed_dict={self.hidden: hidden})

    # reconstruct: 整体运行一遍复原过程，包括提取高阶特征和通过高阶特征复原数据
    def reconstruct(self, X):
        return self.session.run(self.reconstruction, feed_dict={self.x: X, self.scale: self.training_scale})

    # 获取隐藏层的权重w1
    def getWeights(self):
        return self.session.run(self.weights['w1'])

    # 获取隐藏层的偏置b1
    def getBiases(self):
        return self.session.run(self.weights['b1'])

# 读入数据
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 标准化：让数据变成0均值，1标准差的分布
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    x_train = preprocessor.transform(X_train)
    x_test = preprocessor.transform(X_test)
    return x_train, x_test

# get_random_block: 获取随机的block数据
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

# 使用standard_scale对训练集、测试集进行标准化
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

# 定义几个常用参数
n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1

# 创建一个实例
autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=784, n_hidden=200,
                                               transfer_function=tf.nn.softplus,
                                               optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                               scale=0.01)

# 开始训练
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)
        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost / n_samples * batch_size

    if epoch % display_step == 0:
        print("Epoch:", '%0.4d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

# 性能评测
print("Total cost:" + str(autoencoder.calc_total_cost(X_test)))