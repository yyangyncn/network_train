from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

session = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
# y是预测的概率分布，y_是真实的概率分布（即Label的one-hot编码）
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# GD
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 全局参数优化
tf.global_variables_initializer().run()
# 选一个部分迭代
for i in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(100)

    train_step.run({x: batch_xs, y_: batch_ys})

# batch_xs = mnist.train.images
# batch_ys = mnist.train.labels
# train_step.run({x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))