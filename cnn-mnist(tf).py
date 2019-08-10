import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
# load data 
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1,28,28,1])
# first layer
# linear part
W_conv_1 = tf.Variable(tf.truncated_normal([5,5,1,32], stddev = 0.1))
b_conv_1 = tf.Variable(tf.constant(value = 0.1,shape= [32]))
conv_1 = tf.nn.conv2d(x_image, W_conv_1, strides = [1,1,1,1], padding = 'SAME') + b_conv_1
# unlinear part
h_conv_1 = tf.nn.relu(conv_1)
h_pool_1 = tf.nn.max_pool(h_conv_1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME")
# second layer
W_conv_2 = tf.Variable(tf.truncated_normal([5,5,32,64], stddev = 0.1))
b_conv_2 = tf.Variable(tf.constant(value = 0.1,shape = [64]))
conv_2 = tf.nn.conv2d(h_pool_1, W_conv_2, strides = [1,1,1,1], padding = "SAME")
h_conv2 = tf.nn.relu(conv_2)
h_pool_2 = tf.nn.max_pool(h_conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = "SAME")
# flat layer
h_pool_2_flat = tf.reshape(h_pool_2, [-1, 7*7*64])
W_fc_1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev = 0.1))
b_fc_1 = tf.Variable(tf.constant(0.1, shape = [1024]))
h_fc_1 = tf.nn.relu(tf.matmul(h_pool_2_flat, W_fc_1) + b_fc_1)
# dropout regularization
keep_probability = tf.placeholder(tf.float32)
h_fc_1_drop = tf.nn.dropout(h_fc_1, keep_probability)
#final layer
W_fc_2 = tf.Variable(tf.truncated_normal([1024, 10], stddev = 0.1))
b_fc_2 = tf.Variable(tf.constant(0.1, shape = [10]))
# loss function and optimizer
logit_conv = tf.matmul(h_fc_1_drop, W_fc_2) + b_fc_2
y_conv = tf.nn.softmax(logit_conv)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logit_conv, labels = y))
train_step = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(cross_entropy)

correct_predictions = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(10000):
    batch_x, batch_y = mnist.train.next_batch(64)
    sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_probability: 0.5})
print(sess.run(accuracy, feed_dict = {x: mnist.test.images, y:mnist.test.labels, keep_probability: 1}))