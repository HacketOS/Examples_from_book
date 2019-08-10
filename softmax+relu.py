from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt

def _main():

    mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

    X = tf.placeholder(tf.float32,[None, 784])

    # random params initialization
    W_relu = tf.Variable(tf.truncated_normal([784, 100], stddev = 0.1))
    b_relu = tf.Variable(tf.truncated_normal([100], stddev = 0.1))
    # adding hiden layer
    h = tf.nn.relu(tf.matmul(X, W_relu) + b_relu)
    #probability of keeping output
    keep_probability = tf.placeholder(tf.float32)
    # dropout layer
    h_drop = tf.nn.dropout(h, keep_probability)
    

    W = tf.Variable(tf.zeros([100, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(h_drop,W) + b)
    y_ = tf.placeholder(tf.float32, [None, 10])

    logit = tf.matmul(h_drop, W) + b

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logit,labels = y_))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    for i in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={X: batch_xs, y_:batch_ys, keep_probability: 0.5})
    
    print("Accuracy score - %s" %sess.run(accuracy,
     feed_dict={X:mnist.test.images, y_: mnist.test.labels, keep_probability: 0.5}))

    


if __name__ == "__main__":
    _main()