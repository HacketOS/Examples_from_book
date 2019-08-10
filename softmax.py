from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt

def _main():
    #load training data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
    # placeholder for variable train size
    X = tf.placeholder(tf.float32,[None, 784])
    # weights for each pixel for each number
    W = tf.Variable(tf.zeros([784, 10]))
    # bias
    b = tf.Variable(tf.zeros([10]))
    # softmax model
    y = tf.nn.softmax(tf.matmul(X,W) + b)
    # placeholder for true answers
    y_ = tf.placeholder(tf.float32, [None, 10])
    # loss function
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices= [1]))
    # gradient optimizer with learning rate 0.5 
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # correct answers
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    # accuracy score
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #initialization
    init = tf.initialize_all_variables()
    #create tf session
    sess = tf.Session()
    sess.run(init)
    #learning process
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={X: batch_xs, y_:batch_ys})
    
    print("Accuracy score - %s" %sess.run(accuracy,
     feed_dict = {X: mnist.test.images, y_: mnist.test.labels}))

if __name__ == "__main__":
    _main()