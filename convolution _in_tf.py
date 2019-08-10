import tensorflow as tf
import numpy as np 

x_inp = tf.placeholder(tf.float32 , [5,5])
w_inp = tf.placeholder(tf.float32 , [3,3])

# because input data should be [batch_size, high, width, channels]
X = tf.reshape(x_inp, [1,5,5,1])
# and the weight tensor - [high, width, chanels_input, chanels_output]
W = tf.reshape(w_inp, [3,3,1,1])
# shapes for grayscale images [5,5]

x_valid = tf.nn.conv2d(X, W,strides = [1,1,1,1], padding = 'VALID')
x_same = tf.nn.conv2d(X, W, strides = [1,1,1,1], padding = 'SAME')
x_valid_half = tf.nn.conv2d(X, W, strides = [1,2,2,1], padding = 'VALID')
x_same_half = tf.nn.conv2d(X, W, strides = [1,2,2,1], padding = 'SAME')

x = np.array([[0,1,2,1,0],[4,1,0,1,0],[2,0,1,1,1],[1,2,3,1,0],[0,4,3,2,0]])
w = ([[0,1,0],[1,0,1],[2,1,0]])

sess = tf.Session()
y_valid, y_same, y_valid_half, y_same_half = sess.run([x_valid, x_same, x_valid_half, x_same_half], 
feed_dict = {x_inp: x, w_inp: w})

print("y_valid: \n", y_valid[0,:,:,0])
print("y_same: \n", y_same[0,:,:,0])
print("y_valid_half: \n", y_valid_half[0,:,:,0])
print("y_same_half: \n", y_same_half[0,:,:,0])
