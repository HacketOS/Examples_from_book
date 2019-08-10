import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def create_ae(latent_space, batch_size, learning_rate): 
    ae_weights ={"encoder_w": tf.Variable(tf.truncated_normal([784, latent_space], stddev=0.1)),
                "encoder_b": tf.Variable(tf.truncated_normal([latent_space], stddev = 0.1)),
                "decoder_w": tf.Variable(tf.truncated_normal([latent_space, 784], stddev = 0.1)),
                "decoder_b": tf.Variable(tf.truncated_normal([784], stddev = 0.1))}

    ae_input = tf.placeholder(tf.float32, [batch_size,784])
    hidden = tf.nn.sigmoid(tf.matmul(ae_input, ae_weights["encoder_w"]) + ae_weights["encoder_b"])
    visible_logit = tf.matmul(hidden, ae_weights["decoder_w"]) + ae_weights["decoder_b"]
    visible = tf.nn.sigmoid(visible_logit)
    ae_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=visible_logit,
                                                                        labels=ae_input))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    ae_op = optimizer.minimize(ae_cost)
    return ae_op, ae_input

def _main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    batch_size = 64
    latent_space = 128
    learning_rate = 0.1
    ae_op, ae_input = create_ae(latent_space = latent_space, batch_size = batch_size, learning_rate = learning_rate)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    for i in np.arange(10000):
        x_batch, _ = mnist.train.next_batch(batch_size)
        sess.run(ae_op, feed_dict={ae_input: x_batch})

        
if __name__ == '__main__': 
    _main()