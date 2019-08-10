import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

# описание обычного слоя {грубо говоря записываем функция активации в каждом нейроне, что бы потом оптимизировать}
def fullyconnected_layer(tensor, input_size, out_size):
    W = tf.Variable(tf.truncated_normal([input_size, out_size], stddev = 0.1))
    b = tf.Variable(tf.truncated_normal([out_size], stddev = 0.1))
    return tf.nn.tanh(tf.matmul(tensor,W) + b)

#слой нормализации по минибатчам
def batch_norm(tensor, size):
    batch_mean, batch_var = tf.nn.moments(tensor, [0])
    beta = tf.Variable(tf.zeros(size))
    scale = tf.Variable(tf.ones(size))
    return tf.nn.batch_normalization(
        tensor, batch_mean, batch_var, beta,scale, 0.001 )

def _main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    # layers
    h1 = fullyconnected_layer(x, 784, 100)
    h1_batchnorm = batch_norm(h1, 100)
    h2 = fullyconnected_layer(h1_batchnorm, 100, 100)
    y_logit  = fullyconnected_layer(h2, 100, 10)
    # loss function
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = y_logit, labels = y)
    train_optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y_logit,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    for i in range(1000):
        batch_x, batch_y = mnist.train.next_batch(1000)
        sess.run(train_optimizer, feed_dict={x:batch_x, y:batch_y})

    print("Accuracy score - %s" %sess.run(accuracy,
     feed_dict = {x: mnist.test.images, y: mnist.test.labels}))
if __name__ == "__main__":
    _main()