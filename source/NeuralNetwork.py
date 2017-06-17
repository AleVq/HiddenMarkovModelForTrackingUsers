import tensorflow as tf
print('Tensorflow version: ', tf.__version__)
from DataParser import parse_data
import numpy as np
tf.set_random_seed(64)


class NeuralNetwork:

    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        # defining placeholders (input's dimensions)
        self.X = tf.placeholder(tf.float32, [None, self.features])
        self.Y_ = tf.placeholder(tf.float32, [None, self.targets])
        # layers' sizes
        l1 = features * 3
        l2 = features * 4
        l3 = features * 3
        # building layers with randomized values
        self.w1 = tf.Variable(tf.truncated_normal([self.features, l1], stddev=0.1))
        self.b1 = tf.Variable(tf.ones([l1]))
        self.w2 = tf.Variable(tf.truncated_normal([l1, l2], stddev=0.1))
        self.b2 = tf.Variable(tf.ones([l2]))
        self.w3 = tf.Variable(tf.truncated_normal([l2, l3], stddev=0.1))
        self.b3 = tf.Variable(tf.ones([l3]))
        self.w4 = tf.Variable(tf.truncated_normal([l3, self.targets], stddev=0.1))
        self.b4 = tf.Variable(tf.zeros([self.targets]))
        # model, layers' activation function
        self.y1 = tf.nn.sigmoid(tf.matmul(self.X, self.w1) + self.b1)
        self.y2 = tf.nn.sigmoid(tf.matmul(self.y1, self.w2) + self.b2)
        self.y3 = tf.nn.sigmoid(tf.matmul(self.y2, self.w3) + self.b3)
        self.y_scores = tf.nn.sigmoid(tf.matmul(self.y3, self.w4) + self.b4)
        self.y = tf.nn.softmax(self.y_scores)
        self.init = tf.global_variables_initializer()  # initialize all Variables: weights and biases
        # loss function
        self.cross_entropy = - tf.reduce_sum(self.Y_ * tf.log(self.y))
        # training variables
        self.optimizer = tf.train.GradientDescentOptimizer(0.003)
        self.train_step = self.optimizer.minimize(self.cross_entropy)
        self.model = tf.train.Saver()

    def train(self, training_x, training_y):
        train_data = {self.X: training_x, self.Y_: training_y}
        sess = tf.Session()
        sess.run(self.init)
        sess.run(self.train_step, feed_dict=train_data)
        # accuracy of the trained model, between 0 (worst) and 1 (best)
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.Y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.model.save(sess, "/tmp/model.ckpt")

    # input: input vector, supposed to be np.array
    def get_prediction(self, x):
        with tf.Session() as sess:
            self.model.restore(sess, "/tmp/model.ckpt")
            pred = tf.get_default_graph().get_tensor_by_name('Softmax:0')
            result = sess.run(pred, feed_dict={self.X: x})
            return result