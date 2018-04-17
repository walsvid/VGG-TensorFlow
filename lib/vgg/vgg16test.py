import tensorflow as tf
import numpy as np
from lib.networks.base_network import Net


class VGG16_test(Net):
    def __init__(self, cfg_):
        super().__init__(cfg_)
        self.x = tf.placeholder(tf.float32, name='x', shape=[1,
                                                             self.config.image_width,
                                                             self.config.image_height,
                                                             self.config.image_depth])
        self.y = tf.placeholder(tf.int16, name='y', shape=[1, self.config.n_classes])

    def init_saver(self):
        pass

    def conv(self, layer_name, bottom, out_channels, kernel_size=[3, 3], stride=[1, 1, 1, 1]):
        in_channels = bottom.get_shape()[-1]
        with tf.variable_scope(layer_name):
            w = tf.get_variable(name='weights',
                                shape=[kernel_size[0], kernel_size[1],
                                       in_channels, out_channels],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='biases',
                                shape=[out_channels],
                                initializer=tf.constant_initializer(0.0))
            bottom = tf.nn.conv2d(bottom, w, stride, padding='SAME', name='conv')
            bottom = tf.nn.bias_add(bottom, b, name='bias_add')
            bottom = tf.nn.relu(bottom, name='relu')
            return bottom

    def pool(self, layer_name, bottom, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True):
        with tf.name_scope(layer_name):
            if is_max_pool:
                bottom = tf.nn.max_pool(bottom, kernel, stride, padding='SAME', name=layer_name)
            else:
                bottom = tf.nn.avg_pool(bottom, kernel, stride, padding='SAME', name=layer_name)
            return bottom

    def fc(self, layer_name, bottom, out_nodes):
        shape = bottom.get_shape()
        if len(shape) == 4:  # x is 4D tensor
            size = shape[1].value * shape[2].value * shape[3].value
        else:  # x has already flattened
            size = shape[-1].value
        with tf.variable_scope(layer_name):
            w = tf.get_variable('weights',
                                shape=[size, out_nodes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('biases',
                                shape=[out_nodes],
                                initializer=tf.constant_initializer(0.0))
            flat_x = tf.reshape(bottom, [-1, size])
            bottom = tf.nn.bias_add(tf.matmul(flat_x, w), b)
            bottom = tf.nn.relu(bottom)
            return bottom

    def bn(self, layer_name, bottom):
        with tf.name_scope(layer_name):
            epsilon = 1e-3
            batch_mean, batch_var = tf.nn.moments(bottom, [0])
            bottom = tf.nn.batch_normalization(bottom, mean=batch_mean, variance=batch_var, offset=None,
                                               scale=None, variance_epsilon=epsilon)
            return bottom

    def cal_result(self, logits, k=3):
        softmax_result = tf.nn.softmax(logits)
        values, indices = tf.nn.top_k(softmax_result, k)
        return values, indices

    def build_model(self):

        self.conv1_1 = self.conv('conv1_1', self.x, 64, stride=[1, 1, 1, 1])
        self.conv1_2 = self.conv('conv1_2', self.conv1_1, 64, stride=[1, 1, 1, 1])
        self.pool1 = self.pool('pool1', self.conv1_2, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        self.conv2_1 = self.conv('conv2_1', self.pool1, 128, stride=[1, 1, 1, 1])
        self.conv2_2 = self.conv('conv2_2', self.conv2_1, 128, stride=[1, 1, 1, 1])
        self.pool2 = self.pool('pool2', self.conv2_2, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        self.conv3_1 = self.conv('conv3_1', self.pool2, 256, stride=[1, 1, 1, 1])
        self.conv3_2 = self.conv('conv3_2', self.conv3_1, 256, stride=[1, 1, 1, 1])
        self.conv3_3 = self.conv('conv3_3', self.conv3_2, 256, stride=[1, 1, 1, 1])
        self.pool3 = self.pool('pool3', self.conv3_3, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        self.conv4_1 = self.conv('conv4_1', self.pool3, 512, stride=[1, 1, 1, 1])
        self.conv4_2 = self.conv('conv4_2', self.conv4_1, 512, stride=[1, 1, 1, 1])
        self.conv4_3 = self.conv('conv4_3', self.conv4_2, 512, stride=[1, 1, 1, 1])
        self.pool4 = self.pool('pool4', self.conv4_3, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        self.conv5_1 = self.conv('conv5_1', self.pool4, 512, stride=[1, 1, 1, 1])
        self.conv5_2 = self.conv('conv5_2', self.conv5_1, 512, stride=[1, 1, 1, 1])
        self.conv5_3 = self.conv('conv5_3', self.conv5_2, 512, stride=[1, 1, 1, 1])
        self.pool5 = self.pool('pool5', self.conv5_3, kernel=[1, 2, 2, 1], stride=[1, 2, 2, 1], is_max_pool=True)

        self.fc6 = self.fc('fc6', self.pool5, out_nodes=4096)
        self.fc7 = self.fc('fc7', self.fc6, out_nodes=4096)
        self.logits = self.fc('fc8', self.fc7, out_nodes=self.config.n_classes)

        return self.logits
