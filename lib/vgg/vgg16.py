import tensorflow as tf
from lib.networks.base_network import Net


class VGG16(Net):
    def __init__(self, cfg_):
        super().__init__(cfg_)

    def init_saver(self):
        pass

    def build_model(self):
        x = tf.placeholder(tf.float32, shape=[self.config.batch_size,
                                              self.config.image_width,
                                              self.config.image_height,
                                              self.config.image_depth])
        y = tf.placeholder(tf.int16, shape=[self.config.batch_size,
                                            self.config.n_classes])

    def conv(self):
        pass

    def pool(self):
        pass

    def fc(self):
        pass

    def bn(self):
        pass