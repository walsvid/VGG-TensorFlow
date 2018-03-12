import os
import tensorflow as tf
import numpy as np
from lib.utils.config import DataConfig


class DataLoader(object):
    def __init__(self, config, is_train, is_shuffle):
        """
        :param config: input config
        :type config: DataConfig
        :param is_train: is in train phase
        :type is_train: bool
        :param is_shuffle: shuffle data
        :type is_shuffle: bool
        """
        self.config = config
        self.is_train = is_train
        self.is_shuffle = is_shuffle

    def load_data(self):
        pass

    def generate_batch(self):
        pass


class CIFAR10BinDataLoader(DataLoader):
    def __init__(self, config, is_train, is_shuffle):
        """
        :param config: input config
        :type config: DataConfig
        :param is_train: is in train phase
        :type is_train: bool
        :param is_shuffle: shuffle data
        :type is_shuffle: bool
        """
        super().__init__(config, is_train, is_shuffle)
        self.image_width = self.config.image_width
        self.image_height = self.config.image_height
        self.image_depth = self.config.image_depth

        self.label_bytes = self.config.label_bytes
        self.image_bytes = self.image_width * self.image_height * self.image_depth

        self.data_dir = self.config.data_dir
        self.batch_size = self.config.batch_size
        self.n_classes = self.config.n_classes

        self.filename_queue = self.load_data()

    def load_data(self):
        if self.is_train:
            file_names = [os.path.join(self.data_dir, 'data_batch_%d.bin' % ii) for ii in np.arange(1, 6)]
        else:
            file_names = [os.path.join(self.data_dir, 'test_batch.bin')]

        return tf.train.string_input_producer(file_names)

    def generate_batch(self):
        reader = tf.FixedLengthRecordReader(self.label_bytes + self.image_bytes)
        key, value = reader.read(self.filename_queue)

        record_bytes = tf.decode_raw(value, tf.uint8)

        label = tf.slice(record_bytes, [0], [self.label_bytes])
        label = tf.cast(label, tf.int32)

        image_raw = tf.slice(record_bytes, [self.label_bytes], [self.image_bytes])
        image_raw = tf.reshape(image_raw, [self.image_depth, self.image_height, self.image_width])
        image = tf.transpose(image_raw, (1, 2, 0))  # convert from D/H/W to H/W/D
        image = tf.cast(image, tf.float32)

        image = tf.image.per_image_standardization(image)

        if self.is_shuffle:
            image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                              batch_size=self.batch_size,
                                                              num_threads=64,
                                                              capacity=20000,
                                                              min_after_dequeue=3000)
        else:
            image_batch, label_batch = tf.train.batch([image, label],
                                                      batch_size=self.batch_size,
                                                      num_threads=64,
                                                      capacity=2000)

        label_batch = tf.one_hot(label_batch, depth=self.n_classes)
        label_batch = tf.cast(label_batch, dtype=tf.int32)
        label_batch = tf.reshape(label_batch, [self.batch_size, self.n_classes])

        return image_batch, label_batch


class CIFAR10ImageDataLoader(DataLoader):
    def __init__(self, config, is_train, is_shuffle):
        """
        :param config: input config
        :type config: DataConfig
        :param is_train: is in train phase
        :type is_train: bool
        :param is_shuffle: shuffle data
        :type is_shuffle: bool
        """
        super().__init__(config, is_train, is_shuffle)
        self.image_width = self.config.image_width
        self.image_height = self.config.image_height
        self.image_depth = self.config.image_depth

        self.data_dir = self.config.data_dir
        self.batch_size = self.config.batch_size
        self.n_classes = self.config.n_classes

        self.filename_queue = self.load_data()

    def load_data(self):
        return 0
