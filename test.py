import os
import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize
from lib.data_loader.data_loader import CIFAR10BinDataLoader
from lib.utils.config import ConfigReader, TestNetConfig, DataConfig
from lib.vgg.vgg16test import VGG16_test


if __name__ == '__main__':

    config_reader = ConfigReader('experiments/configs/vgg16.yml')
    test_config = TestNetConfig(config_reader.get_test_config())
    data_config = DataConfig(config_reader.get_test_config())

    mode = 'RGB' if test_config.image_depth == 3 else 'L'
    img = imread('automobile10.png', mode=mode)
    img = imresize(img, [test_config.image_height, test_config.image_width]) # height, width
    k = 3
    with open('./data/datasets/cifar-10-batches-bin/batches.meta.txt') as mf:
        class_names = mf.read().splitlines()
        class_names = list(filter(None, class_names))

    net = VGG16_test(test_config)
    logits = net.build_model()
    values, indices = net.cal_result(logits)

    ckpt_path = test_config.model_path
    # start a session
    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    print(ckpt_path)
    try:
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except FileNotFoundError:
        raise 'Check your pretrained {:s}'.format(ckpt_path)

    [prob, ind, out] = sess.run([values, indices, logits], feed_dict={net.x: [img]})
    prob = prob[0]
    ind = ind[0]
    print('\nClassification Result:')
    for i in range(k):
        print('\tCategory Name: %s \n\tProbability: %.2f%%\n' % (class_names[ind[i]], prob[i] * 100))
    sess.close()
