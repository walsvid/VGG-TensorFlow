import os
import numpy as np
import tensorflow as tf
import input_data
import tools
import VGG


IMG_W = 32
IMG_H = 32
N_CLASSES = 10
BATCH_SIZE = 32
learning_rate = 0.01
MAX_STEP = 15000
IS_PRETRAIN = True


def train():
    pre_trained_weights = './vgg16.npy'
    data_dir = './data/cifar-10-batches-bin/'
    train_log_dir = './logs/train/'
    val_log_dir = './logs/val/'

    with tf.name_scope('input'):
        train_image_batch, train_label_batch = input_data.read_cifar10(data_dir=data_dir,
                                                                       is_train=True,
                                                                       batch_size=BATCH_SIZE,
                                                                       shuffle=True)

        val_image_batch, val_label_batch = input_data.read_cifar10(data_dir=data_dir,
                                                                   is_train=False,
                                                                   batch_size=BATCH_SIZE,
                                                                   shuffle=False)

    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int16, shape=[BATCH_SIZE, N_CLASSES])

    logits = VGG.VGG16(x, N_CLASSES, IS_PRETRAIN)
    loss, loss_summary = tools.loss(logits, y_)
    accuracy, acc_summary = tools.accuracy(logits, y_)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = tools.optimize(loss, learning_rate, global_step)

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge([loss_summary, acc_summary])

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    tools.load_with_skip(pre_trained_weights, sess, ['fc6', 'fc7', 'fc8'])

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)

    try:
        for step in np.arange(MAX_STEP):
            # print('====ALL====: %d' % step)
            if coord.should_stop():
                break

            train_image, train_label = sess.run([train_image_batch, train_label_batch])
            _, train_loss, train_acc = sess.run([train_op, loss, accuracy], feed_dict={x: train_image, y_: train_label})

            if step % 50 == 0 or step + 1 == MAX_STEP:
                print('===TRAIN===: Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, train_loss, train_acc))
                summary_str = sess.run(summary_op, feed_dict={x: train_image, y_: train_label})
                train_summary_writer.add_summary(summary_str, step)
            if step % 200 == 0 or step + 1 == MAX_STEP:
                val_image, val_label = sess.run([val_image_batch, val_label_batch])
                val_loss, val_acc = sess.run([loss, accuracy], feed_dict={x: val_image, y_: val_label})
                print('====VAL====: Step %d, val loss = %.4f, val accuracy = %.4f%%' % (step, val_loss, val_acc))
                summary_str = sess.run(summary_op, feed_dict={x: train_image, y_: train_label})
                val_summary_writer.add_summary(summary_str, step)
            if step % 2000 == 0 or step + 1 == MAX_STEP:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('===INFO====: Training completed, reaching the maximum number of steps')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


def main():
    train()


if __name__ == '__main__':
    main()
