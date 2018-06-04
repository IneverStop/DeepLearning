# @File     : utils.py
# @Time     : 2018/5/31 10:34
# @Author   : DingTao
# @Contact  : jownthedarklegion@foxmail.com
# @license  : Copyright(c)

import tensorflow as tf
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt


def convolution_layer(inputs, kernel_size, stride, kernel_nums, name=None):
    channel = int(inputs.get_shape()[-1])

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable("weights", shape=[kernel_size, kernel_size, channel, kernel_nums])
        biases = tf.get_variable("biases", shape=[kernel_nums])
        if stride == 1:
            conv = tf.nn.conv2d(inputs, weights, strides=[1, 1, 1, 1], padding='SAME')
        else:
            pad_total = kernel_size-1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
            conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='VALID')
        conv = tf.nn.bias_add(conv, biases)
        return conv


def batch_normalization(inputs, axis=3, training=True, name=None):
    return tf.layers.batch_normalization(inputs, axis=axis, training=training, name=name, reuse=tf.AUTO_REUSE)


def nonlinear_ops(inputs, ops=tf.nn.relu):
    return ops(inputs)


def max_pool_layer(inputs, kernel_size, stride, padding='VALID', name=None):
    return tf.nn.max_pool(inputs, ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride, stride, 1], padding=padding,
                          name=name)


def average_pool_layer(inputs, kernel_size, stride, padding='VALID', name=None):
    return tf.nn.avg_pool(inputs, [1, kernel_size, kernel_size, 1], [1, stride, stride, 1], padding, name=name)


def fully_connect_layer(inputs, input_dimension, output_dimension, activation=tf.nn.relu, name=None):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        weights = tf.get_variable("weights", shape=[input_dimension, output_dimension], dtype=tf.float32)
        biases = tf.get_variable("biases", shape=[output_dimension], dtype=tf.float32)
        fully_connect = tf.nn.xw_plus_b(inputs, weights, biases, name=scope.name)
        if activation:
            fully_connect = activation(fully_connect)
        return fully_connect


def dropout(inputs, keep_prob, name=None):
    return tf.nn.dropout(inputs, keep_prob=keep_prob, name=name)


def ops_block(inputs, kernel_nums, training, factor=True, name=None):
    """
    This function is used for calculate tensors through shortcut blocks.Shortcuts equals to
    inputs at first, then it will be convolutioned or not depend on value of factor(indicates
    this block is a identity block or convolution block).
    Input tensor will be convolutioned three times, after every convolution operation followed by
    a batch_normalization and relu.
    At the end, this function will output the sum of shortcut and third relu results.
    :param inputs: The value that fed by function res_block's inputs, it's
    :param kernel_nums: Num of each convolution kernel. It's a list like [64, 64, 256].
    :param training: It's a hyper_parameter for batch normalization.
    :param factor: To distinguish identity block and convolution block.
    :param name: Scope name such as 'identity_block1'.
    :return: Tensors through identity block or convolution block.
    """
    if factor:
        pre_name = 'identity_block_conv'
    else:
        pre_name = 'conv_block_conv'
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        shortcut = inputs
        conv1 = convolution_layer(inputs, 1, 1, kernel_nums[0], name=pre_name+'1')
        norm1 = batch_normalization(conv1, training, name='norm1')
        relu1 = nonlinear_ops(norm1)

        conv2 = convolution_layer(relu1, 3, 1, kernel_nums[1], name=pre_name+'2')
        norm2 = batch_normalization(conv2, training, name='norm2')
        relu2 = nonlinear_ops(norm2)

        conv3 = convolution_layer(relu2, 1, 1, kernel_nums[2], name=pre_name+'3')
        norm3 = batch_normalization(conv3, training, name='norm3')

        if factor is False:
            conv4 = convolution_layer(shortcut, 1, 1, kernel_nums[2], name=pre_name+'4')
            shortcut = batch_normalization(conv4, training, name='norm_shortcut')
        add = tf.add(norm3, shortcut)
        result = nonlinear_ops(add)
    return result


def res_block(inputs, kernel_nums, num_of_ops_block):
    """

    :param inputs:
    :param kernel_nums: Num of each convolution kernel. It's a list like [64, 64, 256].
    :param num_of_ops_block: As the name says.
    :return:
    """
    temp_result = inputs
    for i in range(num_of_ops_block):
        temp_result = ops_block(temp_result, kernel_nums, True, False if i == 0 else True, name='block'+str(i+1))
    return temp_result


def get_accuracy(logits, targets):
    batch_predictions = np.argmax(logits, axis=1)
    num_correct = np.sum(np.equal(batch_predictions, targets))
    return 100. * num_correct/batch_predictions.shape[0]


def get_batch_data(data, label, size):
    rand_index = np.random.choice(len(data), size=size)
    rand_data = np.array(data)[rand_index]
    rand_label = np.array(label)[rand_index]
    return rand_data, rand_label


def openfile(path):
    """
    This function is going to open the cifar-10 byte file on the path of 'path'.
    It will concat five train batch file in one list, and test batch in single list.
    Stay with each image array list is a label list.
    :return:  train/test image array list and label.
    """
    data_list = os.listdir(path)
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    for x in data_list:
        with open(path + x, 'rb') as fo:
            dicts = pickle.load(fo, encoding='bytes')
            img_list = convert(np.array(dicts[b'data']))
            lab_list = dicts[b'labels']
        if x != 'test_batch':
            for img in img_list:
                train_images.append(img)
            for label in lab_list:
                train_labels.append(label)
        else:
            for img in img_list:
                test_images.append(img)
            for label in lab_list:
                test_labels.append(label)
    return train_images, train_labels, test_images, test_labels


def convert(img_data):
    """
    Convert the img_data--size of 10000*32*32*3-- what is a one dimension list, to
    single handed image list, which has size of [1000, 32, 32, 3]
    :param img_data: 10000 image, size of 32*32*3, it a one dimension list.
    :return:
    """
    images = []
    img_data = img_data.reshape(10000, 3, 32, 32)
    for x in img_data:
        r = x[0].reshape(1024, 1)
        g = x[1].reshape(1024, 1)
        b = x[2].reshape(1024, 1)
        pic = np.hstack((r, g, b))
        pic = pic.reshape(32, 32, 3)
        images.append(pic)
    return images


def showimg(data, label):
    plt.title(label)
    plt.imshow(data)
    plt.show()


def save_params(param):
    write_file = open('params', 'wb')
    pickle.dump(param.get_value(borrow=True), write_file, -1)
    write_file.close()
