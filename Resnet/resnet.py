# @File     : resnet.py
# @Time     : 2018/5/31 14:01
# @Author   : DingTao
# @Contact  : jownthedarklegion@foxmail.com
# @license  : Copyright(c)
import utils
import tensorflow as tf


class Resnet:
    def __init__(self, depth_of_net, inputs, layer_info, num_of_class):
        self.DEPTH_OF_NET = depth_of_net
        self.INPUTS = inputs
        self.LAYER_INFO = layer_info
        self.NUM_OF_CLASS = num_of_class
        self.build_net()

    def build_net(self):
        with tf.variable_scope('resnet'+str(self.DEPTH_OF_NET)):
            conv1 = utils.convolution_layer(self.INPUTS, kernel_size=7, stride=2, kernel_nums=64, name='conv1')
            norm1 = utils.batch_normalization(conv1)
            relu1 = utils.nonlinear_ops(norm1)  # 112*112

            temp = relu1
            for i in range(4):  # 2:56*56 3:28*28 4:14*14 5:7*7
                name = 'conv'+str(i+2)
                with tf.variable_scope(name):
                    if i == 0:
                        kernel_size = 3
                        padding = 'SAME'
                    else:
                        kernel_size = 1
                        padding = 'VALID'
                    temp = utils.max_pool_layer(temp, kernel_size, 2, padding=padding, name='down_sample')
                    temp = utils.res_block(temp, self.LAYER_INFO[name]['nums'],
                                           self.LAYER_INFO[name]['length'])

            #avg_pool = utils.average_pool_layer(temp, 7, 1, name='avg_pool')
            fcinput = tf.reshape(temp, [-1, 2048])
            self.fc = utils.fully_connect_layer(fcinput, 2048, self.NUM_OF_CLASS, activation=None, name='fc')
            #self.fc = tf.clip_by_value(fc, 1e-10, 1.0)
