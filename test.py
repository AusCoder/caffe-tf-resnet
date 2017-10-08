# -*- coding: utf-8 -*-
import os

import numpy as np
import tensorflow as tf
import caffe
from hypothesis import given
from hypothesis.strategies import integers, text, nothing
from hypothesis.extra.numpy import arrays as gen_arrays

from caffe_ import load_caffemodel, load_network_definition, load_net
from resnet50 import parse_data_block, parse_conv_block, simple_assign


class ConvertTest(object):
    def __init__(self, def_path, data_path):
        self.caffe_net = load_net(def_path, data_path)

        self.tf_sess = tf.Session()
        network = load_network_definition(def_path)
        self.tf_data = parse_data_block(network, 'data')
        self.tf_conv1 = parse_conv_block(network, self.tf_data, 'conv1')
        caffeweights = load_caffemodel(def_path, data_path)
        simple_assign(self.tf_sess, caffeweights)

    def close(self):
        self.tf_sess.close()
    
    def execute_example(self, f):
        return f()

    @given(s = text())
    def test_equality(self, s):
        assert(s == s)

    @given(arr = gen_arrays(
        np.uint8,
        (3, 256, 256),
        elements=integers(min_value=0, max_value=255)
        ))
    def test_conv1(self, arr):
        caffe_imgs = np.array([arr])
        tf_imgs = np.array([np.transpose(arr, (1, 2, 0))])

        self.caffe_net.blobs['data'].data[...] = caffe_imgs
        caffe_out = self.caffe_net.forward(start='data', end='conv1_relu')['conv1']
        caffe_out = np.transpose(caffe_out, (0, 2, 3, 1))

        tf_out = self.tf_sess.run(self.tf_conv1, feed_dict={self.tf_data: tf_imgs})

        assert(np.allclose(tf_out, caffe_out, rtol=0.0, atol=0.0005))


if __name__ == '__main__':
    def_path = './data/ResNet-50-deploy.prototxt'
    data_path = './data/ResNet-50-model.caffemodel'

    convert_test = ConvertTest(def_path, data_path)
    convert_test.test_conv1()
    convert_test.close()