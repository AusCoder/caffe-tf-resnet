# -*- coding: utf-8 -*-
import re

import tensorflow as tf

from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf


class InputParam(object):
    def __init__(self, input_param):
        assert(len(input_param.shape) == 1)
        dims = input_param.shape[0].dim
        assert(len(dims) == 4)
        self.n = dims[0]
        self.c = dims[1]
        self.h = dims[2]
        self.w = dims[3]


class ConvParam(object):
    def __init__(self, conv_param):
        assert(len(conv_param.kernel_size) == 1)
        assert(len(conv_param.pad) == 1)
        assert(len(conv_param.stride) == 1)
        self.c_o = conv_param.num_output
        self.k_s = conv_param.kernel_size[0]
        self.p_s = conv_param.pad[0]
        self.s_s = conv_param.stride[0]
        self.is_biased = conv_param.bias_term


class BNParam(object):
    def __init__(self, bn_param):
        self.use_global_stats = bn_param.use_global_stats


class ScaleParam(object):
    def __init__(self, scale_param):
        self.has_bias = scale_param.bias_term


def build_conv(input_tensor, name, conv_param, trainable=True):
    '''remember: NHWC order'''
    c_i = input_tensor.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = tf.get_variable('weights', 
            shape=[conv_param.k_s, conv_param.k_s, c_i, conv_param.c_o], 
            trainable=trainable)
        bias = tf.get_variable('bias', shape=[conv_param.c_o], trainable=trainable)
        
        paddings = tf.constant([[0, 0], [conv_param.p_s, conv_param.p_s], [conv_param.p_s, conv_param.p_s], [0, 0]])
        padded = tf.pad(input_tensor, paddings, mode='CONSTANT', constant_values=0)
        conv = tf.nn.conv2d(padded, kernel, [1, conv_param.s_s, conv_param.s_s, 1], padding='VALID')
        conv = tf.nn.bias_add(conv, bias)
    return conv


def build_bn(input_tensor, name, bn_param, trainable=True):
    '''
    see this for training: https://github.com/tensorflow/tensorflow/issues/10118
    '''
    c_i = input_tensor.get_shape()[-1]
    with tf.variable_scope(name):
        mean = tf.get_variable('mean', shape=[c_i], trainable=trainable)
        variance = tf.get_variable('variance', shape=[c_i], trainable=trainable)
        # TODO: how does caffe encode the scale and offset for batch norm?
        scale = None
        offset = None
        eps = 1e-5
        bn = tf.nn.batch_normalization(
            input_tensor,
            mean,
            variance,
            offset,
            scale,
            eps
        )
    return bn


def build_scale(input_tensor, name, scale_param, trainable=True):
    c_i = input_tensor.get_shape()[-1]
    with tf.variable_scope(name):
        scale = tf.get_variable('scale', shape=[c_i], trainable=trainable)
        scale = tf.multiply(input_tensor, scale)
        if scale_param.has_bias:
            bias = tf.get_variable('bias', shape=[c_i], trainable=trainable)
            scale = tf.nn.bias_add(scale, bias)
    return scale


def build_relu(input_tensor):
    return tf.nn.relu(input_tensor)


def parse_data_block(network, name):
    idx = 0
    while network.layer[idx].type != 'Input':
        idx += 1
    data_layer = network.layer[idx]
    assert(data_layer.name == name)
    assert(data_layer.input_param is not None)
    dims = data_layer.input_param.shape[0].dim
    assert(len(dims) == 4)
    _, c, h, w = dims
    return tf.placeholder(tf.float32, shape=(None, h, w, c))


def parse_conv_block(network, input_tensor, name):
    '''
    Read a convolutional block starting at name
    '''
    idx = 0
    while network.layer[idx].name != name:
        idx += 1
    conv_layer = network.layer[idx]
    bn_layer = network.layer[idx + 1]
    scale_layer = network.layer[idx + 2]
    assert(conv_layer.type == 'Convolution')
    assert(bn_layer.type == 'BatchNorm')
    assert(scale_layer.type == 'Scale')
    has_relu = network.layer[idx + 3].type == 'ReLU'
    # TODO: add assertions about bottoms and tops matching up
    conv_param = ConvParam(conv_layer.convolution_param)
    bn_param = BNParam(bn_layer.batch_norm_param)
    scale_param = ScaleParam(scale_layer.scale_param)
    conv = build_conv(input_tensor, conv_layer.name, conv_param)
    bn = build_bn(conv, bn_layer.name, bn_param)
    out = build_scale(bn, scale_layer.name, scale_param)
    if has_relu:
        out = build_relu(out)
    return out


def simple_assign(sess, caffeweights):
    for l, w in caffeweights:
        if l == 'conv1':
            with tf.variable_scope('conv1', reuse=True):
                weights = tf.get_variable('weights')
                sess.run(weights.assign(w[0]))
                bias = tf.get_variable('bias')
                sess.run(bias.assign(w[1]))
        if l == 'bn_conv1':
            with tf.variable_scope('bn_conv1', reuse=True):
                mean = tf.get_variable('mean')
                sess.run(mean.assign(w[0]))
                variance = tf.get_variable('variance')
                sess.run(variance.assign(w[1]))
        if l == 'scale_conv1':
            with tf.variable_scope('scale_conv1', reuse=True):
                scale = tf.get_variable('scale')
                sess.run(scale.assign(w[0]))
                bias = tf.get_variable('bias')
                sess.run(bias.assign(w[1]))
