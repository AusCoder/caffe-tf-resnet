# -*- coding: utf-8 -*-
import logging
import os

import numpy as np
import caffe
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf


def load_caffemodel(def_path, data_path):
    net = caffe.Net(str(def_path), str(data_path), caffe.TEST)
    out = [(k, [switch_order(blob.data) for blob in v]) for k, v in net.params.items()]
    return out


def load_network_definition(def_path):
    with open(def_path, 'r') as infile:
        s = infile.read()
        network = caffe_pb2.NetParameter()
        return txtf.Merge(s, network)


def switch_order(arr):
    '''Changes NCHW to NHWC order'''
    if len(arr.shape) != 4:
        return arr
    return np.transpose(arr, (2, 3, 1, 0))


def load_net(def_path, data_path):
    batch_size = 1
    net = caffe.Net(def_path, data_path, caffe.TEST)
    input_width = net.blobs['data'].width
    input_height = net.blobs['data'].height
    num_channels = net.blobs['data'].channels
    net.blobs['data'].reshape(batch_size, num_channels, input_width, input_height)
    return net