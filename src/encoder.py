#!/usr/bin/env python
#coding:utf8
import tensorflow as tf
from collections import namedtuple
import numpy as np

class Encoder():
    def __init__(self, source_hidden_units = 512):
        self.source_hidden_units = source_hidden_units

        with tf.variable_scope('Encoder_GRU_Cell'):
            cell_fw = tf.nn.rnn_cell.GRUCell(self.source_hidden_units)
            self.cell = tf.nn.rnn_cell.MultiRNNCell([cell_fw] * 2)

    def encode(self, inputs, sequence_length):
        with tf.variable_scope('GRU-BRNN'): 
            outputs, states = tf.nn.dynamic_rnn(
                    cell = self.cell,
                    inputs = inputs,
                    sequence_length = sequence_length,
                    dtype = tf.float32)
             
        return  outputs, states
