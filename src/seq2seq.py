#!/usr/bin/env python
#coding:utf8

from helper import padding

import tensorflow as tf
import numpy as np
import logging
logging.basicConfig(level=logging.INFO,format = '%(asctime)s [%(levelname)s] %(message)s')

class seq2seq():
    def __init__(self, config):

        self.config = config

        self.source_vocab_size = config.source_vocab_size
        self.target_vocab_size = config.target_vocab_size

        self.word_dim = config.word_dim
        
        self.source_hidden_units = config.source_hidden_units
        self.target_hidden_units = config.target_hidden_units
         
        self.target_max_length = config.target_max_length
        
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate
        self.epochs = config.epochs
        self.sample_size = config.sample_size
        
        # set random seed
        tf.set_random_seed(1)
        
        # define placeholder
        with tf.variable_scope('placeholder'):
            
            self.source_input = tf.placeholder(tf.int32, [None, None])    
            self.target_input = tf.placeholder(tf.int32, [None, None])
            self.target_gold = tf.placeholder(tf.int32, [None, None])

        with tf.device('/cpu:0'):
            with tf.variable_scope('input_layer'):
                self.source_embedding = tf.get_variable(
                        name = 'source_embedding',
                        dtype = tf.float32,
                        initializer = config.source_emb)
                self.target_embedding = tf.get_variable(
                        name = 'target_embedding',
                        dtype = tf.float32,
                        initializer = config.target_emb)

                encode_input = tf.nn.embedding_lookup(self.source_embedding, self.source_input)
                decode_input = tf.nn.embedding_lookup(self.target_embedding, self.target_input)
            
        source_seq_len = tf.reduce_sum(tf.sign(self.source_input), 1)
        target_seq_len = tf.reduce_sum(tf.sign(self.target_input), 1)
        
        with tf.variable_scope('encoder'):
            encoder_cell = tf.nn.rnn_cell.GRUCell(self.source_hidden_units)
            encoder_cells = tf.nn.rnn_cell.MultiRNNCell([encoder_cell] * 2)
            _ , encoder_state = tf.nn.dynamic_rnn(
                    cell = encoder_cells,
                    inputs = encode_input,
                    sequence_length = source_seq_len,
                    dtype = tf.float32)
                
        self.initial_state = encoder_state
        state = self.initial_state
        output = []
        with tf.variable_scope('decoding'):
            decoder_cell = tf.nn.rnn_cell.GRUCell(self.source_hidden_units)
            decoder_cells = tf.nn.rnn_cell.MultiRNNCell([decoder_cell] * 2)
             
            for step in range(self.target_max_length):
                if step > 0: tf.get_variable_scope().reuse_variables()
                cell_output, state = decoder_cells(
                    inputs = decode_input[:, step, :],
                    state = state)
                output.append(cell_output)
 
        self.final_state = state
        # outputs: [batch_size * source_seq_len, source_hidden_units]
        self.softmax_w = tf.get_variable(
                name = 'weights', 
                shape = [self.target_vocab_size, self.target_hidden_units], 
                dtype = tf.float32)
        self.softmax_b = tf.get_variable(
                name = 'bias',
                shape = [self.target_vocab_size],
                dtype = tf.float32)
    
        # 变换矩阵一定要检查是不是自己想要的结果！！！！！！！
        self.outputs = tf.reshape(tf.concat(1, output), [-1, self.source_hidden_units])
    
        logits = tf.matmul(self.outputs, tf.transpose(self.softmax_w)) + self.softmax_b
        self.next_id = tf.argmax(logits, axis = 1)
        
        self.labels = tf.reshape(self.target_gold, [-1, 1])
        with tf.variable_scope('sampled_softmax_loss'):
            loss = tf.nn.sampled_softmax_loss(
                    weights = self.softmax_w,
                    biases = self.softmax_b,
                    inputs = self.outputs,
                    labels = self.labels,
                    num_sampled = self.sample_size,
                    num_classes = self.target_vocab_size)
                    
            target_mask = tf.sequence_mask(target_seq_len, self.target_max_length, tf.float32)
            target_mask = tf.reshape(target_mask, [-1])
            self.loss = tf.reduce_mean(loss * target_mask)
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            
    def step(self, sess, source):
        outputs = []
        target_input = np.reshape(np.array([2], np.int32), [1, -1])
        feed_dict = {self.source_input:source, self.target_input:target_input} 
        state, predict_id = sess.run([self.final_state, self.next_id], feed_dict = feed_dict)
        
        outputs.append(predict_id[0])
        
        for i in range(9):
            target_input = np.reshape(np.array(predict_id, np.int32), [1, -1])
            feed_dict = {self.target_input: target_input, self.initial_state : state} 
            state, predict_id = sess.run([self.final_state, self.next_id], feed_dict = feed_dict)
            outputs.append(predict_id[0])
        
        return outputs
    
    def inference(self, sess, test_data):

        outputs = []
        for data in test_data:
            print data
            data = np.reshape(np.array(data, np.int32), (1, -1))
            output = self.step(sess, data)
            outputs.append(output)
        return outputs
    def train(self, sess, train_data):
        
        saver = tf.train.Saver()
        epoch_size = train_data.size() / self.batch_size
        
        
        for step in range(epoch_size * self.epochs):
            
            train_batch = train_data.next_batch(self.batch_size)
            source, target_input, target_gold = padding(train_batch, self.target_max_length)
            feed_dict = {self.source_input:source, self.target_input:target_input, self.target_gold:target_gold}
            loss,  _ = sess.run([self.loss, self.train_op], feed_dict = feed_dict)
            
            if step % 100 == 0:
                saver.save(sess, '../model/model')
                logging.info('epochs ' + str(train_data.epochs()) + ' : ' + str(loss))

        saver.save(sess, '../model/model')
