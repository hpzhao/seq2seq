#!/usr/bin/env python
#coding:utf8
import tensorflow as tf

class Decoder():
    def __init__(self, target_vocab_size, layers = 2,  attention_units = 128, target_hidden_units = 512):
        
        self.attention_units = attention_units
        self.target_vocab_size = target_vocab_size
        self.target_hidden_units = target_hidden_units
        self.layers = layers
        
        with tf.variable_scope('GRU_Cell'):        
            self.cell_fw = tf.nn.rnn_cell.GRUCell(self.target_hidden_units)
            self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell_fw] * self.layers)

    def initial_state(self, batch_size, initial_value):
        return self.cell.zero_state(batch_size, tf.float32)

    '''
    global attention function:

    Args:
        encoder_outputs: The outputs of encoder. 
            A Tensor of shape '[batch_size,time_step,outputs_dim]'
        encoder_sequence_length: The valid length of each sequence.
            A Tensor of shape '[batch_size]'
        decoder_output: The current state of the decoder.
            A Tensor of shape '[batch_size,decoder_output_dim]'

    Returns:
        return the final attention output
        A tensor of shape '[batch_size,outputs_dim]'

    '''
    def global_attention(self, encoder_outputs, encoder_sequence_length, decoder_output):
        
        # transform both keys and query into a tensor with 'attention_units' units
        # att_keys :  [batch_size,time_step,attention_units]
        # att_query : [batch_size,attention_units]
        with tf.variable_scope('att_keys_layer'):
            att_keys = tf.contrib.layers.fully_connected(
                    inputs = encoder_outputs,
                    num_outputs = self.attention_units,
                    activation_fn = None)
        with tf.variable_scope('att_query_layer'): 
            att_query = tf.contrib.layers.fully_connected(
                    inputs = decoder_output,
                    num_outputs = self.attention_units,
                    activation_fn = None)
        
        # use dot score
        # scores : [batch_size,time_step]
        scores = tf.reduce_sum(att_keys * tf.expand_dims(att_query,1),[2])

        # replace all scores for padded inputs with tf.float32.min
        scores_mask = tf.sequence_mask(
                lengths = tf.to_int32(encoder_sequence_length),
                maxlen = tf.to_int32(tf.shape(scores)[1]),
                dtype = tf.float32)
        
        scores = scores * scores_mask + ((1.0 - scores_mask) * tf.float32.min)
        
        # Normalize the scores as attention weights
        scores_normalized = tf.nn.softmax(scores)

        # Calculate the weighted average  of the attention inputs
        attention_context = tf.expand_dims(scores_normalized,2) * encoder_outputs
        attention_context = tf.reduce_sum(attention_context,1)
        
        return attention_context
    
    def compute_output(self,encoder_outputs,encoder_sequence_length,decoder_output):
        '''Computes the decoder outputs.'''

        # Compute attention
        with tf.variable_scope('compute_variable'):
            attention_context = self.global_attention(
                    encoder_outputs = encoder_outputs,
                    encoder_sequence_length = encoder_sequence_length,
                    decoder_output = decoder_output)

        # synthesize information between decoder state and attention context
        # see https://arxiv.org/abs/1508.04025v5
        # output shape: [batch_size,decoder_output_dim]
        with tf.variable_scope('att_context_layer'): 
            output = tf.contrib.layers.fully_connected(
                    inputs = tf.concat(1,[decoder_output,attention_context]),
                    num_outputs = self.target_hidden_units,
                    activation_fn = tf.nn.relu)
        
        return output

    def step(self,inputs,state,encoder_outputs,encoder_sequence_length):
        cell_output, cell_state = self.cell(inputs,state)
        #output = self.compute_output(encoder_outputs, encoder_sequence_length, cell_output)
        return cell_output, cell_state

