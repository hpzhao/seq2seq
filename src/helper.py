#!/usr/bin/env python
#coding:utf8
import numpy as np
import cPickle as pkl 
import json
from collections import namedtuple

Config = namedtuple('Config', ['source_emb', 'target_emb',  'source_word2id', 'source_id2word', 'target_word2id', 
                    'target_id2word', 'source_vocab_size', 'target_vocab_size', 'word_dim', 'source_hidden_units', 
                    'target_hidden_units', 'target_max_length', 'batch_size', 'learning_rate', 'epochs', 'sample_size'])

def padding(data, target_max_seg):
    
    source_sent = [list(example[0]) for example in data]
    target_sent = [list(example[1]) for example in data]

    source_max_seg = max([len(example) for example in source_sent])

    batch_size = len(source_sent)
    
    source = np.zeros([batch_size,source_max_seg], np.int32)
    target_input = np.zeros([batch_size, target_max_seg], np.int32)
    target_gold = np.zeros([batch_size, target_max_seg], np.int32)
    
    for i in range(batch_size):
        source_len = len(source_sent[i])
        target_len = len(target_sent[i])
        
        source_real = [source_sent[i][j] for j in range(source_len)]
        source_pad = [0 for _ in range(source_max_seg - source_len)]
        source[i] = source_real + source_pad
    
        if target_len >= target_max_seg - 1:
            target_input[i] = [2] + target_sent[i][:target_max_seg - 1]
            target_gold[i] = target_sent[i][:target_max_seg - 1] + [3]
        else:
            target_input[i] = [2] + target_sent[i][:] + [0 for _ in range(target_max_seg - target_len - 1)]
            target_gold[i] = target_sent[i][:] + [3] + [0 for _ in range(target_max_seg - target_len - 1)]

    return source, target_input, target_gold

def writer(test_data, predict, output, id2word):
    data = json.load(open(test_data))
    with open(output,'w') as f:
        for i in range(10):
            post = data[i][0][0]
            gold = data[i][1][0]
            predicit_ = ''
            for j in range(16):
                #if predict[i][j] not in [0,1,2,3]:
                predicit_ += id2word[predict[i][j]] + ' '
            f.write(('Post: ' + post).encode('utf8'))
            f.write(('Gold: ' + gold).encode('utf8'))
            f.write(('Predict: ' + predicit_ + '\n').encode('utf8'))
            f.write('\n')
    


if __name__ == '__main__':
    data = [[([1,2,3,4],2),([1,8,3],3)],
            [([1,2],2),([5],1)]]
    
    source, source_emotion, target_input, target_gold, target_emotion = padding(data, 3) 
    print source
    print source_emotion 
    print target_input
    print target_gold
    print target_emotion
