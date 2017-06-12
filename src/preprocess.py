#!/usr/bin/env python
#coding:utf8
from dataset import Dataset

import random
import cPickle as pkl
import numpy as np

def build_word_maps(embedding_file):
    word2id, id2word = {}, {}
    
    id2word[0] = '_PAD'
    id2word[1] = '_UNK'
    id2word[2] = '<s>'
    id2word[3] = '</s>'

    word2id['_PAD'] = 0
    word2id['_UNK'] = 1
    word2id['<s>'] = 2
    word2id['</s>'] = 3
    
    zeros = [0 for _ in range(300)]
    embedding_list = [zeros for _ in range(3)]

    for line in open(embedding_file):
        tokens = line.strip().split()
        if len(tokens) != 2:
            word2id[tokens[0]] = len(embedding_list)
            id2word[len(embedding_list)] = tokens[0]
            embedding_list.append(tokens[1:])

    embedding_array = np.array(embedding_list, np.float32)

    return word2id, id2word, embedding_array

def gen_dataset(source_file, target_file):
    source_word2id = pkl.load(open('../data/source_word2id.pkl'))
    target_word2id = pkl.load(open('../data/target_word2id.pkl'))
    
    source_list = []
    for line in open(source_file):
        tokens = line.strip().split()
        sents = tuple(source_word2id[word] if word in source_word2id else 1 for word in tokens)
        source_list.append(sents)

    target_list = []
    for line in open(target_file):
        tokens = line.strip().split()
        sents = tuple(target_word2id[word] if word in target_word2id else 1 for word in tokens)
        target_list.append(sents)

    data_list = zip(source_list, target_list)

    dataset = Dataset(data_list)
    pkl.dump(dataset, open('../data/train_dataset.pkl', 'w'))

    return Dataset(data_list)

if __name__ == '__main__':
    source_word2id, source_id2word, source_embedding = build_word_maps('../data/question.embed')
    target_word2id, target_id2word, target_embedding = build_word_maps('../data/answer.embed')
    
    pkl.dump(source_word2id, open('../data/source_word2id.pkl', 'w'))
    pkl.dump(source_id2word, open('../data/source_id2word.pkl', 'w'))
    pkl.dump(source_embedding, open('../data/source_embedding.pkl', 'w'))
    pkl.dump(target_word2id, open('../data/target_word2id.pkl', 'w'))
    pkl.dump(target_id2word, open('../data/target_id2word.pkl', 'w'))
    pkl.dump(target_embedding, open('../data/target_embedding.pkl', 'w'))

    gen_dataset('../data/question.raw', '../data/answer.raw')
