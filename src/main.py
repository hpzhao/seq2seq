#!/usr/bin/env python
#coding:utf8
import tensorflow as tf
import cPickle as pkl
import logging
from seq2seq import seq2seq
from helper import writer
from helper import Config

logging.basicConfig(level=logging.INFO,format = '%(asctime)s [%(levelname)s] %(message)s')

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('model','../model/model','')
flags.DEFINE_string('train_file','../data/train_dataset.pkl','')
flags.DEFINE_string('test_file', './test.txt','')
flags.DEFINE_string('output_file', './output.txt','')
flags.DEFINE_string('source_emb', '../data/source_embedding.pkl','')
flags.DEFINE_string('target_emb', '../data/target_embedding.pkl','')
flags.DEFINE_string('source_word2id', '../data/source_word2id.pkl','')
flags.DEFINE_string('source_id2word', '../data/source_id2word.pkl','')
flags.DEFINE_string('target_word2id', '../data/target_word2id.pkl', '')
flags.DEFINE_string('target_id2word', '../data/target_id2word.pkl', '')

flags.DEFINE_integer('is_training',1,'is training or testing. default 1 is trainig')

def gen_config(target_max_length):

    source_emb = pkl.load(open(FLAGS.source_emb))
    target_emb = pkl.load(open(FLAGS.target_emb))
    source_word2id = pkl.load(open(FLAGS.source_word2id))
    source_id2word = pkl.load(open(FLAGS.source_id2word))
    target_word2id = pkl.load(open(FLAGS.target_word2id))
    target_id2word = pkl.load(open(FLAGS.target_id2word))

    return Config(
            source_emb = source_emb,
            target_emb = target_emb,
            source_word2id = source_word2id,
            source_id2word = source_id2word,
            target_word2id = target_word2id,
            target_id2word = target_id2word,
            source_vocab_size = source_emb.shape[0],
            target_vocab_size = target_emb.shape[0],
            word_dim = 300,
            source_hidden_units = 512,
            target_hidden_units = 512,
            target_max_length = target_max_length,
            batch_size = 128,
            learning_rate = 1e-3,
            epochs = 10,
            sample_size = 512)

def train(config):
    logging.info('start training...') 
    train_dataset = pkl.load(open(FLAGS.train_file))
    train_dataset.shuffle()

    with tf.variable_scope('model'):
        model = seq2seq(config)
   
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    
    with tf.Session(config = tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        model.train(sess, train_dataset)
    logging.info('end training')

def test(config):
    logging.info('start testing...')
    
    source_word2id = config.source_word2id
    target_id2word = config.target_id2word

    test_data = []
    
    for line in open(FLAGS.test_file):
        tokens = line.strip().split()
        sents = [source_word2id[word] if word in source_word2id else 1 for word in tokens]
        test_data.append(sents)

    with tf.variable_scope('model'):
        model = seq2seq(config)
        
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    
    with tf.Session(config = tf_config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.model)
        outputs = model.inference(sess,test_data)
    
    with open(FLAGS.output_file, 'w') as f:
        for tokens in outputs:
            sents = [target_id2word[id] for id in tokens]
            f.writelines(' '.join(sents))
            f.write('\n')

    logging.info('output_file:' + FLAGS.output_file)
    logging.info('end test')

def main(_):

    if FLAGS.is_training == 1:
        config = gen_config(10)
        train(config)
    else:
        config = gen_config(1)
        test(config)

if __name__ == '__main__':
    tf.app.run()
