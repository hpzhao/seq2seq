#!/usr/bin/env python
#coding:utf8
import numpy
numpy.random.seed(1234)

class Dataset():
    def __init__(self, data_list):
        self._data_list = data_list
        self._data = numpy.array(data_list)
        self._numbers = self._data.shape[0]
        self._index_in_epoch = 0
        self._epochs = 0
    
    def size(self):
        return self._numbers
   
    def shuffle(self):
        perm = numpy.arange(self._numbers)
        numpy.random.shuffle(perm)
        self._data = self._data[perm]
        self._index_in_epoch = 0
        
    def epochs(self):
        return self._epochs
    
    def next_batch(self,batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._numbers:
            #finished epoch
            self._epochs += 1
            #shuffle data
            perm = numpy.arange(self._numbers)
            numpy.random.shuffle(perm)
            self._data = self._data[perm]
            #start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._numbers
        end = self._index_in_epoch
        return self._data[start:end]

if __name__ == '__main__':
    data = [[([1,2,3,4],1),([1,2],2)],[([1,2,4],0),([2,2],5)]]
    test = Dataset(data)
    print test.size()
    print test.next_batch(1)[0][1][1]
    print test.next_batch(1)[0][0][1]
    print test.next_batch(1)[0][0][1]
