
SEQ_LEN = 200
STRIDE_MU = 20 #expected value of stride
STRIDE_SIG = 2 #standard standard deviation of stride

import sys
from os import listdir
from random import choice
from midi_preproc import midiToMatrix
import numpy as np
import scipy
from prefetch_generator import background

bad_files = []

#all files in storage
files = [each for each in listdir('midi') if each.endswith('.mid') and (not ('midi/'+each in bad_files))]

def split_random_file():#returns sparse matrix every SEQ_LEN rows of wich are cut from the vectorized midi file
    fi = choice(files)#select random midi file
    
    cut = []
    
    sparsed = midiToMatrix('midi/'+fi)
    pos = 0
    
    #cut the sequence
    while(sparsed.shape[0] - pos > SEQ_LEN):
        
        cut.append(sparsed[pos: pos+SEQ_LEN])
        
        pos+= np.clip(np.round(np.random.normal(STRIDE_MU, STRIDE_SIG)), 1, SEQ_LEN//2)
    if len(cut) > 0:
        return scipy.sparse.vstack(cut,format='csr')
    else:
        return scipy.sparse.csc_matrix(np.zeros(shape=(0,129)))
    
def generate_minibatch(batch_size=32):
    res = []
    while(sum([x.shape[0] for x in res])/SEQ_LEN <= batch_size):
        res.append(split_random_file())
    return scipy.sparse.vstack(res,format='csr')

@background(max_prefetch=10)
def iterate_minibatches(num_batches, batch_size=32):
    for i in xrange(num_batches):
        yield generate_minibatch(batch_size)