
SEQ_LEN = 200
STRIDE_MU = 20 #expected value of stride
STRIDE_SIG = 2 #standard standard deviation of stride

import sys
import os
from os import listdir
from random import choice
from midi_preproc import midiToMatrix
import numpy as np
import scipy
from prefetch_generator import background
import pickle as pkl

import tqdm as tqdm
def parallelization(fun,massive,tq = True):    
    num_cores = 20#multiprocessing.cpu_count()
    if tq:
        results = np.array(Parallel(n_jobs=num_cores)(delayed(fun)(i) for i in tqdm(massive)))
        return results
    else:
        results = Parallel(n_jobs=num_cores)(delayed(fun)(i) for i in massive)
        return results

bad_files = []

if os.path.isfile('bad_files.pkl'):
    with open('bad_files.pkl','rb') as f:
        bad_files = pkl.load(f)

#all files in storage
files = [each for each in listdir('midi') if each.endswith('.mid') and (not ('midi/'+each in bad_files))]
files_precomp = [each[:-5] for each in listdir('midi') if each.endswith('.prep')]


def split_random_file(precomputed=False):#returns sparse matrix every SEQ_LEN rows of wich are cut from the vectorized midi file
    fi = choice(files_precomp) if precomputed else choice(files)#select random midi file
    
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
    
def generate_minibatch(batch_size=32, precomputed=True):
    res = []
    while(sum([x.shape[0] for x in res])/SEQ_LEN <= batch_size):
        res.append(split_random_file(precomputed))
    res = scipy.sparse.vstack(res,format='csr')
    
    return res[:SEQ_LEN * batch_size]

#@background(max_prefetch=10)
def iterate_minibatches(num_batches, batch_size=32, precomputed=True):
    for i in xrange(num_batches):
        yield generate_minibatch(batch_size,precomputed)