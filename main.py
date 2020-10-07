#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
import matplotlib.pyplot as plt

import networkx as nx
import networkit as nk

from representations.explicit import Squashed
from sklearn.utils.extmath import randomized_svd

from gensim.models import KeyedVectors
from gensim.test.utils import datapath

import torch

from scipy.sparse import csr_matrix
from scipy.stats import bernoulli
from auction_lap import auction_lap

import codecs, sys, time, math, argparse, ot
import numpy as np
from grave_algorithm import *
from IPython import get_ipython


# ## Import PMI matrix and build SSPMI graph from it

# In[16]:


pmi_matrix_loc = "pmis/text8.w2.t200/pmi"
vocab_loc = pmi_matrix_loc + ".words.vocab"


# In[17]:


explicit = Squashed(pmi_matrix_loc, normalize=False, neg=5)
SSPMI = explicit.m.tocsr()


# In[18]:


vocab_size = SSPMI.shape[0]
emb_dim = 200


# In[19]:


print(vocab_size)


# In[20]:


SSPMI_B = bernoulli.rvs(SSPMI.todense())
SSPMI_B = np.maximum(SSPMI_B, SSPMI_B.T)


# In[21]:


SSPMI_G = nx.from_numpy_matrix(SSPMI_B)
SSPMI_G = nk.nxadapter.nx2nk(SSPMI_G)


# ## Obtain word embeddings from the SSPMI matrix and evaluate them

# In[22]:


U1, S1, V1T = randomized_svd(SSPMI, n_components=emb_dim)


# In[23]:


W1 = U1 @ np.diag(np.sqrt(S1))
get_ipython().system('rm embeddings/*')
np.savetxt('embeddings/w1.txt', W1, delimiter=' ')
get_ipython().system('echo $vocab_size $emb_dim > embeddings/emb1')
get_ipython().system('paste $vocab_loc embeddings/w1.txt -d " " >> embeddings/emb1')


# In[24]:


wv_from_text = KeyedVectors.load_word2vec_format(os.path.join("embeddings", "emb1"), binary=False)
ws353 = wv_from_text.evaluate_word_pairs(datapath('wordsim353.tsv'))
men = wv_from_text.evaluate_word_pairs('testsets/bruni_men.txt')
mturk = wv_from_text.evaluate_word_pairs('testsets/radinsky_mturk.txt')
rare = wv_from_text.evaluate_word_pairs('testsets/luong_rare.txt')
google = wv_from_text.evaluate_word_analogies(datapath('questions-words.txt'))
msr = wv_from_text.evaluate_word_analogies('testsets/msr.txt')
print('WS353 = %.3f' % ws353[0][0], end=', ')
print('MEN = %.3f' % men[0][0], end=', ')
print('M. Turk = %.3f' % mturk[0][0], end=', ')
print('Rare = %.3f' % rare[0][0], end=', ')
print('Google = %.3f' % google[0], end=', ')
print('MSR = %.3f' % msr[0])


# ## Fit generator to SSPMI graph and generate a RHG

# In[25]:


RHG = nk.generators.HyperbolicGenerator(SSPMI_G.numberOfNodes()).fit(SSPMI_G).generate()
RHG_M = nk.algebraic.adjacencyMatrix(RHG)


# ## Find Transformation R

# In[26]:


U2, S2, V2T = randomized_svd(RHG_M, n_components=emb_dim)
W2 = U2 @ np.diag(np.sqrt(S2))
x_src = W1.copy()
x_tgt = W2.copy()


# In[27]:


#normalizing and centralizing x_src
x_src -= x_src.mean(axis=0)[np.newaxis, :]
x_src /= np.linalg.norm(x_src, axis=1)[:, np.newaxis] + 1e-8


# In[28]:


#normalizing and centralizing x_tgt
x_tgt -= x_tgt.mean(axis=0)[np.newaxis, :]
x_tgt /= np.linalg.norm(x_tgt, axis=1)[:, np.newaxis] + 1e-8


# In[29]:


print("\nComputing initial mapping with convex relaxation...")
t0 = time.time()
R0 = convex_init(x_src[:1500], x_tgt[:1500], reg=args.reg, apply_sqrt=True)
print("Done [%03d sec]" % math.floor(time.time() - t0))


# In[30]:


print("\nComputing mapping with Wasserstein Procrustes...")
t0 = time.time()
R = align(x_src, x_tgt, R0.copy(), bsz=400, lr=args.lr, niter=1000,
          nepoch=4, reg=args.reg, nmax=vocab_size)
print("Done [%03d sec]" % math.floor(time.time() - t0))


# ## Find permutation P

# In[31]:


x_tgt_cpu = torch.from_numpy(x_tgt).float().cpu()
x_src_cpu = torch.from_numpy(x_src).float().cpu()
R_cpu = torch.from_numpy(R).float().cpu()
P_ = torch.mm(torch.mm(x_tgt_cpu, R_cpu.t()),x_src_cpu.t()).t()
del x_tgt_cpu, x_src_cpu, R_cpu


# In[32]:


print("\nComputing permutation with auction-lap algorithm...")
t0 = time.time()

_, y, _ = auction_lap(P_, eps=None)
P = csr_matrix((np.ones(vocab_size, dtype=np.int64), (np.arange(vocab_size), y.cpu().numpy())), 
               shape=(vocab_size, vocab_size))

print("Done [%03d sec]" % math.floor(time.time() - t0))


# ## Word embeddings evaluate

# In[33]:


np.savetxt('embeddings/w3.txt', P@W2, delimiter=' ')
get_ipython().system('echo $vocab_size $emb_dim > embeddings/emb3')
get_ipython().system('paste $vocab_loc embeddings/w3.txt -d " " >> embeddings/emb3')


# In[34]:


wv_from_text = KeyedVectors.load_word2vec_format(os.path.join("embeddings", "emb3"), binary=False)
ws353 = wv_from_text.evaluate_word_pairs(datapath('wordsim353.tsv'))
men = wv_from_text.evaluate_word_pairs('testsets/bruni_men.txt')
mturk = wv_from_text.evaluate_word_pairs('testsets/radinsky_mturk.txt')
rare = wv_from_text.evaluate_word_pairs('testsets/luong_rare.txt')
google = wv_from_text.evaluate_word_analogies(datapath('questions-words.txt'))
msr = wv_from_text.evaluate_word_analogies('testsets/msr.txt')
print('WS353 = %.3f' % ws353[0][0], end=', ')
print('MEN = %.3f' % men[0][0], end=', ')
print('M. Turk = %.3f' % mturk[0][0], end=', ')
print('Rare = %.3f' % rare[0][0], end=', ')
print('Google = %.3f' % google[0], end=', ')
print('MSR = %.3f' % msr[0])

