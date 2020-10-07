import codecs, sys, time, math, argparse, ot
import numpy as np
from utilsss import *

parser = argparse.ArgumentParser(description='Wasserstein Procrustes for Embedding Alignment')
parser.add_argument('--model_src', type=str, help='Path to source word embeddings')
parser.add_argument('--model_tgt', type=str, help='Path to target word embeddings')
parser.add_argument('--lexicon', type=str, help='Path to the evaluation lexicon')
parser.add_argument('--output_src', default='', type=str, help='Path to save the aligned source embeddings')
parser.add_argument('--output_tgt', default='', type=str, help='Path to save the aligned target embeddings')
parser.add_argument('--seed', default=1111, type=int, help='Random number generator seed')
parser.add_argument('--nepoch', default=5, type=int, help='Number of epochs')
parser.add_argument('--niter', default=5000, type=int, help='Initial number of iterations')
parser.add_argument('--bsz', default=500, type=int, help='Initial batch size')
parser.add_argument('--lr', default=500., type=float, help='Learning rate')
parser.add_argument('--nmax', default=20000, type=int, help='Vocabulary size for learning the alignment')
parser.add_argument('--reg', default=0.05, type=float, help='Regularization parameter for sinkhorn')
args, unknown = parser.parse_known_args()

def objective(X, Y, R, n=5000):
    Xn, Yn = X[:n], Y[:n]
    C = -np.dot(np.dot(Xn, R), Yn.T)
    P = ot.sinkhorn(np.ones(n), np.ones(n), C, 0.025, stopThr=1e-3)
    return 1000 * np.linalg.norm(np.dot(Xn, R) - np.dot(P, Yn)) / n

def sqrt_eig(x):
    U, s, VT = np.linalg.svd(x, full_matrices=False)
    return np.dot(U, np.dot(np.diag(np.sqrt(s)), VT))

def align(X, Y, R, lr=10., bsz=200, nepoch=5, niter=1000,
          nmax=10000, reg=0.05, verbose=True):
    for epoch in range(1, nepoch + 1):
        for _it in range(1, niter + 1):
            # sample mini-batch
            xt = X[np.random.permutation(nmax)[:bsz], :]
            yt = Y[np.random.permutation(nmax)[:bsz], :]
            # compute OT on minibatch
            C = -np.dot(np.dot(xt, R), yt.T)
            P = ot.sinkhorn(np.ones(bsz), np.ones(bsz), C, reg, stopThr=1e-3)
            # compute gradient
            G = - np.dot(xt.T, np.dot(P, yt))
            R -= lr / bsz * G
            # project on orthogonal matrices
            U, s, VT = np.linalg.svd(R)
            R = np.dot(U, VT)
        bsz *= 2
        niter //= 4
        if verbose:
            print("epoch: %d  obj: %.3f" % (epoch, objective(X, Y, R)))
    return R

def convex_init(X, Y, niter=100, reg=0.05, apply_sqrt=False):
    n, d = X.shape
    if apply_sqrt:
        X, Y = sqrt_eig(X), sqrt_eig(Y)
    K_X, K_Y = np.dot(X, X.T), np.dot(Y, Y.T)
    K_Y *= np.linalg.norm(K_X) / np.linalg.norm(K_Y)
    K2_X, K2_Y = np.dot(K_X, K_X), np.dot(K_Y, K_Y)
    P = np.ones([n, n]) / float(n)
    for it in range(1, niter + 1):
        G = np.dot(P, K2_X) + np.dot(K2_Y, P) - 2 * np.dot(K_Y, np.dot(P, K_X))
        q = ot.sinkhorn(np.ones(n), np.ones(n), G, reg, stopThr=1e-3)
        alpha = 2.0 / float(2.0 + it)
        P = alpha * q + (1.0 - alpha) * P
    obj = np.linalg.norm(np.dot(P, K_X) - np.dot(K_Y, P))
    print(obj)
    return procrustes(np.dot(P, X), Y).T