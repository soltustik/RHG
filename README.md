# RHG
Construct word embeddings from the Random Hyperbolic Graph

### Prerequisites

Code is written in Python 3.6. It requires the following Python modules: `gensim`, `argparse`, `networkit`, `networkx`, `torch`, `sklearn`. 
You can install them via:
```
pip install gensim argparse networkit networkx torch sklearn
```
It also requires Python Optimal Transport `ot`. `cython` and `numpy` need to be installed prior to installing POT. This can be done easily with
```
pip install numpy cython
```

```
pip install POT
```

Note that for easier access the module is name ot instead of pot.

# Running the tests

To create and evaluate word vectors from RHG for 3446 words 

```
mkdir embeddings
ipython main.py
```



