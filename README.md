# Learning Ideological Embeddings From Information Cascades

This repository contains code and data related to the paper ***"Learning Ideological Embeddings From Information Cascades"*** by Corrado Monti, Giuseppe Manco, Çigdem Aslay and Francesco Bonchi, published at CIKM 2021. If you use the provided data or code, we would appreciate a citation to the paper:

```
@inproceedings{monti2020learningideological,
  title={Learning Ideological Embeddings From Information Cascades},
  author={Monti, Corrado and Manco, Giuseppe and Aslay, Çigdem and Bonchi, Francesco},
  booktitle={Proceedings of the 30th ACM International Conference on Information and Knowledge Management (CIKM)},
  year={2021}
}
```

Here you will find (i) code to apply the algorithm we designed on your data (ii) the Reddit and the (anonymized) Twitter dataset we used in the paper (iii) code to reproduce our experiments.

## Ideology2vec tool

In [`ideology2vec.py`](ideology2vec.py), we provide a Python module to use our algorithm on your data. It requires `tqdm`(https://github.com/tqdm/tqdm) and `tensorflow`(https://github.com/tensorflow/tensorflow/).

This module contains a function called `compute_ideological_embeddings` that turns propagations of items with known topics on a graph into multidimensional polarities for the nodes of the graph. To understand the details, consult the documentation of the function or the [`ideology2vec-example.ipynb`](ideology2vec-example.ipynb) notebook, that contains a toy example for its usage.

## Provided data set

In [`data/`](data/), we provide the data sets we used in our paper:

- [`reddit-politics`](data/reddit-politics/) contains data about which subreddits posted certain URLs.

- [`twitter-vaccines`](data/twitter-vaccines/) contains an anonymized retweet data set: no text, usernames, or any other personal info is provided, only a bipartite graph of retweets.

For both data set, there is a README in their directory providing acknowledgements and further details.

## Reproducibility

In order to reproduce our experiments, we provide [our scripts in `src/`](src/). They need a set of dependencies, listed in `environment.yml`. To install them and run experiments, do:

```
conda create --name multid --file environment.yml
conda activate multid
cd src
python synthetic.py
python main.py twitter
python main.py reddit
```
