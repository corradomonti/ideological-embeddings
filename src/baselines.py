import node2vec

import networkx as nx
import numpy as np
import sklearn.linear_model
from tqdm import tqdm

from collections import Counter

def one_hot(v, N):
    vec = np.zeros(N)
    vec[v] = 1.
    return vec
    
def sklearn_X_Y(N, training_set, use_topics=True, embeddings=None):
    X, Y = [], []
    for t in tqdm(training_set, desc=f"Building matrix {'with' if use_topics else 'without'} topics, {'with' if embeddings is not None else 'without'} embeddings"):
        encoded = []
        if use_topics:
            encoded += [t['gamma']]
        if embeddings is not None:
            hadamard_product = embeddings[t['u']] * embeddings[t['v']]
            encoded += [hadamard_product]
        else:
            encoded += [one_hot(t['v'], N), one_hot(t['u'], N)]
        X.append(np.concatenate(encoded))
        Y.append(t['is_active'])
    return np.array(X), np.array(Y)

def logistic_regression(N, training_set, test_set, use_topics=True, embeddings=None):
    X, Y = sklearn_X_Y(N, training_set, use_topics=use_topics, embeddings=embeddings)
    print(f"Train: X shape: {X.shape}, Y shape: {Y.shape}")
    logistic = sklearn.linear_model.LogisticRegression()
    logistic.fit(X, Y)
    X_test, _Y_test = sklearn_X_Y(N, test_set, use_topics=use_topics, embeddings=embeddings)
    print(f"Test: X shape: {X_test.shape}, Y shape: {_Y_test.shape}")
    Y_pred = logistic.predict_log_proba(X_test)
    return Y_pred[:, 1]

def compute_node2vec(N, train_set, dimensions=128, threshold=0, p=1, q=1, iterations=5,
                     walk_length=10, window_size=5, num_walks=500):
    print("Computing node2vec. Preprocessing node2vec graph...")
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    weighted_edges = Counter([(x['u'], x['v']) for x in train_set]).items()
    max_weight = max(w for (u, v), w in weighted_edges)
    for (u, v), w in tqdm(weighted_edges, desc="Building node2vec graph"):
        if w > threshold:
            G.add_edge(u, v, weight=w/max_weight)
    
    print("Computing node2vec...")
    model = node2vec.main(G, p=p, q=q, dimensions=dimensions,
       iterations=iterations, walk_length=walk_length, window_size=window_size, num_walks=num_walks)
    vectors = np.array([model.wv.get_vector(str(i)) for i in range(N)])
    assert vectors.shape == (N, dimensions)
    return vectors
