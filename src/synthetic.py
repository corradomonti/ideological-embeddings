import tfmodel

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

import random

DEFAULT_POLARIZATION = 4.
DEFAULT_MIX_TOPICS = .125
DEFAULT_INTEREST_ALPHA = .9
DEFAULT_INTEREST_BETA = .1

def generate_training_set(N, K, num_items, graph,
    polarization=DEFAULT_POLARIZATION, mix_topics=DEFAULT_MIX_TOPICS,
    interest_alpha=DEFAULT_INTEREST_ALPHA, interest_beta=DEFAULT_INTEREST_BETA,
    true_phi=None, true_theta=None):
    
    if true_theta is None:
        true_theta = np.random.beta(interest_alpha, interest_beta, size=(N, K))
    if true_phi is None:
        true_phi = np.random.beta(1. / polarization, 1. / polarization, size=(N, K))
        
    train = []
    for _ in tqdm(range(num_items)):
        u = np.random.randint(N)
        gamma = np.random.dirichlet([mix_topics] * K)
        exposed = {(u, v) for v in graph[u]}
        already_seen = {u} | graph[u]
        
        while exposed:
            u, v = random.choice(list(exposed))
            exposed.remove((u, v))
            topic = np.random.choice(np.arange(K), p=gamma)
            polarity_u = np.random.random() < true_phi[u, topic]
            polarity_v = np.random.random() < true_phi[v, topic]
            is_active = (
                (np.random.random() < true_theta[v][topic]) and 
                (polarity_u == polarity_v)
            )
            if is_active:
                for w in graph[v]:
                    if w not in already_seen:
                        exposed.add((v, w))
                        already_seen.add(w)
        
            train.append({'u': u, 'v': v, 'gamma': gamma, 'is_active': is_active})
    return true_theta, true_phi, train

def synth_experiment(N, K, num_items, graph, polarization, graph_name):
    print(f"Synthetic experiment with {N} nodes, {K} topics, {num_items} items, "
          f"polarization {polarization}, on graph '{graph_name}'.")
    true_theta, true_phi, training_set = generate_training_set(N=N, K=K, num_items=num_items,
        graph=graph, polarization=polarization)
    _, _, test_set = generate_training_set(N=N, K=K, num_items=(num_items // 10),
        graph=graph, true_phi=true_phi, true_theta=true_theta, polarization=None)

    model = tfmodel.build_model(N, K)
    estimated_phi, estimated_theta, _loss_values = tfmodel.optimize(model, training_set,
                                                           decay_epochs=10,
                                                           num_epochs=15,
                                                           starter_learning_rate=1.0, 
                                                           end_learning_rate=0.01)
    test_labels = np.array([x['is_active'] for x in test_set ], dtype=np.bool)
    test_predictions = tfmodel.test_predictions(model, estimated_phi, estimated_theta, test_set)
    return pd.DataFrame([
        [graph_name, N, K, num_items, polarization] + 
        [true_theta, true_phi, estimated_phi, estimated_theta, test_labels, test_predictions]
    ], columns=(
        "graph_name, N, K, num_items, polarization".split(", ") +
        "true_theta, true_phi, estimated_phi, estimated_theta, test_labels, test_predictions".split(", ")
    ))


def main():
    random.seed(1234)
    np.random.seed(1234)
    
    N = 100
    K = 4
    df = pd.DataFrame()
    
    COMPLETE, BARABASI = 'complete', 'barabasi'
    
    for graph_name in (COMPLETE, BARABASI):
        if graph_name == COMPLETE:
            graph = {u: {v for v in range(N) if u != v} for u in range(N)}
        else:
            nx_graph = nx.barabasi_albert_graph(N, 10)
            graph = {u: set(nx_graph.neighbors(u)) for u in nx_graph.nodes}
        
        for num_items in (1000, 10000, 100000):
            polarization = 4
            results = synth_experiment(N=N, K=K, num_items=num_items, graph=graph,
                polarization=polarization, graph_name=graph_name)
            df = df.append(results, ignore_index=True)
            df.to_pickle("../data/results/synthetic.pickle")
        
        for polarization in (1, 16):
            num_items = 10000
            results = synth_experiment(N=N, K=K, num_items=num_items, graph=graph,
                polarization=polarization, graph_name=graph_name)
            df = df.append(results, ignore_index=True)
            df.to_pickle("../data/results/synthetic.pickle")
    
if __name__ == '__main__':
    main()
