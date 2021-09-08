"""
Code to compute the ideological embeddings on your data set of topical activations.
The façade function of the module is `compute_ideological_embeddings`. Please see its documentation.
For more information, see the paper:

"Modeling Information Cascades in Multidimensional Ideological Space"
by Corrado Monti, Giuseppe Manco, Cigdem Aslay, Francesco Bonchi.
Proceedings of the 30th ACM International Conference on Information and Knowledge Management
(CIKM2021).

Requirements: tensorflow, tqdm.

"""

import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import placeholder, Session, global_variables_initializer
from tensorflow.compat.v1.train import AdagradOptimizer
from tensorflow.compat.v1.losses import log_loss
from tqdm.autonotebook import tqdm

from collections import namedtuple
import itertools
from operator import itemgetter

EPSILON = 1E-10
BATCH_SIZE = None

Example = namedtuple("Example", ["gamma", "src", "dest", "successful"])

def _generate_examples(item2nodes, graph, gamma, item_set=None, neg2pos_ratio=2., max_seed_set=10):
    if item_set is None:
        item_set = range(len(gamma))
    for i in tqdm(item_set, desc="Building activations"):
        us = np.random.choice(list(item2nodes[i]), size=max_seed_set) if len(item2nodes[i]) > max_seed_set else item2nodes[i]
        for u in us:
            positives = {v for v in item2nodes[i] if v != u}
            negatives = {v for v in graph[u] if v != u and v not in positives}
            num_negatives = int(neg2pos_ratio * len(positives))
            if len(negatives) > num_negatives:
                negatives = np.random.choice(list(negatives), size=num_negatives)
            for v in positives:
                yield Example(gamma=list(gamma[i]), src=int(u), dest=int(v), successful=True)
            for v in negatives:
                yield Example(gamma=list(gamma[i]), src=int(u), dest=int(v), successful=False)

def _build_model(N, K, initial_variance=0.25):
    tfgraph = tf.Graph()
    with tfgraph.as_default():
        spreader = placeholder(tf.int32, shape=[BATCH_SIZE], name='u')
        receiver = placeholder(tf.int32, shape=[BATCH_SIZE], name='v')
        gamma = placeholder(tf.float32, shape=[BATCH_SIZE, K], name='gamma')
        is_active = placeholder(tf.float32, shape=[BATCH_SIZE], name='is_active')

        theta = tf.Variable(tf.constant(0.5, shape=[N, K]), name='theta')
        phi = tf.Variable(
            tf.random.uniform([N, K], - initial_variance, + initial_variance),
            name='phi'
        )
        phi_u_hat = tf.nn.embedding_lookup(phi, spreader)
        phi_v_hat = tf.nn.embedding_lookup(phi, receiver)
        
        phi_u_hat = tf.clip_by_value(phi_u_hat, -5, 5)
        phi_v_hat = tf.clip_by_value(phi_v_hat, -5, 5)
        phi_u = tf.sigmoid(phi_u_hat)
        phi_v = tf.sigmoid(phi_v_hat)

        theta_v_hat = tf.nn.embedding_lookup(theta, receiver)
        theta_v = tf.clip_by_value(theta_v_hat, 0, 1)

        polarity_agreement = phi_u * phi_v + (1. - phi_u) * (1. - phi_v)
        topic_likelihood = gamma * theta_v * polarity_agreement
        likelihood = tf.clip_by_value(tf.reduce_sum(topic_likelihood, axis=1), 0., 1.)
        loss = log_loss(is_active, likelihood)
        
        Model = namedtuple('Model',
            'tfgraph, K, spreader, receiver, gamma, is_active, theta, phi, likelihood, loss')
        return Model(
             tfgraph, K, spreader, receiver, gamma, is_active, theta, phi, likelihood, loss
        )
        
def _minibatch_from_iterator(dataset_iterator, batch_size):
    while True:
        minibatch_tuple = tuple(itertools.islice(dataset_iterator, batch_size))
        if not minibatch_tuple:
            break
        yield [np.array(minibatch_field) for minibatch_field in zip(*minibatch_tuple)]

def _optimize(model, training_set, num_epochs=5, decay_epochs=3,
             starter_learning_rate=1.0, end_learning_rate=0.01,
             train_batch_size=2048):
    
    with model.tfgraph.as_default():
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.compat.v1.train.polynomial_decay(starter_learning_rate,
          global_step,
          int(decay_epochs * len(training_set) / train_batch_size),
          end_learning_rate,
          power=1.0)
        optimizer = AdagradOptimizer(learning_rate=learning_rate).minimize(model.loss, global_step=global_step)
    
    loss_values = []
    progress = tqdm(total=(num_epochs * len(training_set)), desc="Optimizing")

    with Session(graph=model.tfgraph) as session:
        session.run(global_variables_initializer())
        for epoch in range(num_epochs):
            np.random.shuffle(training_set)
            for tr_gamma, tr_u, tr_v, tr_is_active in _minibatch_from_iterator(iter(training_set), train_batch_size):
                feed_dict = {model.spreader: tr_u, model.receiver: tr_v,
                             model.is_active: tr_is_active, model.gamma:  tr_gamma}
                _, current_loss, lr_val = session.run([optimizer, model.loss, learning_rate], feed_dict=feed_dict)
                loss_values.append(current_loss)
                progress.update(train_batch_size)
                progress.set_postfix(epoch=epoch, loss=current_loss, learning_rate=lr_val,)
            if epoch > decay_epochs and np.isnan(current_loss):
                raise OverflowError()
        progress.close()
        estimated_phi = model.phi.eval() if model.phi is not None else None
        estimated_theta = model.theta.eval() if model.theta is not None else None
        return estimated_phi, estimated_theta, loss_values[-1]

def _predict(model, polarities, interests, spreader, candidate, item2topics):
    with Session(graph=model.tfgraph) as session:
        model.phi.load(polarities)
        model.theta.load(interests)
        feed_dict = {model.spreader: spreader, model.receiver: candidate, model.gamma: item2topics}
        predictions = session.run(model.likelihood, feed_dict=feed_dict)
        return np.exp(predictions)
        
def compute_ideological_embeddings(item2nodes, graph, item2topics, 
    multiple_restarts=1, neg2pos_ratio=1., max_seed_set=50, initial_variance=0.25,
    **optimization_kwargs):
    """ Computes the ideological embeddings: given a graph and a set of propagations (items
        reshared by a set of nodes, with each item having a known topic distribution), it
        computes the embeddings for each node; that is, their polarities on axes defined by
        the topic space.
    
    Args:
        item2nodes (dict): a dictionary from item to the set of nodes that reshared that item.
            They all need to be represented as sequential integers; for instance, nodes have
            to be integers from 0 to N-1.
        graph (dict): a dictionary from each node to their followers (all nodes must be
            integers from 0 to N-1).
        item2topics: dictionary or numpy matrix such that item2topics[i] returns the numpy vector
            representing the topic distribution of item i. Each topic distribution must sum to 1.
    
    Kwargs:
        multiple_restarts (int): how many times the algorithm will be executed with different initialization.
            Increasing this is the easy way to improve results; time is linear w.r.t. this parameter.
        seed (int): 
        neg2pos_ratio (float): number of negative examples taken for each positive example.
        max_seed_set (int): upper bound for the number of nodes that are taken as source for each item.
        initial_variance (float): variance of the opinions in random initialization.
    
    Returns:
        polarities (np.array): the ideological embeddings, as a numpy matrix φ such that φ[u, z] is the embedding of node u on topic z.
            They are in the whole [-inf, inf] range, can be converted on [0, 1] through a sigmoid 1 / (1 + np.exp(-x)) 
        interests (np.array): interests of each node, as a numpy matrix θ such that θ[u, z] is the interest of node u on topic z as a [0, 1] value.
        loss (float): the avg log likelihood as a measure of quality of the computed embeddings on this data.
    """
    N = len(graph)
    K = len(item2topics[0])
    Q = len(item2topics)
    
    assert all( 0 <= src < N for src in graph), \
        "All keys in graph must be indexes between 0 and N"
    assert all( 0 <= dst < N for succs in graph.values() for dst in succs), \
        "All successors in graph must be indexes between 0 and N"
    assert all( 0 <= u < N for nodes in item2nodes.values() for u in nodes), \
        "All nodes in item2nodes must be indexes between 0 and N"
    assert all(len(item2topics[i]) == K for i in range(Q)), \
        "Each value in item2topics must be a K-dimensional vector."
    
    examples = list(_generate_examples(item2nodes, graph, item2topics,
        neg2pos_ratio=neg2pos_ratio, max_seed_set=max_seed_set))
    
    results = []
    multiple_restarts_iter = range(multiple_restarts)
    if multiple_restarts > 1:
        multiple_restarts_iter = tqdm(multiple_restarts_iter, desc="Multiple restarts")
    for _ in multiple_restarts_iter:
        tfmodel = _build_model(N, K, initial_variance=initial_variance)
        results.append(_optimize(tfmodel, examples, **optimization_kwargs))
    return min(results, key=itemgetter(2)) # Returns the results with the minimum loss.
        
    
