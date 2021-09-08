import numpy as np
import tensorflow as tf
from tqdm import tqdm

from collections import namedtuple
import itertools

EPSILON = 1E-10
BATCH_SIZE = None

def build_model(N, K, initial_variance=0.25):
    tfgraph = tf.Graph()
    with tfgraph.as_default():
        spreader = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='u')
        receiver = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='v')
        gamma = tf.placeholder(tf.float32, shape=[BATCH_SIZE, K], name='gamma')
        is_active = tf.placeholder(tf.float32, shape=[BATCH_SIZE], name='is_active')

        theta = tf.Variable(tf.constant(0.5, shape=[N, K]), name='theta')
        phi = tf.Variable(tf.random_uniform([N, K], - initial_variance, + initial_variance), name='phi')
        phi_u_hat = tf.nn.embedding_lookup(phi, spreader)
        phi_v_hat = tf.nn.embedding_lookup(phi, receiver)
        
        phi_u_hat = tf.clip_by_value(phi_u_hat, -5, 5)
        phi_v_hat = tf.clip_by_value(phi_v_hat, -5, 5)
        phi_u = tf.sigmoid(phi_u_hat)
        phi_v = tf.sigmoid(phi_v_hat)

        theta_v_hat = tf.nn.embedding_lookup(theta, receiver)
        theta_v = tf.clip_by_value(theta_v_hat, 0, 1)

        embedded_input = [gamma, theta_v_hat, phi_u_hat, phi_v_hat]

        polarity_agreement = phi_u * phi_v + (1. - phi_u) * (1. - phi_v)
        topic_likelihood = gamma * theta_v * polarity_agreement
        likelihood = tf.clip_by_value(tf.reduce_sum(topic_likelihood, axis=1), 0., 1.)
        loss = tf.losses.log_loss(is_active, likelihood)
        
        Model = namedtuple('Model',
            'tfgraph, K, spreader, receiver, gamma, is_active, theta, phi, embedded_input, likelihood, loss')
        return Model(
             tfgraph, K, spreader, receiver, gamma, is_active, theta, phi, embedded_input, likelihood, loss
        )
        
def minibatch_from_iterator(dataset_iterator, batch_size):
    while True:
        minibatch_tuple = tuple(itertools.islice(dataset_iterator, batch_size))
        if not minibatch_tuple:
            break
        yield [np.array(minibatch_field) for minibatch_field in zip(*minibatch_tuple)]

def optimize(model, training_set, num_epochs=5, 
            decay_epochs=3, starter_learning_rate=1.0, end_learning_rate=0.01,
            train_batch_size=2048):
    
    with model.tfgraph.as_default():
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.compat.v1.train.polynomial_decay(starter_learning_rate,
          global_step,
          int(decay_epochs * len(training_set) / train_batch_size),
          end_learning_rate,
          power=1.0)
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(model.loss, global_step=global_step)
    
    loss_values = []
    pbar = tqdm(total=(num_epochs * len(training_set)))

    with tf.Session(graph=model.tfgraph) as session:
        session.run(tf.global_variables_initializer())
        for epoch in range(num_epochs):
            np.random.shuffle(training_set)
            dataset_iterator = ( (x['gamma'], x['u'], x['v'], x['is_active']) for x in training_set)
            for tr_gamma, tr_u, tr_v, tr_is_active in minibatch_from_iterator(dataset_iterator, train_batch_size):
                feed_dict = {model.spreader: tr_u, model.receiver: tr_v,
                             model.is_active: tr_is_active, model.gamma:  tr_gamma}
                _, current_loss, lr_val = session.run(
                                                                [optimizer, model.loss, learning_rate],
                                                                feed_dict=feed_dict)
                loss_values.append(current_loss)
                pbar.update(train_batch_size)
                pbar.set_postfix(epoch=epoch, loss=current_loss, learning_rate=lr_val,)
            if epoch > decay_epochs and np.isnan(current_loss):
                raise OverflowError()
        pbar.close()
        estimated_phi = model.phi.eval() if model.phi is not None else None
        estimated_theta = model.theta.eval() if model.theta is not None else None
        return estimated_phi, estimated_theta, loss_values

def test_predictions(model, phi, theta, test_set):
    test_u, test_v = np.array([np.array([x['u'], x['v']]) for x in test_set ]).T
    test_gamma = np.array([np.array(x['gamma']) for x in test_set ])
        
    with tf.Session(graph=model.tfgraph) as session:
        model.phi.load(phi)
        model.theta.load(theta)
        feed_dict = {model.spreader: test_u, model.receiver: test_v,  model.gamma:  test_gamma}
        predictions = session.run(model.likelihood, feed_dict=feed_dict)
        return np.exp(predictions)
