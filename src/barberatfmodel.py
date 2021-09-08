import numpy as np
import tensorflow as tf

from tqdm import tqdm

from collections import namedtuple
import itertools

EPSILON = 1E-3
BATCH_SIZE = None

def build_model(N, K, initial_variance=0.25, lambdas = None):
    tfgraph = tf.Graph()
    with tfgraph.as_default():
        spreader = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='u')
        receiver = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='v')
        is_active = tf.placeholder(tf.float32, shape=[BATCH_SIZE], name='is_active')

        theta = tf.Variable(tf.random.uniform([N, K], - initial_variance, + initial_variance), name='theta')
        phi = tf.Variable(tf.random.uniform([N, K], - initial_variance, + initial_variance), name='phi')
        alpha = tf.Variable(tf.random.uniform([N,1], - initial_variance, + initial_variance), name='alpha')
        beta = tf.Variable(tf.random.uniform([N,1], - initial_variance, + initial_variance), name='beta')


        theta_i = tf.nn.embedding_lookup(theta, receiver)
        phi_j = tf.nn.embedding_lookup(phi, spreader)

        beta_i = tf.squeeze(tf.nn.embedding_lookup(alpha, receiver),axis=1)
        alpha_j = tf.squeeze(tf.nn.embedding_lookup(beta, spreader),axis=1)

        logit = alpha_j + beta_i - tf.reduce_sum(tf.pow(theta_i - phi_j, 2), axis=1)

        loss = tf.keras.losses.binary_crossentropy(is_active,logit, from_logits = True)

        phi_reg = tf.nn.l2_loss(phi)
        theta_reg = tf.nn.l2_loss(theta)
        alpha_reg = tf.nn.l2_loss(alpha)
        beta_reg  = tf.nn.l2_loss(beta)

        if lambdas is None or len(lambdas) < 4:
            loss = loss + 0.001 * (phi_reg + theta_reg + alpha_reg + beta_reg)
        else:
            loss = loss + lambdas[0] * phi_reg + lambdas[1] * theta_reg + lambdas[2] * alpha_reg + lambdas[3] * beta_reg
        
        Model = namedtuple('Model',
            'tfgraph, K, spreader, receiver, is_active, theta, phi, alpha, beta, logit, loss')
        return Model(
             tfgraph, K, spreader, receiver, is_active, theta, phi, alpha, beta, logit, loss
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
                             model.is_active: tr_is_active}
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
        estimated_alpha = model.alpha.eval() if model.alpha is not None else None
        estimated_beta = model.beta.eval() if model.beta is not None else None
        return estimated_phi, estimated_theta, estimated_alpha, estimated_beta, loss_values

def test_predictions(model, phi, theta, alpha, beta, test_set):
    test_u, test_v = np.array([np.array([x['u'], x['v']]) for x in test_set ]).T
        
    with tf.Session(graph=model.tfgraph) as session:
        model.phi.load(phi)
        model.theta.load(theta)
        model.alpha.load(alpha)
        model.beta.load(beta)
        feed_dict = {model.spreader: test_u, model.receiver: test_v}
        predictions = session.run(model.logit, feed_dict=feed_dict)
        return np.exp(predictions)
