import numpy as np
import pandas as pd
import sklearn.model_selection
from tqdm import tqdm

from collections import Counter, defaultdict
import pickle

def generate_activation_data(i2u2t, nodes, selected_items, gamma):
    for i in tqdm(selected_items):
        u = min(i2u2t[i], key=i2u2t[i].get)
        positives = {v for v in i2u2t[i] if v != u}
        negatives = {v for v in nodes if v != u and v not in positives}
        negatives = set(np.random.choice(list(negatives), size=len(positives)))
        for v in positives:
            yield {"gamma": list(gamma[i]),
                   "u": int(u),
                   "v": int(v),
                   "is_active": True}
        for v in negatives:
            yield {"gamma": list(gamma[i]),
                   "u": int(u),
                   "v": int(v),
                   "is_active": False}
                   
def generate_reddit_kfolds(k=10, seed=123456789):
    """ Read and parses the Reddit dataset, including its doc2vec-based topic distribution. 
    Returns:
        N: number of nodes in the data set
        K: number of topics in the data set
        fold_iterator: an iterator over (train_set, test_set) for each fold.
    """
    
    gamma = pd.read_pickle("../data/politics-subreddit/items-doc2vec-cluster.pickle")
    tiu = pd.read_csv("../data/politics-subreddit/tiu.csv.gz")
    
    nodes = set()
    i2u2t = defaultdict(dict)
    for t, i, u in tiu.values:
        i2u2t[i][u] = t
        nodes.add(u)
    
    item_list = tiu['item'].unique()
    print("K-Fold seed:", seed)
    kfold = sklearn.model_selection.KFold(n_splits=k, random_state=seed)
    kfold.get_n_splits(item_list)
    
    fold_iterator = (
        (
            list(generate_activation_data(i2u2t, nodes, train_items, gamma)),
            list(generate_activation_data(i2u2t, nodes, test_items, gamma))
        ) for train_items, test_items in kfold.split(item_list)
    )
    
    return max(nodes) + 1, gamma.shape[1], fold_iterator


def generate_twitter_activation_data(tw2user, node2nodes_who_shared_an_item, item_set, gamma, neg2pos_ratio=2):
    for i in tqdm(item_set, desc="Building fold"):
        us = np.random.choice(list(tw2user[i]), size=10) if len(tw2user[i]) > 10 else tw2user[i]
        for u in us:
            positives = {v for v in tw2user[i] if v != u}
            candidate_negatives = {v for v, count in node2nodes_who_shared_an_item[u].items() if count > 1}
            negatives = {v for v in candidate_negatives if v != u and v not in positives}
            num_negatives = neg2pos_ratio * len(positives)
            if len(negatives) > num_negatives:
                negatives = set(np.random.choice(list(negatives), size=num_negatives))
            for v in positives:
                yield {"gamma": list(gamma[i]),
                       "u": int(u),
                       "v": int(v),
                       "is_active": True}
            for v in negatives:
                yield {"gamma": list(gamma[i]),
                       "u": int(u),
                       "v": int(v),
                       "is_active": False}

def generate_twitter_kfolds(k=10, seed=123456789):
    with open("../data/novax/processed_twitter_dataset.pickle", "rb") as f:
        selected_hashtags, tw2user, gamma = pickle.load(f)
    users = {u for us in tw2user.values() for u in us}
    print(f"Hashtags: {selected_hashtags}\n{len(tw2user)} items, {len(users)} users")

    node2nodes_who_shared_an_item = {u: Counter() for u in users}
    for _tw, us in tqdm(tw2user.items(), desc="Building graph"):
        for u in us:
            for v in us:
                node2nodes_who_shared_an_item[u][v] += 1
    
    item_list = range(len(gamma))
    kfold = sklearn.model_selection.KFold(n_splits=k, random_state=seed)
    kfold.get_n_splits(item_list)
    
    fold_iterator = (
        (
            list(generate_twitter_activation_data(tw2user, node2nodes_who_shared_an_item, train_items, gamma)),
            list(generate_twitter_activation_data(tw2user, node2nodes_who_shared_an_item, test_items, gamma))
        ) for train_items, test_items in kfold.split(item_list)
    )
    
    return len(users), gamma.shape[1], fold_iterator
