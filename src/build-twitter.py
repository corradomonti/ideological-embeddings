
import numpy as np
import pandas as pd
from tqdm import tqdm

from collections import defaultdict
import pickle

with open("../data/novax/complete_user2tw.pickle", "rb") as f:
    complete_user2tw = pickle.load(f)

with open("../data/novax/complete_tw2ht.pickle", "rb") as f:
    complete_tw2ht = pickle.load(f)

top_hashtags = pd.read_csv("../data/novax/hashtags.csv")

def filter_by_hashtags(desired_hashtags):
    indexes = {i for i, t in top_hashtags.values if t in desired_hashtags}
    pertinent_tw2ht = {tw: (indexes & set(hts))
                       for tw, hts in tqdm(complete_tw2ht.items()) if indexes & set(hts)}
    pertinent_user2tw = {user: 
                         {tw for tw in tws if tw in pertinent_tw2ht}
                         for user, tws in tqdm(complete_user2tw.items())}
    pertinent_user2tw = {user: tws for user, tws in pertinent_user2tw.items() if tws}
    return pertinent_tw2ht, pertinent_user2tw


def iterative_reduction(user2tw, min_tw_per_user, min_user_per_tw):
    user2tw = user2tw.copy()
    
    tw2user = defaultdict(set)
    for u, tweets in user2tw.items():
        for t in tweets:
            tw2user[t].add(u)
    
    while True:
        num_removed_tweet = 0
        num_removed_users = 0
        for t, users in tqdm(list(tw2user.items()), desc="Pruning tweets"):
            if len(users) < min_user_per_tw:
                del tw2user[t]
                num_removed_tweet += 1
                for u in users:
                    user2tw[u].remove(t)
                    if len(user2tw[u]) == 0:
                        num_removed_users += 1
                        del user2tw[u]
        print("Removed %d tweets, emptying %d users" % (num_removed_tweet, num_removed_users))

        num_removed_users = 0
        num_removed_tweet = 0
        for u, tweets in tqdm(list(user2tw.items()), desc="Pruning users"):
            if len(tweets) < min_tw_per_user:
                del user2tw[u]
                num_removed_users += 1
                for t in tweets:
                    tw2user[t].remove(u)
                    if len(tw2user[t]) == 0:
                        del tw2user[t]
                        num_removed_tweet += 1
        print("Removed %d users, emptying %d tweets" % (num_removed_users, num_removed_tweet))
        
        
        assert all(t in tw2user for tweets in user2tw.values() for t in tweets)
        assert all(u in user2tw for users in tw2user.values() for u in users)
            
        obs_min_tw_per_user = min(map(len, user2tw.values()))
        obs_min_user_per_tw = min(map(len, tw2user.values()))
        print('Observed minimum tweets per user:', obs_min_tw_per_user)
        print('Observed minimum users per tweet:', obs_min_user_per_tw)

        if obs_min_tw_per_user >= min_tw_per_user:
            if obs_min_user_per_tw >= min_user_per_tw:
                return user2tw, tw2user

def compute_gamma(tw2id, selected_hashtags):
    selected_hashtags_index = {i for i, t in top_hashtags.values if t in selected_hashtags}
    hashtags_index2topic_index = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_hashtags_index)}
    gamma = np.zeros((len(tw2id), len(selected_hashtags)))

    for tw, i in tqdm(tw2id.items(), desc="Computing topic matrix"):
        topics = [hashtags_index2topic_index[complete_ht_index] 
                  for complete_ht_index in complete_tw2ht[tw]
                  if complete_ht_index in selected_hashtags_index]
        gamma[i, topics] = 1
            
    gamma = (gamma.T / np.sum(gamma, 1)).T
    print(pd.DataFrame(zip(selected_hashtags, np.mean(gamma, axis=0)), columns=["Topic", "Avg gamma[k]"]))
    return gamma

def build_twitter_dataset(selected_hashtags):
    _pertinent_tw2ht, pertinent_user2tw = filter_by_hashtags(selected_hashtags)
    user2tw, tw2user = iterative_reduction(pertinent_user2tw, min_tw_per_user=100, min_user_per_tw=10)

    user2id = {user: new_id for new_id, user in enumerate(user2tw.keys())}
    tw2id = {tweet: new_id for new_id, tweet in enumerate(tw2user.keys())}

    tw2user = defaultdict(set)
    for u, ts in user2tw.items():
        for t in ts:
            tw2user[tw2id[t]].add(user2id[u])
            
    gamma = compute_gamma(tw2id, selected_hashtags)
    
    print(f"{len(selected_hashtags)} topics, {len(user2id)} users, {len(tw2id)} tweets, {sum(map(len, tw2user.values()))} activations")
    return selected_hashtags, tw2user, gamma
    
if __name__ == '__main__':
    dataset = build_twitter_dataset({
        'Salvini', 
        'Dimaio', 
        'Renzi',
        'Vaccini',
        'Migranti',
        'Tav',
    })
    with open("../data/novax/processed_twitter_dataset.pickle", 'wb') as f:
        pickle.dump(dataset, f)
