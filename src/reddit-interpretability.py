
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf

from collections import Counter, defaultdict
import os
from pathlib import Path

import tfmodel
from reader import generate_activation_data

def read_all_reddit():
    gamma = pd.read_pickle("../data/reddit-politics/items-doc2vec-cluster.pickle")
    tiu = pd.read_csv("../data/reddit-politics/tiu.csv.gz")
    
    nodes = set()
    i2u2t = defaultdict(dict)
    for t, i, u in tiu.values:
        i2u2t[i][u] = t
        nodes.add(u)
    
    item_list = tiu['item'].unique()
    return max(nodes) + 1, gamma.shape[1], list(generate_activation_data(i2u2t, nodes, item_list, gamma))

def main():
    results_path = Path("../results/reddit/")
    if not results_path.exists():
        os.makedirs(results_path)

    N, K, dataset = read_all_reddit()

    np.random.seed(12345678)
    tf.set_random_seed(12345678)

    model = tfmodel.build_model(N, K, initial_variance=0.25)

    grid = [tfmodel.optimize(model, dataset, decay_epochs=8, num_epochs=10) for _ in range(10)]

    subreddit2id = pd.read_pickle("../data/reddit-politics/subreddits.pickle")
    id2subreddit = {s: i for i, s in subreddit2id.items()}

    topics = ["Campaign", "Foreign policy", "Minorities", "Economy", "Emailgate"]

    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    best_loss = min(loss[-1] for phi, th, loss in grid)

    most_popular = [i for i, count in Counter([x[k] for x in dataset for k in 'uv']).most_common(15)]

    i = 0
    for estimated_phi, _estimated_theta, loss in grid:
        if loss[-1] == best_loss:
            for x, y in [("Economy", "Emailgate"), ("Minorities", "Foreign policy")]:
                df = pd.DataFrame(sigmoid(estimated_phi), columns=topics)
                df["Subreddit"] = [id2subreddit[i] for i in df.index]

                plt.figure(figsize=(7, 7))
                sns.set_style("whitegrid")
                assert x in df.columns and y in df.columns

                pl = sns.scatterplot(data=df[df.index.isin(most_popular)], x=x, y=y, marker='o', s=200, alpha=0.5)
                for line in df.index:
                    if line in most_popular:
                        pl.text(df[x][line], df[y][line],
                                 df["Subreddit"][line],
                                 horizontalalignment='left', size='large',
                                 color='black')
                plt.savefig(results_path / f"reddit-polarity-plot-{i}.pdf")
                i += 1
    print(f"Results saved in {results_path.resolve()} ✅")

if __name__ == '__main__':
    main()
