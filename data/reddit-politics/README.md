## Politics Subreddit dataset

### Extraction

- Each node is a subreddit, each item is a URL.
- We consider the time span from 1/1/2011 to 1/1/2019 (8 years of Reddit).
- We consider only the top 50 subreddits most similar to `r/politics`.
- We consider only URLs that have been posted to more of 5 of those subreddits (22047 URLs).
- Then, we extracted the topics of these URLs by running doc2vec and then clustering these vectors around K=5 clusters. Each URL is then represented as a point in 5-dimensional topic space, according to fuzzy K-Means.

### Content

- `items.csv.gz` is a gzipped csv containing the index of each item, its URL, and all the titles that it has been been posted with. The titles of each Reddit post for a single URL are separated by `§`.
- `subreddits.pickle` is a pickled `dict` associating the 50 subreddits to their indexes.
- `tiu.csv.gz` is a gzipped csv where each line reports the time (as unix timestamp) in which a URL was posted, the index of such URL, and the index of the subreddit it was posted to.
- `items-doc2vec-cluster.pickle` is a pickled numpy matrix with the doc2vec clustering output: the i-th row corresponds to the topic distribution of the i-th document.
