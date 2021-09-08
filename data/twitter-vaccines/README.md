# Twitter Vaccines Dataset

This data set has been obtained by processing the data gathered for

> "Falling into the Echo Chamber: the Italian Vaccination Debate on Twitter"
> by Alessandro Cossard, Gianmarco De Francisci Morales, Yelena Mejova, Daniela Paolotti

From their data (7152 users and their tweets), we build a processed data file by (i) selecting a subset of hashtags and (ii) removing users and tweets so that every Twitter user in the data set has at least 100 retweets and every retweet has been shared by at least 10 users. This processed file is saved in `processed_twitter_dataset.pickle`. This file is a pickled tuple with

```
(selected_hashtags, tw2user, gamma)
```

Where `selected_hashtags` is the list of selected hashtags, `tw2user` is a map from tweet to users, each one with an anonymized 0-to-N index, and gamma is a numpy matrix that in row i contains the hashtag-based topics of tweet i (each column correspond to a hashtag in `selected_hashtags`).
