import barberatfmodel
import baselines
import evals
import tfmodel
import reader

import pandas as pd

import sys
from timeit import default_timer as timer

TWITTER, REDDIT = 'twitter', 'reddit'

EXECUTE_FULL_LOGREG_FOR_TWITTER = False

def main(dataset):
    print("Dataset:", dataset)
    assert dataset in {TWITTER, REDDIT}
    if dataset == REDDIT:
        N, K, fold_iterator = reader.generate_reddit_kfolds()
    else:
        N, K, fold_iterator = reader.generate_twitter_kfolds()

    results = pd.DataFrame()
    label2preds = {}
    label2times = {}

    for fold_index, (train_set, test_set) in enumerate(fold_iterator):
        for dim in ((2*K+1), 128):
            for p in (.5, 1., 2.):
                for q in (.5, 1., 2.):
                    for use_topics in (False, True):
                        threshold = 5 if dataset == TWITTER else 0
                        key = (f"node2vec [dim={dim}, p={p}, q={q}, t={threshold}" 
                                    + (" + topics" if use_topics else ""))
                        start = timer()
                        pred = baselines.logistic_regression(N, train_set, test_set,
                                     use_topics=use_topics,
                                     embeddings=baselines.compute_node2vec(
                                        N, train_set, dimensions=dim, p=p, q=q, threshold=threshold)
                        )
                        label2times[key] = timer() - start
                        label2preds[key] = pred
                        print(f"Fold {fold_index}, {key}: Done.")
        
        #### OUR MODEL #######
        model = tfmodel.build_model(N, K)
        start = timer()
        estimated_phi, estimated_theta, _loss_values = tfmodel.optimize(model, train_set,
                                                               decay_epochs=3,
                                                               num_epochs=5)
        label2times['Our model'] = timer() - start
        predictions = tfmodel.test_predictions(model, estimated_phi, estimated_theta, test_set)
        label2preds['Our model'] = predictions
        
        #### BARBERA'S MODEL #######
        bmodel = barberatfmodel.build_model(N,1)
        start = timer()
        estimated_phi_b, estimated_theta_b, estimated_alpha_b, estimated_beta_b, _loss_values_b = barberatfmodel.optimize(bmodel, train_set,
                                                               decay_epochs=3,
                                                               num_epochs=5)
        label2times['Barbera model'] = timer() - start
        predictions_barbera = barberatfmodel.test_predictions(bmodel, estimated_phi_b,
                estimated_theta_b, estimated_alpha_b, estimated_beta_b, test_set)
        label2preds['Barbera model'] = predictions_barbera
        
        if EXECUTE_FULL_LOGREG_FOR_TWITTER or (dataset != TWITTER):
            start = timer()
            logreg_pred = baselines.logistic_regression(N, train_set, test_set, use_topics=False)
            label2preds['Original information'] = logreg_pred
            label2times['Original information'] = timer() - start
            
            start = timer()
            logreg_t_pred = baselines.logistic_regression(N, train_set, test_set, use_topics=True)
            label2preds['Original inf. + Topics'] = logreg_t_pred
            label2times['Original inf. + Topics'] = timer() - start
        
        fold_results, _figure_path = evals.plot_curve(test_set,
                       label2preds=label2preds,
                       label2times=label2times,
                       output_basepath=f"../data/results/{dataset}/roc-{fold_index}-fold.png")

        print(f"Results for fold {fold_index} =======\n", fold_results.to_string(index=False))
        fold_results["Fold"] = fold_index
        results = pd.concat([results, fold_results])
        results.to_csv(f"../data/results/{dataset}/results.csv", index=False)

    print(results.groupby('Algorithm').mean())
    
if __name__ == '__main__':
    main(sys.argv[1] if sys.argv[1:] else REDDIT)
