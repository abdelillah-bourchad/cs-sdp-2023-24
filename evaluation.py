import os
import sys

sys.path.append("python/")

import metrics
import numpy as np
from data import Dataloader
from models import TwoClustersMIP, HeuristicModel
from time import time


if __name__ == "__main__":
    ### First part: test of the MIP model
    data_loader = Dataloader("data/dataset_4")  # Path to test dataset
    X, Y = data_loader.load()

    model = TwoClustersMIP(
        n_clusters=2, n_pieces=5
    )  # You can add your model's arguments here, the best would be set up the right ones as default.
    
    s = time()
    # The first formulation with errors minimization 
    model.fit(X, Y) 
    d = round((time()-s)/60, 2)
    print(f'Execution time : {d} min')
    
    # Uncoment To run the fit with the second problem formulation (the one that minimizes the sum of the binary vector)
    
    # model.fit_beta(X, Y)  # The second formulation

    # %Pairs Explained
    pairs_explained = metrics.PairsExplained()
    print("Percentage of explained preferences:", pairs_explained.from_model(model, X, Y))

    # %Cluster Intersection
    cluster_intersection = metrics.ClusterIntersection()

    Z = data_loader.get_ground_truth_labels()
    print("% of pairs well grouped together by the model:")
    print("Cluster intersection for all samples:", cluster_intersection.from_model(model, X, Y, Z))

    ### 2nd part: test of the heuristic model
    data_loader = Dataloader("data/dataset_10")  # Path to test dataset
    X, Y = data_loader.load()

    indexes = np.linspace(0, len(X) - 1, num=len(X), dtype=int)
    np.random.shuffle(indexes)
    train_indexes = indexes[: int(len(indexes) * 0.8)]
    test_indexes = indexes[int(len(indexes) * 0.8) :]

    print(train_indexes, test_indexes)
    X_train = X[train_indexes]
    Y_train = Y[train_indexes]
    
    
    hyper_params = {'positive_mutation':0.8, 'negative_mutation':0.8, 'instances_limit':1000}
    model = HeuristicModel(n_clusters=3, n_features=X_train.shape[1], n_instances=X_train.shape[0], n_pieces=5, n_iter=100, hyper_params=hyper_params)
    model.fit(X_train, Y_train)

    X_test = X[test_indexes]
    Y_test = Y[test_indexes]
    Z_test = data_loader.get_ground_truth_labels()[test_indexes]

    # Validation on test set
    # %Pairs Explained
    pairs_explained = metrics.PairsExplained()
    print("Percentage of explained preferences:", pairs_explained.from_model(model, X_test, Y_test))

    # %Cluster Intersection
    cluster_intersection = metrics.ClusterIntersection()
    print("% of pairs well grouped together by the model:")
    print(
        "Cluster intersection for all samples:",
        cluster_intersection.from_model(model, X_test, Y_test, Z_test),
    )
