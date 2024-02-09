import os
import sys

sys.path.append("python/")

import metrics
import numpy as np
from data import Dataloader
# from models_using_errors import HeuristicModel, TwoClustersMIP
from models import TwoClustersMIP, HeuristicModel


if __name__ == "__main__":
    #===========================================================================================================================
    ### First part: test of the MIP model
    # data_loader = Dataloader("data/dataset_4")  # Path to test dataset
    # X, Y = data_loader.load()

    # model = TwoClustersMIP(
    #     n_clusters=2, n_pieces=5
    # )  # You can add your model's arguments here, the best would be set up the right ones as default.
    
    # # The first formulation with errors minimization 
    # model.fit(X, Y) 
    
    # # Uncoment To run the fit with the second problem formulation (the one that minimizes the sum of the binary vector)
    
    # # model.fit_beta(X, Y)  # The second formulation

    # # %Pairs Explained
    # pairs_explained = metrics.PairsExplained()
    # print("Percentage of explained preferences:", pairs_explained.from_model(model, X, Y))

    # # %Cluster Intersection
    # cluster_intersection = metrics.ClusterIntersection()

    # Z = data_loader.get_ground_truth_labels()
    # print("% of pairs well grouped together by the model:")
    # print("Cluster intersection for all samples:", cluster_intersection.from_model(model, X, Y, Z))
    #===========================================================================================================================

    ### 2nd part: test of the heuristic model
    data_loader = Dataloader("data/dataset_10")  # Path to test dataset
    X, Y = data_loader.load()

    Res_explanation={'negative_mutation':[], 'positive_mutation':[], 'instances_limit':[]}
    Res_intersection={'negative_mutation':[], 'positive_mutation':[], 'instances_limit':[]}

    for k in ['negative_mutation', 'positive_mutation']:
        hyper_params={'negative_mutation':0.8, 'positive_mutation':0.8, 'instances_limit':1000}
        for i in list(np.arange(0.5, 0.99, 0.05))+[0.98]:
            hyper_params[k]=round(i, 2)
            l_a, l_b=[], []
            for j in range(10):
                model = HeuristicModel(n_clusters=4, n_instances=X.shape[0], n_features=X.shape[1], n_pieces=5, n_iter=5, hyper_params=hyper_params)
                model.fit(X, Y)

                Z = data_loader.get_ground_truth_labels()
                # Validation on test set
                # %Pairs Explained
                pairs_explained = metrics.PairsExplained()
                a = pairs_explained.from_model(model, X, Y)
                print("Percentage of explained preferences:", a)

                # %Cluster Intersection
                cluster_intersection = metrics.ClusterIntersection()
                b = cluster_intersection.from_model(model, X, Y, Z)
                print("% of pairs well grouped together by the model:")
                print( "Cluster intersection for all samples:", b)
                l_a.append(a)
                l_b.append(b)

            Res_explanation[k].append(np.mean(l_a))
            Res_intersection[k].append(np.mean(l_b))



        hyper_params={'negative_mutation':0.8, 'positive_mutation':0.8, 'instances_limit':1000}
        for i in np.arange(200, 2000, 100):
            hyper_params['instances_limit']=i
            l_a, l_b=[], []
            for j in range(10):
                model = HeuristicModel(n_clusters=4, n_instances=X.shape[0], n_features=X.shape[1], n_pieces=5, n_iter=5, hyper_params=hyper_params)
                model.fit(X, Y)

                Z = data_loader.get_ground_truth_labels()
                # Validation on test set
                # %Pairs Explained
                pairs_explained = metrics.PairsExplained()
                a = pairs_explained.from_model(model, X, Y)
                print("Percentage of explained preferences:", a)

                # %Cluster Intersection
                cluster_intersection = metrics.ClusterIntersection()
                b = cluster_intersection.from_model(model, X, Y, Z)
                print("% of pairs well grouped together by the model:")
                print( "Cluster intersection for all samples:", b)
                l_a.append(a)
                l_b.append(b)
                
            Res_explanation['instances_limit'].append(np.mean(l_a))
            Res_intersection['instances_limit'].append(np.mean(l_b))


