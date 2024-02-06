import os
import sys
import matplotlib.pyplot as plt

sys.path.append("python/")

import metrics
import numpy as np
from data import Dataloader
# from models_using_errors import HeuristicModel, TwoClustersMIP
from models import TwoClustersMIP


if __name__ == "__main__":
    ### First part: test of the MIP model
    data_loader = Dataloader("data/dataset_4")  # Path to test dataset
    X, Y = data_loader.load()


    model = TwoClustersMIP(
        n_clusters=2, n_pieces=5
    )  # You can add your model's arguments here, the best would be set up the right ones as default.
    
    def flip_values_unique(X, Y, n):
        """
        Introduce noise into the dataset by flipping `n` unique pairs of values between two numpy arrays, `X` and `Y`.
        This function is designed to simulate data variation or introduce noise to test the robustness of a model by 
        exchanging a specified number of values between the preferred (X) and unchosen (Y) feature sets. 

        Parameters
        ----------
        X : np.ndarray
            A numpy array of shape (n_samples, n_features), representing the features of preferred elements.
        Y : np.ndarray
            A numpy array of shape (n_samples, n_features), representing the features of unchosen elements.
        n : int
            The number of unique value pairs to be flipped between `X` and `Y`. This count must not exceed the 
            total number of elements in either array.
        """
        # Générer n indices aléatoires uniques

        idx = np.random.choice(X.shape[0], size=n, replace=False)
        
        # Pour chaque indice, échanger (flipper) les éléments correspondants entre X et Y
        for i in idx:
            temp = X[i].copy()  # Copier pour éviter la modification en place
            X[i], Y[i] = Y[i], temp
        
        return X, Y

    L_Percentage_of_explained_preferences = []


    # for n in range(5,35,5):
    n=5 #---------> à suprimer si boucle
    X, Y = flip_values_unique(X, Y, int(X.shape[0]*n/100))
    # The first formulation with errors minimization 
    model.fit(X, Y) 
    
    # Uncoment To run the fit with the second problem formulation (the one that minimizes the sum of the binary vector)
    
    # model.fit_beta(X, Y)  # The second formulation

    # %Pairs Explained
    pairs_explained = metrics.PairsExplained()
    k = pairs_explained.from_model(model, X, Y)
    print("Percentage of explained preferences:", k)
    L_Percentage_of_explained_preferences.append(k)

    # %Cluster Intersection
    cluster_intersection = metrics.ClusterIntersection()

    Z = data_loader.get_ground_truth_labels()
    print("% of pairs well grouped together by the model:")
    print("Cluster intersection for all samples:", cluster_intersection.from_model(model, X, Y, Z))

    
    print(L_Percentage_of_explained_preferences)

    # Créer un graphique
    axe =range(5,30,5)
    plt.plot(axe, L_Percentage_of_explained_preferences, marker='o')

    plt.title('Percentage of explained preferences by noise pourcentage')
    plt.xlabel('pourcentage of noise')
    plt.ylabel('Preferences')


    plt.show()