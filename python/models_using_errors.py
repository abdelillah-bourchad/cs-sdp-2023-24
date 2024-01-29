import pickle
from abc import abstractmethod

import numpy as np
from gurobipy import Model, GRB, quicksum, max_



class BaseModel(object):
    """
    Base class for models, to be used as coding pattern skeleton.
    Can be used for a model on a single cluster or on multiple clusters"""

    def __init__(self):
        """Initialization of your model and its hyper-parameters"""
        pass

    @abstractmethod
    def fit(self, X, Y):
        """Fit function to find the parameters according to (X, Y) data.
        (X, Y) formatting must be so that X[i] is preferred to Y[i] for all i.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        # Customize what happens in the fit function
        return

    @abstractmethod
    def predict_utility(self, X):
        """Method to call the decision function of your model

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        # Customize what happens in the predict utility function
        return

    def predict_preference(self, X, Y):
        """Method to predict which pair is preferred between X[i] and Y[i] for all i.
        Returns a preference for each cluster.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of preferences for each cluster. 1 if X is preferred to Y, 0 otherwise
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return (X_u - Y_u > 0).astype(int)

    def predict_cluster(self, X, Y):
        """Predict which cluster prefers X over Y THE MOST, meaning that if several cluster prefer X over Y, it will
        be assigned to the cluster showing the highest utility difference). The reversal is True if none of the clusters
        prefer X over Y.
        Compared to predict_preference, it indicates a cluster index.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, ) index of cluster with highest preference difference between X and Y.
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return np.argmax(X_u - Y_u, axis=1)

    def save_model(self, path):
        """Save the model in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the file in which the model will be saved
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(clf, path):
        """Load a model saved in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the path to the file to load
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model


class RandomExampleModel(BaseModel):
    """Example of a model on two clusters, drawing random coefficients.
    You can use it to understand how to write your own model and the data format that we are waiting for.
    This model does not work well but you should have the same data formatting with TwoClustersMIP.
    """

    def __init__(self):
        self.seed = 444
        self.weights = self.instantiate()

    def instantiate(self):
        """No particular instantiation"""
        return []

    def fit(self, X, Y):
        """fit function, sets random weights for each cluster. Totally independant from X & Y.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        np.random.seed(self.seed)
        indexes = np.random.randint(0, 2, (len(X)))
        num_features = X.shape[1]
        weights_1 = np.random.rand(num_features)
        weights_2 = np.random.rand(num_features)

        weights_1 = weights_1 / np.sum(weights_1)
        weights_2 = weights_2 / np.sum(weights_2)
        self.weights = [weights_1, weights_2]
        return self

    def predict_utility(self, X):
        """Simple utility function from random weights.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        return np.stack([np.dot(X, self.weights[0]), np.dot(X, self.weights[1])], axis=1)


class TwoClustersMIP(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self, n_pieces, n_clusters, epsilon):
        """Initialization of the MIP Variables

        Parameters
        ----------
        n_pieces: int
            Number of pieces for the utility function of each feature.
        n°clusters: int
            Number of clusters to implement in the MIP.
        """
        self.seed = 123
        self.L = n_pieces
        self.K = n_clusters
        self.epsilon = epsilon
        self.model = self.instantiate()

    def instantiate(self):
        """Instantiation of the MIP Variables - To be completed."""
        np.random.seed(self.seed)
        model = Model("TwoClustersMIP")
        return model
    
   

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        self.n = X.shape[1]
        self.P = X.shape[0]
       

        def get_last_segment_index( x, i):
            upper_bounds = np.ones(self.n)
            lower_bounds = np.zeros(self.n)
            segments = np.linspace(lower_bounds[i], upper_bounds[i], self.L + 1)
            last_segment_index = np.argmax(x < segments) - 1
            return last_segment_index

        
        def get_breakpoint( i, l):
            upper_bounds = np.ones(self.n)
            lower_bounds = np.zeros(self.n)
            segments = np.linspace(lower_bounds[i], upper_bounds[i], self.L + 1)
            breakpoint = segments[-1] if l >= len(segments) else segments[l]
            return breakpoint
        
        def plot_utility_functions(U):
            import matplotlib.pyplot as plt
            for k in range(self.K):
                for i in range(self.n):
                    plt.plot([get_breakpoint(i, l) for l in range(self.L+1)], [U[k, i, l] for l in range(self.L+1)])
                plt.legend(["feature {}".format(i) for i in range(self.n)])
                plt.show()

        # Variables
        ## Utility functions
        self.U = {
            (cluster, feature, segment): self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, name="util_{}_{}_{}".format(cluster, feature, segment), ub=1)
                for cluster in range(self.K)
                for feature in range(self.n)
                for segment in range(self.L+1)
        }
        ## Error in overestimation and underestimation
        # Overestimation error
        self.sigma_x_plus = {
            (j): self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, name="sigma_x_plus_{}".format(j), ub=1)
                for j in range(self.P)
        }
        self.sigma_y_plus = {
            (j): self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, name="sigma_y_plus_{}".format(j), ub=1)
                for j in range(self.P)
        }


        # erreur de sous-estimation
        self.sigma_x_moins = {
            (j): self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, name="sigma_x_moins_{}".format(j), ub=1)
                for j in range(self.P)
        }
        self.sigma_y_moins = {
            (j): self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, name="sigma_y_moins_{}".format(j), ub=1)
                for j in range(self.P)
        }

        self.a1 = {
            (k, j): self.model.addVar(
                vtype=GRB.BINARY, name="a1_{}_{}".format(k, j))
                for k in range(self.K)
                for j in range(self.P)
        } # 1 if X is preferred to Y for cluster k, 0 otherwise

        M = 100
        uik_xij = {}
        for k in range(self.K):
            for i in range(self.n):
                for j in range(self.P):
                    l = get_last_segment_index(X[j, i], i)
                   
                    bp = get_breakpoint(i, l)
                    bp1 = get_breakpoint(i, l+1)
                    uik_xij[k, i, j] = self.U[(k, i, l)] + ((X[j, i] - bp) / (bp1 - bp)) * (self.U[(k, i, l+1)] - self.U[(k, i, l)])
        
        uik_yij = {}
        for k in range(self.K):
            for i in range(self.n):
                for j in range(self.P):
                    l = get_last_segment_index(Y[j, i], i)
                    bp = get_breakpoint(i, l)
                    bp1 = get_breakpoint(i, l+1)
                    uik_yij[k, i, j] = self.U[(k, i, l)] + ((Y[j, i] - bp) / (bp1 - bp)) * (self.U[(k, i, l+1)] - self.U[(k, i, l)])
        
        uk_xj = {}
        for k in range(self.K):
            for j in range(self.P):
                uk_xj[k, j] = quicksum(uik_xij[k, i, j] for i in range(self.n))
        
        uk_yj = {}
        for k in range(self.K):
            for j in range(self.P):
                uk_yj[k, j] = quicksum(uik_yij[k, i, j] for i in range(self.n))


        # Constraints
        
        # Constraint ensuring that the difference between the utility of X and Y is greater than or equal to a specified minimum
        self.model.addConstrs(
            (uk_xj[k, j] - self.sigma_x_plus[j] + self.sigma_x_moins[j] - uk_yj[k, j] + self.sigma_y_plus[j] - self.sigma_y_moins[j] - self.epsilon >= -M*(1-self.a1[(k,j)]) for j in range(self.P) for k in range(self.K))
        )

    
        # Constraint ensuring that the difference between the utility of X and Y is less than or equal to a specified maximum
        self.model.addConstrs(
            (uk_xj[k, j] - self.sigma_x_plus[j] + self.sigma_x_moins[j] - uk_yj[k, j] + self.sigma_y_plus[j] - self.sigma_y_moins[j] - self.epsilon <= M*self.a1[(k, j)] - self.epsilon
            for j in range(self.P) for k in range(self.K))
        )

        
        # Constraint ensuring at least one cluster is chosen for each sample
        for j in range(self.P):
            self.model.addConstr(
                quicksum(self.a1[(k, j)] for k in range(self.K)) >= 1
            )
    


        ## Monothonicity : 
        self.model.addConstrs(
            (self.U[(k, i, l+1)] - self.U[(k, i, l)]>=self.epsilon for k in range(self.K) for i in range(self.n) for l in range(self.L)))
        ### total score is one, start of each score is 0
        self.model.addConstrs(
            (self.U[(k, i, 0)] == 0 for k in range(self.K) for i in range(self.n)))
        self.model.addConstrs(
            (quicksum(self.U[(k, i, self.L)] for i in range(self.n)) == 1 for k in range(self.K)))
        
        # Objective
        self.model.setObjective(quicksum((self.sigma_x_plus[j] + self.sigma_x_moins[j] + self.sigma_y_plus[j] + self.sigma_y_moins[j]) for j in range(self.P)), GRB.MINIMIZE)




        
        # Solve
        self.model.params.outputflag = 0
        self.model.update()
        self.model.optimize()
        if self.model.status == GRB.INFEASIBLE:
            print("\n Pas de solution")
            raise Exception("Infeasible")
        elif self.model.status == GRB.UNBOUNDED:
            print("\n Le programme n'est pas borné :(")
            raise Exception("Unbounded")
        else:
            print(" On a trouvé une solution optimale!")
            # print the x of objective function
            print("objective function x: ", self.model.objVal)
            self.U = {(k, i, l): self.U[k, i, l].x for k in range(self.K) for i in range(self.n) for l in range(self.L+1)}
            self.a1 = {(k, j): self.a1[k, j].x for k in range(self.K) for j in range(self.P)}
        
            # Call the plot_utility_functions function
            plot_utility_functions(self.U)
        return self

    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        maxs = np.ones(self.n)
        mins = np.zeros(self.n)

        def get_breakpoint(i, l):
            segments = np.linspace(mins[i], maxs[i], self.L + 1)
            if l >= len(segments):
                return segments[-1]
            return segments[l]

        def get_last_segment_index(x, i):
            segments = np.linspace(mins[i], maxs[i], self.L + 1)
            last_index = np.argmax(x < segments) - 1
            return last_index
        
        utilities = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            for j in range(X.shape[0]):
                for i in range(self.n):
                    l = get_last_segment_index(X[j, i], i)
                    utilities[j, k] += self.U[k, i,l] + ((X[j, i] - get_breakpoint(i, l+1) - self.U[k, i,l]))
        return utilities


class HeuristicModel(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self):
        """Initialization of the Heuristic Model.
        """
        self.seed = 123
        self.models = self.instantiate()

    def instantiate(self):
        """Instantiation of the MIP Variables"""
        # To be completed
        return

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        # To be completed
        return

    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        # To be completed
        # Do not forget that this method is called in predict_preference (line 42) and therefor should return well-organized data for it to work.
        return