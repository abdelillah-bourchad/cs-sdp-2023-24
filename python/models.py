import pickle
from abc import abstractmethod

import numpy as np
from gurobipy import Model, GRB, quicksum, max_

from IPython.display import clear_output


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
        # print(X_u.shape, Y_u.shape)

        return (X_u - Y_u > 0).astype(int) #, X_u - Y_u

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

    def __init__(self, n_pieces, n_clusters, epsilon=10e-5):
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
        
        print('Fitting the UTA models using the errors formulation')
        
        Min_b, Max_b = 0, 1 
        segments = np.linspace(Min_b, Max_b, self.L + 1)
        
        def get_index(x):
            idx = np.argmax(segments>x)
            return idx-1, segments[idx-1], segments[idx]
        
        ## Variables declaration
        
        # Utility function params
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
        
        # Underestimation error
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

        self.a = {
            (k, j): self.model.addVar(
                vtype=GRB.BINARY, name="a_{}_{}".format(k, j))
                for k in range(self.K)
                for j in range(self.P)
        }
        
        uik_xij = {}
        for k in range(self.K):
            for i in range(self.n):
                for j in range(self.P):
                    l, bp1, bp2 = get_index(X[j, i])
                    uik_xij[k, i, j] = self.U[(k, i, l)] + ((X[j, i] - bp1) / (bp2 - bp1)) * (self.U[(k, i, l+1)] - self.U[(k, i, l)])
        
        uik_yij = {}
        for k in range(self.K):
            for i in range(self.n):
                for j in range(self.P):
                    l, bp1, bp2 = get_index(Y[j, i])
                    uik_yij[k, i, j] = self.U[(k, i, l)] + ((Y[j, i] - bp1) / (bp2 - bp1)) * (self.U[(k, i, l+1)] - self.U[(k, i, l)])
        
        uk_xj = {}
        for k in range(self.K):
            for j in range(self.P):
                uk_xj[k, j] = quicksum(uik_xij[k, i, j] for i in range(self.n))
        
        uk_yj = {}
        for k in range(self.K):
            for j in range(self.P):
                uk_yj[k, j] = quicksum(uik_yij[k, i, j] for i in range(self.n))
                
                
        ## Constraints Definition
        
        M=10
        # Constraint ensuring that the difference between the utility of X and Y is greater than or equal to a specified minimum
        self.model.addConstrs(
            (uk_xj[k, j] - self.sigma_x_plus[j] + self.sigma_x_moins[j] - uk_yj[k, j] + self.sigma_y_plus[j] - self.sigma_y_moins[j] - self.epsilon >= -M*(1-self.a[(k,j)]) for j in range(self.P) for k in range(self.K))
        )

    
        # Constraint ensuring that the difference between the utility of X and Y is less than or equal to a specified maximum
        self.model.addConstrs(
            (uk_xj[k, j] - self.sigma_x_plus[j] + self.sigma_x_moins[j] - uk_yj[k, j] + self.sigma_y_plus[j] - self.sigma_y_moins[j] - self.epsilon <= M*self.a[(k, j)] - self.epsilon
            for j in range(self.P) for k in range(self.K))
        )

        
        # Constraint ensuring at least one cluster is chosen for each sample
        for j in range(self.P):
            self.model.addConstr(
                quicksum(self.a[(k, j)] for k in range(self.K)) >= 1
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
        print("##########")
        print("Training started")
        self.model.params.outputflag = 0
        self.model.update()
        self.model.optimize()
        
        print("minimum objective function x: ", self.model.objVal)
        self.U = {(k, i, l): self.U[k, i, l].x for k in range(self.K) for i in range(self.n) for l in range(self.L+1)}
        self.a = {(k, j): self.a[k, j].x for k in range(self.K) for j in range(self.P)}
        
        return self
    
    def fit_beta(self, X, Y):
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
        
        Min_b, Max_b = 0, 1 
        segments = np.linspace(Min_b, Max_b, self.L + 1)
        
        def get_index(x):
            idx = np.argmax(segments>x)
            return idx-1, segments[idx-1], segments[idx]
        
        ## Variables declaration
        
        # Utility function params
        self.U = {
            (cluster, feature, segment): self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, name="util_{}_{}_{}".format(cluster, feature, segment), ub=1)
                for cluster in range(self.K)
                for feature in range(self.n)
                for segment in range(self.L+1)
        }

        self.a = {
            (k, j): self.model.addVar(
                vtype=GRB.BINARY, name="a_{}_{}".format(k, j))
                for k in range(self.K)
                for j in range(self.P)
        }
        
        uik_xij = {}
        for k in range(self.K):
            for i in range(self.n):
                for j in range(self.P):
                    l, bp1, bp2 = get_index(X[j, i])
                    uik_xij[k, i, j] = self.U[(k, i, l)] + ((X[j, i] - bp1) / (bp2 - bp1)) * (self.U[(k, i, l+1)] - self.U[(k, i, l)])
        
        uik_yij = {}
        for k in range(self.K):
            for i in range(self.n):
                for j in range(self.P):
                    l, bp1, bp2 = get_index(Y[j, i])
                    uik_yij[k, i, j] = self.U[(k, i, l)] + ((Y[j, i] - bp1) / (bp2 - bp1)) * (self.U[(k, i, l+1)] - self.U[(k, i, l)])
        
        uk_xj = {}
        for k in range(self.K):
            for j in range(self.P):
                uk_xj[k, j] = quicksum(uik_xij[k, i, j] for i in range(self.n))
        
        uk_yj = {}
        for k in range(self.K):
            for j in range(self.P):
                uk_yj[k, j] = quicksum(uik_yij[k, i, j] for i in range(self.n))
                
                
        ## Constraints Definition
        
        M=10
        # Constraint ensuring that the difference between the utility of X and Y is greater than or equal to a specified minimum
        self.model.addConstrs(
            (-uk_xj[k, j] + uk_yj[k, j] + M*self.a[(k,j)] >= self.epsilon for j in range(self.P) for k in range(self.K))
        )

        
        # Constraint ensuring at least one cluster is chosen for each sample
        for j in range(self.P):
            self.model.addConstr(
                quicksum(self.a[(k, j)] for k in range(self.K)) >= 1
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
        self.model.setObjective(quicksum(self.a[(k, j)] for k in range(self.K) for j in range(self.P)), GRB.MINIMIZE)

        # Solve
        print("##########")
        print("Training started")
        self.model.params.outputflag = 0
        self.model.update()
        self.model.optimize()
        
        print("minimum objective function x: ", self.model.objVal)
        self.U = {(k, i, l): self.U[k, i, l].x for k in range(self.K) for i in range(self.n) for l in range(self.L+1)}
        self.a = {(k, j): self.a[k, j].x for k in range(self.K) for j in range(self.P)}
        
        return self
    
    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        Min_b, Max_b = 0, 1 
        segments = np.linspace(Min_b, Max_b, self.L + 1)

        def get_index(x):
            idx = np.argmax(segments>x)
            return idx-1, segments[idx-1], segments[idx]
        
        utilities = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            for j in range(X.shape[0]):
                for i in range(self.n):
                    l, bp1, bp2 = get_index(X[j, i])
                    utilities[j, k] += self.U[(k, i, l)] + ((X[j, i] - bp1) / (bp2 - bp1)) * (self.U[(k, i, l+1)] - self.U[(k, i, l)])
        return utilities
    
class HeuristicModel(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self, n_clusters, n_instances, n_features, n_pieces, n_iter, hyper_params):
        """Initialization of the Heuristic Model.
        """
        self.seed = 123
        self.K = n_clusters
        self.P = n_instances
        self.n = n_features
        self.L = n_pieces
        self.n_iter = n_iter
        self.hyper_params = hyper_params
        self.models = self.instantiate()

    def instantiate(self):
        """Instantiation of the MIP Variables"""
        np.random.seed(self.seed)
        models = {k:None for k in range(self.K)}
        return models

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        def init_proba(P, K) : 
            labels = [np.random.beta(0.5, 0.5, P) for k in range(K)]
            return np.array(labels).T

        def fit_uta_proba(X, Y, weights) :
            
            n = X.shape[1]
            P = X.shape[0]
            L = 5
            model = Model("TwoClustersHeuristic")
            
            upper_bounds, lower_bounds = 1, 0 
            segments = np.linspace(lower_bounds, upper_bounds, L + 1)

            def get_last_segment_index(x):
                if x>=segments[-1] : 
                    return L-1, segments[-2], segments[-1]
                last_segment_index = np.argmax(segments>x)
                return last_segment_index-1, segments[last_segment_index-1], segments[last_segment_index]
            
            ### Variables definitions

            # Utlilty function params
            U = {(feature, segment): model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="util_{}_{}".format(feature, segment), ub=1) for feature in range(n) for segment in range(L+1)}

            w = {
                (j): model.addVar(
                    vtype=GRB.CONTINUOUS, name="w_{}".format(j), lb=weights[j], ub=weights[j])
                    for j in range(P)
            }
            
            # Over estimation errors
            sigma_x_plus = {
                (j): model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0, name="sigma_x_plus_{}".format(j), ub=1)
                    for j in range(P)
            }
            sigma_y_plus = {
                (j): model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0, name="sigma_y_plus_{}".format(j), ub=1)
                    for j in range(P)
            }

            # Under estimation errors
            sigma_x_moins = {
                (j): model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0, name="sigma_x_moins_{}".format(j), ub=1)
                    for j in range(P)
            }
            sigma_y_moins = {
                (j): model.addVar(
                    vtype=GRB.CONTINUOUS, lb=0, name="sigma_y_moins_{}".format(j), ub=1)
                    for j in range(P)
            }

            # utility vectors
            u_xij = {}
            for i in range(n):
                for j in range(P):
                    l, bp, bp1 = get_last_segment_index(X[j, i])
                    try : 
                        u_xij[i, j] = U[(i, l)] + ((X[j, i] - bp) / (bp1 - bp)) * (U[(i, l+1)] - U[(i, l)])
                    except : 
                        print(l, bp, bp1, X[j, i])

            u_yij = {}
            for i in range(n):
                for j in range(P):
                    l, bp, bp1 = get_last_segment_index(Y[j, i])
                    try :
                        u_yij[i, j] = U[(i, l)] + ((Y[j, i] - bp) / (bp1 - bp)) * (U[(i, l+1)] - U[(i, l)])
                    except : 
                        print(l, bp, bp1, Y[j, i])
                
            # Utility function for every couple
            u_xj = {}
            for j in range(P):
                u_xj[j] = quicksum(u_xij[i, j] for i in range(n))

            u_yj = {}
            for j in range(P):
                u_yj[j] = quicksum(u_yij[i, j] for i in range(n))
                    

            # Constraints  
            epsilon = 0.00000000001
                    
            # Constraint ensuring that the difference between the utility of X and Y is greater than or equal to a specified minimum
            model.addConstrs(
                (u_xj[j] - sigma_x_plus[j] + sigma_x_moins[j] - u_yj[j] + sigma_y_plus[j] - sigma_y_moins[j] - epsilon >= 0 for j in range(P))
            )
            
            ## Monothonicity : 
            model.addConstrs(
                (U[(i, l+1)] - U[(i, l)]>=epsilon for i in range(n) for l in range(L)))
            ### total score is one, start of each score is 0
            model.addConstrs(
                (U[(i, 0)] == 0 for i in range(n)))
            model.addConstr(
                (quicksum(U[(i, L)] for i in range(n)) == 1))
            
            sigma = {}
            for j in range(P) : 
                sigma[j] = (sigma_x_plus[j] + sigma_x_moins[j] + sigma_y_plus[j] + sigma_y_moins[j])

            # Objective
            model.setObjective(quicksum(w[j]*sigma[j] for j in range(P)), GRB.MINIMIZE) 
            # model.setObjective((w[0]*sigma[0] + w[1]*sigma[1]), GRB.MINIMIZE) 
            
            model.params.outputflag = 0
            model.update()
            model.optimize()
            
            U = {(i, l): U[i, l].x for i in range(n) for l in range(L+1)}
            return U

        def update_proba(p, coef): 
            p_new = p*coef
            if p_new>1 : 
                return 1 
            return p_new

        def predict_util(X, U):
            
            n = X.shape[1]
            P = X.shape[0]
            L = 5
            
            Min_b, Max_b = 0, 1 
            segments = np.linspace(Min_b, Max_b, L + 1)

            def get_index(x):
                if x>=segments[-1] : 
                    return L-1, segments[-2], segments[-1]
                last_segment_index = np.argmax(segments>x)
                return last_segment_index-1, segments[last_segment_index-1], segments[last_segment_index]
            
            utilities = np.zeros(P)
            for j in range(P):
                for i in range(n):
                    l, bp1, bp2 = get_index(X[j, i])
                    utilities[j] += U[(i, l)] + ((X[j, i] - bp1) / (bp2 - bp1)) * (U[(i, l+1)] - U[(i, l)])
                    
            return utilities 

        def estimate_proba(utilities, old_labels) : 
            
            P = utilities[0][0].shape[0]
            
            labels = np.zeros((P, len(utilities)))
            
            for j in range(P) : 
                
                unclassified = True 
                
                for k in range(len(utilities)) : 
                    if utilities[k][0][j]>utilities[k][1][j] : 
                        unclassified = False 
                        prob = np.random.uniform(0, 1)>=self.hyper_params['negative_mutation']
                        if not(prob) : 
                            labels[j, k] = update_proba(old_labels[j, k], 1.2)
                        else : 
                            labels[j, k] = update_proba(old_labels[j, k], 0.5)
                
                if unclassified : 
                    for k in range(len(utilities)) : 
                        if old_labels[j, k] <= 0.5 : 
                            labels[j, k] = update_proba(old_labels[j, k], 1.2)
                        else : 
                            labels[j, k] = update_proba(old_labels[j, k], 0.8)
                
                for k in range(len(utilities)) : 
                    if utilities[k][0][j]<=utilities[k][1][j] : 
                        prob = np.random.uniform(0, 1)>=self.hyper_params['positive_mutation']
                        if prob : labels[j, k] = update_proba(old_labels[j, k], 1.2)
                    
            return labels

        def score(utilities) : 
            
            n = 0
            P = len(utilities[0][0])
            for i in range(P) : 
                for k in range(len(utilities)) :  
                    if utilities[k][0][i]>utilities[k][1][i] : 
                        n += 1
                        break
            return n/P


        it = 0 
        labels = init_proba(self.P, self.K)
        utilities = {k:(None, None) for k in range(self.K)}
        
        Lim = self.hyper_params['instances_limit']
        for k in range(self.K) : 
            args_k = np.argsort(labels[:, k])[-Lim:]
            self.models[k] = fit_uta_proba(X[args_k], Y[args_k], labels[:, k][args_k])
        
        self.history = {'acc':[], 'models':[]}
        sc = 0
        
        while it<self.n_iter : 
            print(f"Iteration {it+1} : ")
            print(f"Explained pairs pourcentage : {sc}")
            
            self.history['acc'].append(sc) 
            
            for k in range(self.K) : 
                args_k = np.argsort(labels[:, k])[-Lim:]
                self.models[k] = fit_uta_proba(X[args_k], Y[args_k], labels[:, k][args_k])
                Ux = predict_util(X, self.models[k])
                Uy = predict_util(Y, self.models[k])
                utilities[k] = (Ux, Uy)
        
            self.history['models'].append(self.models)
            labels = estimate_proba(utilities, labels)
            
            sc = score(utilities)
            clear_output(wait=False)
            it+=1
            
        return self

    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements
        """
        # To be completed
        # Do not forget that this method is called in predict_preference (line 42) and therefor should return well-organized data for it to work.
        def predict_util(X, U): 
            n = X.shape[1]
            P = X.shape[0]
            L = 5
            
            Min_b, Max_b = 0, 1 
            segments = np.linspace(Min_b, Max_b, L + 1)

            def get_index(x):
                if x>=segments[-1] : 
                    return L-1, segments[-2], segments[-1]
                last_segment_index = np.argmax(segments>x)
                return last_segment_index-1, segments[last_segment_index-1], segments[last_segment_index]
            
            utilities = np.zeros(P)
            for j in range(P):
                for i in range(n):
                    l, bp1, bp2 = get_index(X[j, i])
                    utilities[j] += U[(i, l)] + ((X[j, i] - bp1) / (bp2 - bp1)) * (U[(i, l+1)] - U[(i, l)])
                    
            return utilities 
        
        utilities = np.zeros((self.P, self.K))
        for k in range(self.K) : 
            utilities[:, k] = predict_util(X, self.models[k])
        return utilities
    
    
    
    