import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
    
    def _euclidean_distance(self, x1, x2):
        """
        Compute the Euclidean distance between two vectors

        Parameters:
            x1 (array-like): A vector in the feature space.
            x2 (array-like): A vector in the feature space.
            
        Returns
        float - Euclidean distance between x1 and x2
        """
        return np.sqrt(np.sum((x1-x2)**2))
    
    def _manhattan_distance(self, x1, x2):
        """
        Compute the Manhattan distance between two vectors

        Parameters:
            x1 (array-like): A vector in the feature space.
            x2 (array-like): A vector in the feature space.
            
        Returns
        float - Manhattan distance between x1 and x2
        """
        return np.sum(np.abs(x1-x2))
    
    def _minkowski_distance(self, x1, x2):
        """
        Compute the Minkowski distance between two vectors

        Parameters:
            x1 (array-like): A vector in the feature space.
            x2 (array-like): A vector in the feature space.
            
        Returns
        float - Minkowski distance between x1 and x2
        """
        return np.sum(np.abs(x1-x2)**self.k) ** (1/self.k)
    
    def fit(self, x, y):
        """
        Fit the model using X as training data and y as target values

        Args:
            x (array-like): Training data
            y (array-like): Target values
        """
        
        self.X_train = x
        self.y_train = y
        
    def predict(self, X):
        """
        Predict the class labels for the provided data

        Args:
            X (array-like): Data to predict the class labels

        Returns
        array-like - Predicted class labels
        """
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
    
    def _predict(self, x):
        """
        Predict the class label for a single data point

        Args:
            x (array-like): Data point to predict the class label

        Returns
        int - Predicted class label
        """
        if self.distance_metric == 'euclidean':
            distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        elif self.distance_metric == 'manhattan':
            distances = [self._manhattan_distance(x, x_train) for x_train in self.X_train]
        elif self.distance_metric == 'minkowski':
            distances = [self._minkowski_distance(x, x_train) for x_train in self.X_train]
        else:
            raise ValueError("Invalid distance metric. Choose from Euclidean, Manhattan or Minkowski")
        
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)

        return most_common[0][0]
    
