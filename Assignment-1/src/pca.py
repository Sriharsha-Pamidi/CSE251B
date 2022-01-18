import numpy as np


class PCA:
	"""
    This class handles all things related to PCA for PA1.

    You can add any new parameters you need to any functions. This is an 
    outline to help you get started.

    You should run PCA on both the training and validation / testing datasets 
    using the same object.

    For the visualization of the principal components, use the internal 
    parameters that are set in `fit`.
    """
	
	def __init__(self, num_components):
		"""
        Setup the PCA object. 

        Parameters
        ----------
        num_components : int
            The number of principal components to reduce to.
        """
		self.num_components = num_components
	
	def fit(self, X):
		"""
        Set the internal parameters of the PCA object to the data.

        Parameters
        ----------
        X : np.array
            Training data to fit internal parameters.
        """
		self.mean_vector = np.mean(X, axis=0)
		self.covariance_matrix = np.cov(X.T)
		self.eigen_values, self.eigen_vectors = np.linalg.eig(self.covariance_matrix)
		self.pca_indices = np.array(-1 * self.eigen_values)
		self.pca_indices = self.pca_indices.argsort()[:self.num_components]
		self.transform_matrix = self.eigen_vectors[:,self.pca_indices]
		self.transform_sigma = np.sqrt(self.eigen_values[self.pca_indices])
		pass
	
	def transform(self, X):
		"""
        Use the internal parameters set with `fit` to transform data.

        Make sure you are using internal parameters computed during `fit`
        and not recomputing parameters every time!

        Parameters
        ----------
        X : np.array - size n*k
            Data to perform dimensionality reduction on

        Returns
        -------
            Transformed dataset with lower dimensionality
        """
		
		X = X - self.mean_vector
		X = np.dot(X, self.transform_matrix)
		# X = X/self.transform_sigma
		return X
	
	def fit_transform(self, X):
		self.fit(X)
		return self.transform(X)
