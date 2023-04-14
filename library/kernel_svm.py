import numpy as np 
import numpy.typing as npt

from enum import Enum

class KernelType(Enum):
	LINEAR = "linear"
	RBF = "rbf"

class SVM_SGA:
	def __init__(self, kernel: str = "linear", C: float = 1.0, gamma: float = 1.0, learning_rate: float = 10e-3,  epochs: int = 10000) -> None:
		self.C = C
		self.gamma =gamma

		self.learning_rate = learning_rate
		self.epochs = epochs
		self.tol = 1e-8
		self.epsilon = 1e-8

		self.set_kernel(kernel)

	def linear_kernel(self, X_i, X_j):
		return np.dot(X_i.T, X_j)

	def rbf_kernel(self, X_i, X_j, gamma):
		return np.exp((- np.linalg.norm(X_i - X_j) ** 2) * gamma)

	def set_kernel(self, kernel: str):
		if kernel == KernelType.LINEAR.value:
			self.kernel = self.linear_kernel
		elif kernel == KernelType.RBF.value:
			self.kernel = self.rbf_kernel
		else:
			raise ValueError("Unsupported kernel.")

	def compute_kernel_matrix(self, X1, X2, gamma):
		n, p = X1.shape
		m, p = X2.shape

		K = np.zeros((n, m))

		for i in range(0, n):
			for j in range(0, m):
				K[i, j] = self.rbf_kernel(X1[i,:], X2[j,:], gamma)

		return K

	def fit(self, X, y):
		# n observations, p variables
		n, p = X.shape

		# add 1 at the end of each row
		X = np.hstack((X, np.ones((len(X), 1))))

		self.X = X 
		# ensure target class \in {-1, 1}
		self.y = np.where(y <= 0, -1, 1)

		# compute kernel matrix
		K = self.compute_kernel_matrix(X1=X, X2=X, gamma=self.gamma)

		# compute eta vector
		self.etas = np.zeros((n))

		for k in range(0, n):
			self.etas[k] = 1 / K[k, k]

		# initialize Lagrange multipliers
		self.alphas = np.zeros((n))
		self.bias = 0

		for t in range(0, self.epochs):
			current_alphas = self.alphas.copy()
			# update each alpha component (stochastic gradient ascent)
			for k in range(0, n):
				current_alphas[k] = current_alphas[k] + self.etas[k] * (1 - (self.y[k] * np.sum(current_alphas * self.y * K[:, k])))

				# ensure 0 <= alpha_k <= C
				if (current_alphas[k] < 0):
					current_alphas[k] = 0
				elif current_alphas[k] > self.C:
					current_alphas[k] = self.C

			# break if convergence
			if np.linalg.norm(current_alphas - self.alphas) <= self.tol:
				print(f"converged at loop {t}")
				break
			
			# update alphas
			self.alphas = current_alphas

		#self.compute_params(K)
		self.compute_support_vectors()

	def compute_params(self, K):
		self.b = self.y - np.sum(self.alphas * self.y * K, axis=0)
		self.w = np.sum(self.alphas * self.y * K, axis=0)

	def compute_support_vectors(self):
		support_vectors_idx = self.alphas > self.epsilon
		self.support_vectors = self.X[support_vectors_idx][:,:-1]
				
	def predict(self, X):
		K = self.compute_kernel_matrix(X1=self.X, X2=X, gamma=self.gamma)
		return np.sign(np.sum(self.alphas * self.y * K, axis=0) + self.b)