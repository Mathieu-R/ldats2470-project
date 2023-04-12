import numpy as np 
import numpy.typing as npt

from enum import Enum

class KernelType(Enum):
	LINEAR = "linear"
	RBF = "rbf"

class SVM_SGA:
	def __init__(self, kernel: str = "linear", C: float = 1.0, gamma: float = 1.0, learning_rate: float = 10e-3,  epochs: int = 1000) -> None:
		self.C = C
		self.gamma =gamma

		self.learning_rate = learning_rate
		self.epochs = epochs
		self.tol = 10e-6
		self.epsilon = 10e-8

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
		print(X)
		print(y)
		# n observations, p variables
		n, p = X.shape

		# add 1 at the end of each row
		X = np.hstack((X, np.ones((len(X), 1))))

		self.X = X 
		self.y = y

		# ensure target class \in {-1, 1}
		target = np.where(y <= 0, -1, 1)

		# self.w = np.zeros(n)
		# self.b = 0

		# # at each iteration
		# for _ in range(self.epochs):
		# 	# check if each observation is outside the margin
		# 	for (i, Xi) in enumerate(X):
		# 		yi = target[i]
		# 		h = self.h(Xi)
		# 		constraint = (yi * h) >= 1
				
		# 		# update with gradient descent
		# 		self.update_params(yi, Xi, constraint)

		# compute kernel matrix
		K = self.compute_kernel_matrix(X1=X, X2=X, gamma=self.gamma)

		# compute eta vector
		self.etas = np.zeros((n))

		for k in range(0, n):
			self.etas[k] = 1 / K[k, k]

		# initialize Lagrange multipliers
		self.alphas = np.zeros((n))
		self.bias = 0

		t = 0

		for _ in range(self.epochs):
			alpha = self.alphas[t]
			# for each observation
			for k in range(0, n):
				self.alphas[k] = self.alphas[k] + self.etas[k] * (1 - (y[k] * np.sum(self.alphas * y * K[:, k])))

				# ensure 0 <= alpha_k <= C
				if (self.alphas[k] < 0):
					self.alphas[k] = 0
				elif self.alphas[k] > self.C:
					self.alphas[k] = self.C

			self.alphas[t + 1] = alpha
			t += 1

			if (self.alphas[t] - self.alphas[t - 1]) <= self.tol:
				break

		#self.compute_params(K)
		self.compute_support_vectors()

	def compute_params(self, K):
		self.b = self.y - np.sum(self.alphas * self.y * K, axis=0)
		self.w = np.sum(self.alphas * self.y * K, axis=0)

	def compute_support_vectors(self):
		support_vectors_idx = self.alphas > self.epsilon
		self.support_vectors = self.X[support_vectors_idx]
				
	def predict(self, X):
		K = self.compute_kernel_matrix(X1=self.X, X2=X, gamma=self.gamma)
		return np.sign(np.sum(self.alphas * self.y * K, axis=0) + self.b)