import numpy as np 
import numpy.typing as npt

class SVM:
	def __init__(self, learning_rate = 10e-3, C = 100, epochs = 1000) -> None:
		self.learning_rate = learning_rate
		self.C = C
		self.epochs = epochs

	def fit(self, X, y):
		# n observations, p variables
		n, p = X.shape

		# ensure target class \in {-1, 1}
		target = np.where(y <= 0, -1, 1)

		self.w = np.zeros(n)
		self.b = 0

		# at each iteration
		for _ in range(self.epochs):
			# check if each observation is outside the margin
			for (i, Xi) in enumerate(X):
				yi = target[i]
				h = self.h(Xi)
				constraint = (yi * h) >= 1
				
				# update with gradient descent
				self.update_params(yi, Xi, constraint)
				
	
	def h(self, Xi: npt.NDArray):
		return np.dot(self.w.T, Xi) + self.b
	
	def update_params(self, yi: int, Xi: npt.NDArray, constraint: bool):
		if constraint:
			self.w -= self.learning_rate * self.w
		else:
			self.w -= self.learning_rate * (self.w - np.dot(yi, Xi))
			self.b -= self.learning_rate * yi

	def predict(self, X):
		return np.sign(self.h(X))