import numpy as np 
import numpy.typing as npt

from scipy.optimize import minimize

class SoftMarginSVM():
	def __init__(self) -> None:
		pass

	def fit(self, X, y) -> None:
		# n observations, p variables
		self.n, self.p = X.shape
		
		# maximize the lagrangian 
		# equivalent to minimizing the negative lagrangian
		result = minimize(
			fun=lambda alpha: -self.dual_lagrangian(X, y, alpha),
			x0=np.ones(self.n),
			method="SLSQP",
			jac=lambda alpha: -self.dual_lagrangian_jac(X, y, alpha),
			constraints=
		)

		alpha = result.x
		w = np.sum(alpha * X * y)

		support_vectors = X[alpha > 10e-6]
		support_vectors_labels = y[alpha > 10e-6]
		b = support_vectors_labels[0] - np.dot(w.T, support_vectors[0])

	def predict(self, X):
		pass

	def dual_lagrangian(self, X, y, alpha) -> float:
		"""Compute the objective function (i.e. the dual Lagrangian) to maximize.

		Args:
			X (_type_): _description_
			y (_type_): _description_
			alpha (_type_): _description_

		Returns:
			float: _description_
		"""
		# pairwise multiplication of X matrix
		# \sum_{i=1}^{n} \sum_{j=1}^{n} X_i^T * X_j
		XiTXj = np.matmul(X, X.T)

		# pairwise multiplication of target y
		yiyj = y * y.reshape(-1,1)

		return np.sum(alpha) - (1/2) * (alpha.T @ (alpha @ (XiTXj * yiyj)))

	def dual_lagrangian_jac(self, X, y, alpha) -> npt.NDArray:
		"""Compute the jacobian of the objective function with respect to the the Lagrange multipliers alpha.

		Args:
			X (_type_): _description_
			y (_type_): _description_
			alpha (_type_): _description_

		Returns:
			float: _description_
		"""
		# pairwise multiplication of X matrix
		# \sum_{i=1}^{n} \sum_{j=1}^{n} X_i^T * X_j
		XiTXj = np.matmul(X, X.T)

		# pairwise multiplication of target y
		yiyj = y * y.reshape(-1,1)


		return np.ones(self.n) - (X.T @ (XiTXj * yiyj))

	def constraints(self, X, y):
		# \sum_{i=1}^{n} \alpha_i \cdot y_i = 0 
		# <\alpha, y> = 0
		def eq_constraint(alpha, y):
			return np.dot(alpha, y) 

		def eq_constraint_jac(alpha, y):
			return y

		def ineq_constraint(alpha, y):
			pass

		return (
			{"type": "eq", "fun": lambda alpha: eq_constraint(alpha, y), "jac": lambda alpha: eq_constraint_jac(alpha, y)},
			{"type": "ineq", "fun": lambda alpha: np.dot(alpha, y), "jac": lambda alpha: y}
		)
