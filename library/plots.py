import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

def plot_svm(X: npt.NDArray, y: npt.NDArray, support_vectors: npt.NDArray):
	# project points on the 2 first principal components
	pca = PCA(n_components=2)
	X_pca = pca.fit_transform(X)
	support_vectors_pca = pca.fit_transform(support_vectors)

	fig, ax = plt.subplots(1, figsize = (12, 7))

	options = {
		0: {
			"color": "royalblue",
			"label": "Healty",
			"marker": "o",
			"alpha": 0.7
		},
		1: {
			"color": "orange",
			"label": "Parkinson",
			"marker": "o",
			"alpha": 0.7
		}
	}

	for label in np.unique(y.tolist()):
		idx = np.where(y.tolist() == label)

		# plot points (that are not support vectors)
		ax.scatter(X_pca[:,0][idx], X_pca[:,1][idx], c=options[idx]["color"], label=options[idx]["label"], alpha=options[idx]["alpha"])

	# plot support vectors
	ax.scatter(support_vectors_pca[:,0], support_vectors_pca[:,1], label="support vectors")

	ax.legend(loc="best")
	ax.set_xlabel("principal component 1")
	ax.set_ylabel("principal component 2")

	plt.show()