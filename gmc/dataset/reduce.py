import numpy as np

def pca(x):
	x_mean = np.mean(x, axis=0)
	x = np.subtract(x, x_mean)
	sigma = np.matmul(x.T, x)
	sigma = np.divide(sigma, x.shape[0])
	U, S, V = np.linalg.svd(sigma)
	S_cum = np.cumsum(S)
	S_ = np.divide(S_cum, S_cum[-1])
	k=1
	for s in S_:
		if s > 0.95:
			break
		k += 1
	U_ = U[:, :k]
	Z = np.matmul(U_.T,x.T)

	return Z