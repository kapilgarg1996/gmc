import numpy as np

def pca(x, k=None):
    x_mean = np.mean(x, axis=0)
    x = np.subtract(x, x_mean)
    sigma = np.matmul(x.T, x)
    sigma = np.divide(sigma, x.shape[0])
    U, S, V = np.linalg.svd(sigma)
    S_cum = np.cumsum(S)
    S_ = np.divide(S_cum, S_cum[-1])
    if k is None:
        k=1
        for s in S_:
            if s > 0.99:
                break
            k += 1
    U_ = U[:, :k]
    Z = np.matmul(U_.T,x.T)

    return Z.T

def lda(x, y):
    c = []
    for label in y:
        c.append(np.where(label==1)[0][0]+1)
    c = np.array(c)
    S_W, S_B = scatter_within(x, c), scatter_between(x, c)
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    W = get_components(x, eig_vals, eig_vecs, n_comp=2)
    X_lda = x.dot(W)
    return X_lda

def comp_mean_vectors(X, y):
    class_labels = np.unique(y)
    n_classes = class_labels.shape[0]
    mean_vectors = []
    for cl in class_labels:
        mean_vectors.append(np.mean(X[y==cl], axis=0))

    return mean_vectors

def scatter_within(X, y):
    class_labels = np.unique(y)
    n_classes = class_labels.shape[0]
    n_features = X.shape[1]
    mean_vectors = comp_mean_vectors(X, y)
    S_W = np.zeros((n_features, n_features))
    for cl, mv in zip(class_labels, mean_vectors):
        class_sc_mat = np.zeros((n_features, n_features))                 
        for row in X[y == cl]:
            row, mv = row.reshape(n_features, 1), mv.reshape(n_features, 1)
            class_sc_mat += (row-mv).dot((row-mv).T)
        S_W += class_sc_mat                           
    return S_W

def scatter_between(X, y):
    overall_mean = np.mean(X, axis=0)
    n_features = X.shape[1]
    mean_vectors = comp_mean_vectors(X, y)    
    S_B = np.zeros((n_features, n_features))
    for i, mean_vec in enumerate(mean_vectors):  
        n = X[y==i+1,:].shape[0]
        mean_vec = mean_vec.reshape(n_features, 1)
        overall_mean = overall_mean.reshape(n_features, 1)
        S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
    return S_B

def get_components(X, eig_vals, eig_vecs, n_comp=2):
    n_features = X.shape[1]
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    W = np.hstack([eig_pairs[i][1].reshape(n_features, 1) for i in range(0, n_comp)])
    return W