import numpy as np

def pca(X, n_components):
    X_normalized = X - np.mean(X, axis=0)
    covariance_matrix = np.cov(X_normalized, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    idx = np.argsort(eigenvalues)[::-1][:n_components]
    principal_components = eigenvectors[:, idx]
    X_projected = X_normalized @ principal_components
    return X_projected, principal_components