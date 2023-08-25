import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import itertools


def create_data(x_training, variance=0.01, degree: int = 4):
    np.random.seed(42)

    X = x_training[:, np.newaxis]
    y = X**2 - X + np.random.normal(0, variance, X.shape)
    y_training = y.squeeze()

    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)

    x_interval = np.linspace(-1.0, 1.0, 1000)
    y_interval = np.linspace(-3.0, 3.0, 1000)

    return X_poly, y, x_interval, y_interval, y_training


def create_test_data(x_interval, degree: int = 4):
    poly_features = PolynomialFeatures(degree=degree)
    X_pred = x_interval.reshape(-1, 1)
    X_pred_poly = poly_features.fit_transform(X_pred)
    return X_pred_poly


def fit_linear_regression(X_poly, y):
    theta = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y
    return theta


def calc_gaussian_probabilities(mean, variance, y_interval):
    sigma = np.sqrt(variance)
    probabilities = (
        1
        / (sigma * np.sqrt(2 * np.pi))
        * np.exp(-((y_interval - mean) ** 2) / (2 * variance))
    )
    return probabilities


def get_pnml_normalization_factor(X_pred_poly, X_poly, x_inv=None):
    if x_inv is None:
        x_inv = np.linalg.inv(X_poly.T @ X_poly)
    return 1 + np.diag(X_pred_poly @ x_inv @ X_pred_poly.T)


def calc_subspace_prediction(X_pred_poly, X_poly, y, U, s, subspace):
    if len(subspace) > 1:
        s_mat_inv = np.diag(1 / s[[subspace]].squeeze())
    else:
        s_mat_inv = np.diag(1 / s[subspace])
    U_subspace = U[:, subspace]

    x_inv_subspace = U_subspace @ s_mat_inv @ U_subspace.T
    theta_subspace = x_inv_subspace @ X_poly.T @ y
    y_pred_svd = X_pred_poly @ theta_subspace
    return y_pred_svd, x_inv_subspace


def create_subspaces_all_permutations(num_subspaces):
    combs = list(
        list(itertools.combinations(range(num_subspaces), num_comb))
        for num_comb in range(1, num_subspaces)
    )
    combs = list(itertools.chain.from_iterable(combs))
    combs.append([i for i in range(num_subspaces)])
    t_combs = []
    for comb in combs:
        if len(comb) == 1:
            t_combs.append([comb[0]])
        else:
            t_combs.append(comb)
    return t_combs


def create_subspaces_permutations(num_subspaces, hierarchies):
    combs = list(itertools.combinations(range(num_subspaces), hierarchies))
    combs = list(itertools.chain.from_iterable(combs))
    combs.append([i for i in range(num_subspaces)])
    t_combs = []
    for comb in combs:
        if len(comb) == 1:
            t_combs.append([comb[0]])
        else:
            t_combs.append(comb)
    return t_combs
