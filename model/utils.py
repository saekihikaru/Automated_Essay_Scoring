import numpy as np
from sklearn.metrics import f1_score


def f1_micro(y_true, y_pred):
    return -f1_score(y_true, y_pred, average="micro", zero_division=0)


def f1_micro_lgb(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    return "f1_micro", f1_score(y_true, y_pred, average="micro", zero_division=0), True

def quadratic_weighted_kappa(y_true, y_pred):
    """
    Calculate the Quadratic Weighted Kappa.
    
    :param y_true: Array of actual ratings
    :param y_pred: Array of predicted ratings
    :return: Quadratic Weighted Kappa score
    """
    min_rating = min(min(y_true), min(y_pred))
    max_rating = max(max(y_true), max(y_pred))
    N = max_rating - min_rating + 1

    O = np.zeros((N, N))
    for a, p in zip(y_true, y_pred):
        O[a - min_rating][p - min_rating] += 1

    w = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            w[i][j] = ((i - j) ** 2) / ((N - 1) ** 2)
    
    act_hist = np.zeros(N)
    for item in y_true:
        act_hist[item - min_rating] += 1

    pred_hist = np.zeros(N)
    for item in y_pred:
        pred_hist[item - min_rating] += 1

    E = np.outer(act_hist, pred_hist)
    E = E / E.sum()

    O = O / O.sum()

    num = np.sum(w * O)
    den = np.sum(w * E)
    
    return 1 - (num / den)