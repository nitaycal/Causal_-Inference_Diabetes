import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score


def propensity(X, t, fitted_best_model=None):
    if fitted_best_model is not None:
        classifier = fitted_best_model
    else:
        classifier = LogisticRegression(max_iter=10000)
        classifier.fit(X, t)
    acc = balanced_accuracy_score(t, classifier.predict(X))
    e = classifier.predict_proba(X)[:, classifier.classes_.tolist().index(1)]
    return e, acc


def ipw(t, y, e):
    indices = t == 1
    t1, y1, e1 = t[indices], y[indices], e[indices]
    t0, y0, e0 = t[~indices], y[~indices], e[~indices]
    ate0 = (1 / np.sum((1 - t0) / (1 - e0))) * np.sum((y0 * (1 - t0)) / (1 - e0))
    ate1 = (1 / np.sum(t1 / e1)) * np.sum((y1 * t1) / e1)
    ate = ate1 - ate0
    return ate1, ate0, ate


def s_learner(X, t, y, fitted_best_model=None):
    Xt = pd.concat([X, t], axis=1)
    Xt0, Xt1 = X.copy(), X.copy()
    Xt0[t.name] = 0
    Xt1[t.name] = 1
    if fitted_best_model is not None:
        classifier = fitted_best_model
    else:
        classifier = LogisticRegression(max_iter=10000)
        classifier.fit(Xt, y)
    acc = balanced_accuracy_score(y, classifier.predict(Xt))
    y0_hat = classifier.predict(Xt0)
    y1_hat = classifier.predict(Xt1)
    ate0 = np.mean(y0_hat)
    ate1 = np.mean(y1_hat)
    ate = ate1 - ate0
    ate0_star = (np.sum(y0_hat[t == 1]) + np.sum(y[t == 0])) / (len(y))
    ate1_star = (np.sum(y1_hat[t == 0]) + np.sum(y[t == 1])) / (len(y))
    ate_star = ate1_star - ate0_star
    return ate1, ate0, ate, ate1_star, ate0_star, ate_star, acc


def t_learner(X, t, y, fitted_best_model0=None, fitted_best_model1=None):
    if fitted_best_model0 is not None:
        classifier0 = fitted_best_model0
    else:
        classifier0 = LogisticRegression(max_iter=10000).fit(X[t == 0], y[t == 0])
    if fitted_best_model1 is not None:
        classifier1 = fitted_best_model1
    else:
        classifier1 = LogisticRegression(max_iter=10000).fit(X[t == 1], y[t == 1])
    acc0 = balanced_accuracy_score(y[t == 0], classifier0.predict(X[t == 0]))
    acc1 = balanced_accuracy_score(y[t == 1], classifier1.predict(X[t == 1]))
    y0_hat = classifier0.predict(X)
    y1_hat = classifier1.predict(X)
    ate0 = np.mean(y0_hat)
    ate1 = np.mean(y1_hat)
    ate = ate1 - ate0
    ate0_star = (np.sum(y0_hat[t == 1]) + np.sum(y[t == 0])) / (len(y))
    ate1_star = (np.sum(y1_hat[t == 0]) + np.sum(y[t == 1])) / (len(y))
    ate_star = ate1_star - ate0_star
    return ate1, ate0, ate, ate1_star, ate0_star, ate_star, acc0, acc1


def matching(X, t, y, k=1, fitted_nca_model=None):
    if fitted_nca_model is not None:
        X_normed = fitted_nca_model.transform(X)
    else:
        scaler = sklearn.preprocessing.StandardScaler()
        X_normed = scaler.fit_transform(X)
    knn0 = KNeighborsClassifier(k).fit(X_normed[t == 0], y[t == 0])
    knn1 = KNeighborsClassifier(k).fit(X_normed[t == 1], y[t == 1])
    acc0 = balanced_accuracy_score(y[t == 0], knn0.predict(X_normed[t == 0]))
    acc1 = balanced_accuracy_score(y[t == 1], knn1.predict(X_normed[t == 1]))
    matches0 = knn1.predict(X_normed[t == 0])
    matches1 = knn0.predict(X_normed[t == 1])
    ate0 = (np.sum(matches1) + np.sum(y[t == 0])) / (len(y))
    ate1 = (np.sum(matches0) + np.sum(y[t == 1])) / (len(y))
    ate = ate1 - ate0
    return ate1, ate0, ate, acc0, acc1
