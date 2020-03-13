from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from metric_learn import NCA
import numpy as np
import time
import pickle


# list of scorings: https://scikit-learn.org/stable/modules/model_evaluation.html#multimetric-scoring


class NCAMModel():
    def __init__(self, n_components=-1, max_iter=100, verobose=1, n_neighbors=-1, n_jobs=None, **kwargs):
        self._nca_kwrags = {'n_components': n_components if n_components > 0 else 2,
                            'max_iter': max_iter, 'verbose': verobose, 'init': 'auto'}
        self._nca_kwrags.update(NCAMModel.get_func_kwargs(NCA.__init__, **kwargs))

        self._knn_kwargs = {'n_neighbors': n_neighbors if n_neighbors > 0 else 5,
                            'n_jobs': n_jobs, 'weights': 'distance'}
        self._knn_kwargs.update(NCAMModel.get_func_kwargs(KNeighborsClassifier.__init__, **kwargs))

        self._nca = NCA(**self._nca_kwrags)
        self._knn = KNeighborsClassifier(**self._knn_kwargs)

        self._find_n_components = False if n_components > 0 else True
        self._find_k = False if n_neighbors > 0 else True
        self._feature_importances_ = None

    @staticmethod
    def list_of_ks(samples):
        n = max(int(samples * 0.05), 20)
        sieve = [True] * n
        for i in range(3, int(n ** 0.5) + 1, 2):
            if sieve[i]:
                sieve[i * i::2 * i] = [False] * ((n - i * i - 1) // (2 * i) + 1)
        primes = [1 ,2] + [i for i in range(3, n, 2) if sieve[i]]
        ks = []
        i, jump = 0, 0
        while i < len(primes):
            if i % 4 == 0:
                jump += 1
            ks.append(primes[i])
            i += jump
        return ks

    @staticmethod
    def get_func_kwargs(func, **kwargs):
        if func is not None:
            return {k: v for k, v in kwargs.items() if k in func.__code__.co_varnames}
        else:
            return {}

    @property
    def find_k(self):
        return self._find_k

    @property
    def knn(self):
        return self._knn

    def transform(self, X):
        return self._nca.transform(X)

    def _fit(self, X, y):
        self._nca.fit(X, y)
        fitted_X = self.transform(X)
        self._feature_importances_ = np.sum(self._nca.components_.T, axis=1)
        self._feature_importances_ = abs(self._feature_importances_ / np.sum(self._feature_importances_))
        self.knn.fit(fitted_X, y)
        if self._find_k:
            self.tune_k(X, y)
        return self

    def fit(self, X, y):
        if self._find_n_components:
            self.tune_n_components(X, y)
            return self
        return self._fit(X, y)

    def predict(self, X):
        fitted_X = self.transform(X)
        return self._knn.predict(fitted_X)

    def predict_proba(self, X):
        fitted_X = self.transform(X)
        return self.knn.predict_proba(fitted_X)

    @property
    def feature_importances_(self):
        return self._feature_importances_

    def tune_k(self, X, y, scoring='accuracy', cv=4, tranform_X=True, tune=True):
        fitted_X = self.transform(X) if tranform_X else X
        best_score, best_k, scores = None, self._knn.n_neighbors, []
        for k in NCAMModel.list_of_ks(fitted_X.shape[0]):
            self._knn.n_neighbors = k
            score = cross_val_score(self._knn, X=fitted_X, y=y, cv=cv,
                                          scoring=scoring, n_jobs=self._knn_kwargs['n_jobs']).mean()
            if best_score is None or score >= best_score:
                best_score, best_k = score, k
            scores.append(score)
        if tune:
            self._knn.n_neighbors = best_k
            self._knn_kwargs['n_neighbors'] = best_k
        return best_k, best_score

    @staticmethod
    def _get_componenets_list_by_pca(X, step=0.1, verbose=False):
        pca = PCA(n_components=0.9)
        pca.fit(X)
        components, bounded_var = [], 0
        for comp_i, e_var in enumerate(pca.explained_variance_ratio_):
            bounded_var += e_var
            if bounded_var > step:
                bounded_var = 0
                components.append(comp_i+1)
        if verbose:
            print('PCA explained variances: ', pca.explained_variance_ratio_)
            print('components: ', components)
        return components

    def tune_n_components(self, X, y, components=None, tune=True, test_size=0):
        v = self._nca_kwrags['verbose']
        start_time = time.time()
        components = components if isinstance(components, list) \
                                else NCAMModel._get_componenets_list_by_pca(X, step=0.1, verbose=v)
        best_score, best_comps = None, self._nca.n_components
        temp_knn = KNeighborsClassifier(**self._knn_kwargs)
        if test_size * len(y) <= 50:
            X_train, y_train = shuffle(X, y)
            X_test, y_test = shuffle(X, y)
            temp_knn.weights = 'uniform'
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        if v:
            print('Tuning n_componenets from ', components)
        for comps in components:
            temp_nca = NCA(**self._nca_kwrags)
            temp_nca.verbose = False
            temp_nca.n_components = comps
            if v:
                current_time = time.time()
                print(f'--- Fitting n_componenets={comps}')
            temp_nca.fit(X, y)
            fitted_X_train, fitted_X_test = temp_nca.transform(X_train), temp_nca.transform(X_test)
            temp_knn.fit(fitted_X_train, y_train)
            scores = []
            for k in NCAMModel.list_of_ks(fitted_X_train.shape[0]):
                temp_knn.n_neighbors = k
                scores.append(temp_knn.score(fitted_X_test, y_test))
            score = np.mean(scores)
            if v:
                print(f'\t\t Fitting time: {round(time.time() - current_time, 2)}s'
                      f' | From start: {round(time.time() - start_time, 2)}s')
                print(f'\t\t Avg. score: {score} | within {len(scores)} different KNNs')
            if best_score is None or score >= best_score:
                best_score, best_comps = score, comps
        if tune:
            if self._nca_kwrags['verbose']:
                current_time = time.time()
                print(f'------------- Fitting NCAModel with best n_componenets={best_comps} -------------')
            self._nca.n_components = best_comps
            self._nca_kwrags['n_components'] = best_comps
            self._find_n_components = False
            self._fit(X, y)
            if v:
                print(f'\t\t Fitting time: {round(time.time() - current_time, 2)}s'
                      f' | From start: {round(time.time() - start_time, 2)}s')
        return best_comps, best_score

    def tune_fit_transform(self, X, y, components=None, test_size=0):
        self.tune_n_components(X, y, components=components, tune=True, test_size=test_size)
        return self.transform(X)

    def save_model(self, fname):
        with open(fname, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_model(fname):
        with open(fname, 'rb') as file:
            return pickle.load(file)
