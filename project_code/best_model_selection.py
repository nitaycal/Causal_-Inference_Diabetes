from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier, XGBRegressor
from metric_learn import NCA, MLKR
import pickle
import numpy as np

# list of scorings: https://scikit-learn.org/stable/modules/model_evaluation.html#multimetric-scoring
# list of objectives : https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters
# Bayesian optimization


class BestModel:
    def __init__(self, model='xgboost', init_points=3, n_iter=2, **init_kwargs):
        if not isinstance(model, str):
            raise Exception("model should be a string")
        self._model_str = model.lower().replace(' ', '')
        if self._model_str in ('best', 'bestmodel', 'bestclassifier', 'classifier'):
            self._init_function = None
            self._optimize_function = None
            self._model_str = 'classifier'
        elif self._model_str in ('bestregressor', 'regressor'):
            self._init_function = None
            self._optimize_function = None
            self._model_str = 'regressor'
        elif self._model_str in ('xgboost', 'xgbclass', 'xgboostclassifier', 'xgbclassifier'):
            self._init_function = self._init_XGBClassifier
            self._optimize_function = self._optimize_XGBClassifier
            self._model_str = 'xgboost'
        elif self._model_str in ('randomforest', 'rfc', 'randomforestclassifier'):
            self._init_function = self._init_RandomForestClassifier
            self._optimize_function = self._optimize_RandomForestClassifier
            self._model_str = 'rfc'
        elif self._model_str in ('xgbreg', 'xgboostregressor', 'xgbregressor'):
            self._init_function = self._init_XGBRegressor
            self._optimize_function = self._optimize_XGBRegressor
            self._model_str = 'xgbreg'
        elif self._model_str in ('rfr', 'randomforestregressor'):
            self._init_function = self._init_RandomForestRegressor
            self._optimize_function = self._optimize_RandomForestRegressor
            self._model_str = 'rfr'
        elif self._model_str in ('lr', 'logisticregression'):
            self._init_function = self._init_LogisticRegression
            self._optimize_function = self._optimize_LogisticRegression
        elif self._model_str in ('svm'):
            self._init_function = self._init_SVM
            self._optimize_function = self._optimize_SVM
            self._model_str = 'lr'
        elif self._model_str in ('linearregression', 'lasso'):
            self._init_function = self._init_Lasso
            self._optimize_function = self._optimize_Lasso
            self._model_str = 'lasso'
        else:
            raise Exception("Model should be one of the following: 'xgboost', 'rcf', 'rfr', 'lr', 'lasso', 'ridge', 'ncaknn', 'mlkrknn'")
        self._init_kwargs = init_kwargs
        self._fitted_model = None
        self._feature_importances_ = None
        self._init_points = init_points
        self._n_iter = n_iter

    def _bayesian_optimization(self, cv_function, parameters):
        gp_params = {"alpha": 1e-5, 'init_points': self._init_points, 'n_iter': self._n_iter}
        bo = BayesianOptimization(cv_function, parameters)
        bo.maximize(**gp_params)
        return bo.max

    def get_init_kwargs(self):
        if self._init_function is not None:
            return {k: v for k, v in self._init_kwargs.items() if k in self._init_function.__code__.co_varnames}
        else:
            return {}

    # -------------- init & optimize functions -------------- #

    # ----- rfc -----

    @staticmethod
    def _init_RandomForestClassifier(params, **kwargs):
        return RandomForestClassifier(
            n_estimators=int(max(params['n_estimators'], 1)),
            max_depth=int(max(params['max_depth'], 1)),
            min_samples_split=int(max(params['min_samples_split'], 2)),
            min_samples_leaf=int(max(params['min_samples_leaf'], 2)),
            n_jobs=-1,
            random_state=42,
            class_weight='balanced')

    @staticmethod
    def _optimize_RandomForestClassifier(X, y, cv_splits=5, scoring='roc_auc', n_jobs=-1, **kwargs):
        def cv_function(n_estimators, max_depth, min_samples_split, min_samples_leaf):
            params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_split': min_samples_split,
                      'min_samples_leaf': min_samples_leaf}
            return cross_val_score(BestModel._init_RandomForestClassifier(params, **kwargs), X=X, y=y, cv=cv_splits,
                                   scoring=scoring, n_jobs=n_jobs).mean()

        parameters = {"n_estimators": (10, 150),
                      "max_depth": (5, 75),
                      "min_samples_split": (2, 100),
                      "min_samples_leaf": (2, 50)}
        return cv_function, parameters

    # ----- rfr -----

    @staticmethod
    def _init_RandomForestRegressor(params, **kwargs):
        return RandomForestRegressor(
            n_estimators=int(max(params['n_estimators'], 1)),
            max_depth=int(max(params['max_depth'], 1)),
            min_samples_split=int(max(params['min_samples_split'], 2)),
            min_samples_leaf=int(max(params['min_samples_leaf'], 2)),
            n_jobs=-1,
            random_state=42)

    @staticmethod
    def _optimize_RandomForestRegressor(X, y, cv_splits=5, scoring='neg_mean_squared_error', n_jobs=-1, **kwargs):
        def cv_function(n_estimators, max_depth, min_samples_split, min_samples_leaf):
            params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_samples_split': min_samples_split,
                      'min_samples_leaf': min_samples_leaf}
            return cross_val_score(BestModel._init_RandomForestRegressor(params, **kwargs), X=X, y=y, cv=cv_splits,
                                   scoring=scoring, n_jobs=n_jobs).mean()

        parameters = {"n_estimators": (10, 500),
                      "max_depth": (5, 50),
                      "min_samples_split": (2, 100),
                      "min_samples_leaf": (2, 50)}
        return cv_function, parameters

    # ----- xgboost classifier -----

    @staticmethod
    def _init_XGBClassifier(params, objective='binary:logistic', **kwargs):
        return XGBClassifier(
            objective=objective,
            learning_rate=max(params['eta'], 0),
            gamma=max(params['gamma'], 0),
            max_depth=int(max(params['max_depth'], 1)),
            n_estimators=int(max(params['n_estimators'], 1)),
            min_child_weight=int(max(params['min_child_weight'], 1)),
            seed=42,
            nthread=-1)

    @staticmethod
    def _optimize_XGBClassifier(X, y, cv_splits=5, scoring='roc_auc', n_jobs=-1, **kwargs):
        def cv_function(eta, gamma, max_depth, n_estimators, min_child_weight):
            params = {'eta': eta, 'gamma': gamma, 'max_depth': max_depth, 'n_estimators': n_estimators,
                      'min_child_weight': min_child_weight}
            return cross_val_score(BestModel._init_XGBClassifier(params, **kwargs), X=X, y=y, cv=cv_splits,
                                   scoring=scoring, n_jobs=n_jobs).mean()

        parameters = {"eta": (0.001, 0.4),
                      "gamma": (0, 15),
                      "max_depth": (1, 75),
                      "n_estimators": (1, 150),
                      "min_child_weight": (1, 20)}
        return cv_function, parameters

    # ----- xgboost regressor ----- #

    @staticmethod
    def _init_XGBRegressor(params, objective='reg:squarederror', **kwargs):
        return XGBRegressor(
            objective=objective,
            learning_rate=max(params['eta'], 0),
            gamma=max(params['gamma'], 0),
            max_depth=int(max(params['max_depth'], 1)),
            n_estimators=int(max(params['n_estimators'], 1)),
            min_child_weight=int(max(params['min_child_weight'], 1)),
            seed=42,
            nthread=-1)

    @staticmethod
    def _optimize_XGBRegressor(X, y, cv_splits=5, scoring='neg_mean_squared_error', n_jobs=-1, **kwargs):
        def cv_function(eta, gamma, max_depth, n_estimators, min_child_weight):
            params = {'eta': eta, 'gamma': gamma, 'max_depth': max_depth, 'n_estimators': n_estimators,
                      'min_child_weight': min_child_weight}
            return cross_val_score(BestModel._init_XGBRegressor(params, **kwargs), X=X, y=y, cv=cv_splits,
                                   scoring=scoring, n_jobs=n_jobs).mean()

        parameters = {"eta": (0.001, 0.4),
                      "gamma": (0, 15),
                      "max_depth": (1, 100),
                      "n_estimators": (1, 500),
                      "min_child_weight": (1, 20)}
        return cv_function, parameters

    # ----- lasso ----- #

    @staticmethod
    def _init_Lasso(params, **kwargs):
        if params['alpha'] < 0.25:
            return LinearRegression(n_jobs=-1)
        else:
            return Lasso(alpha=max(params['alpha'], 0.25))

    @staticmethod
    def _optimize_Lasso(X, y, cv_splits=5, scoring='neg_mean_squared_error', n_jobs=-1, **kwargs):
        def cv_function(alpha):
            params = {'alpha': alpha}
            return cross_val_score(BestModel._init_Lasso(params, **kwargs), X=X, y=y, cv=cv_splits,
                                   scoring=scoring, n_jobs=n_jobs).mean()

        parameters = {"alpha": (0.0, 10)}
        return cv_function, parameters

    # ----- lr ----- #

    @staticmethod
    def _init_LogisticRegression(params, **kwargs):
        return LogisticRegression(C=max(params['C'], 0.00001), max_iter=2000 ,solver='liblinear')

    @staticmethod
    def _optimize_LogisticRegression(X, y, cv_splits=5, scoring='roc_auc', n_jobs=-1, **kwargs):
        def cv_function(C):
            params = {'C': C}
            return cross_val_score(BestModel._init_LogisticRegression(params, **kwargs), X=X, y=y, cv=cv_splits,
                                   scoring=scoring, n_jobs=n_jobs).mean()

        parameters = {"C": (0.0, 1000)}
        return cv_function, parameters

    # ----- svm ----- #

    @staticmethod
    def _init_SVM(params, **kwargs):
        return LinearSVC(C=max(params['C'], 0.00001), max_iter=1500)

    @staticmethod
    def _optimize_SVM(X, y, cv_splits=5, scoring='roc_auc', n_jobs=-1, **kwargs):
        def cv_function(C):
            params = {'C': C}
            return cross_val_score(BestModel._init_SVM(params, **kwargs), X=X, y=y, cv=cv_splits,
                                   scoring=scoring, n_jobs=n_jobs).mean()

        parameters = {"C": (0.0, 1000)}
        return cv_function, parameters

    # -------------- sklearn API functions -------------- #

    def fit(self, X, y, cv_splits=5, scoring='roc_auc', n_jobs=-1, params=None):
        if self._model_str in ('classifier', 'regressor'):
            models_to_check = ('svm', 'rfc', 'lr') if self._model_str == 'classifier' else (
            'xgbreg', 'rfr', 'lasso')
            best_model, best_score, best_params = None, None, None
            for model_str in models_to_check:
                print(f'------------------ working on {model_str} ------------------')
                model = BestModel(model_str, self._init_points, self._n_iter)
                cv_function, parameters = model._optimize_function(X, y, cv_splits, scoring, n_jobs, **self.get_init_kwargs())
                best_solution = model._bayesian_optimization(cv_function, parameters)
                params, score = best_solution["params"], best_solution["target"]
                print(f'\tResults for {model_str}:\n\t\tbest params={params}\n\t\tbest score={score}')
                if best_score is None or score > best_score:
                    best_model, best_score, best_params = model, score, params
            best_model._fit(X, y, cv_splits, scoring, n_jobs, best_params)
            self.__dict__.update(best_model.__dict__)
            return self._fitted_model
        else:
            return self._fit(X, y, cv_splits, scoring, n_jobs, params)

    def _fit(self, X, y, cv_splits=5, scoring='roc_auc', n_jobs=-1, params=None):
        if params is None:
            cv_function, parameters = self._optimize_function(X, y, cv_splits, scoring, n_jobs, **self.get_init_kwargs())
            best_solution = self._bayesian_optimization(cv_function, parameters)
            params = best_solution["params"]
        model = self._init_function(params, **self.get_init_kwargs())
        model.fit(X, y)
        self._fitted_model = model
        if self._model_str in ('lr', 'lasso'):
            self.feature_importances_ = self._fitted_model.coef_
        else:
            self.feature_importances_ = self._fitted_model.feature_importances_
        return self._fitted_model

    def predict(self, X):
        if self._fitted_model is not None:
            return self._fitted_model.predict(X)
        else:
            raise Exception('Model should be fitted before prediction')

    def fit_predict(self, X, y, cv_splits=5, scoring='roc_auc', n_jobs=-1):
        self.fit(X, y, cv_splits, scoring, n_jobs)
        return self.predict(X)

    def transform(self, X):
        if self._model_str in ('nca', 'mlkr'):
            if self._fitted_model is not None:
                return self._fitted_model.transform(X)
            else:
                raise Exception('Model should be fitted before prediction')
        else:
            raise Exception('Only nca or mlkr models have transform method')

    def predict_proba(self, X):
        if self._fitted_model is not None:
            return self._fitted_model.predict_proba(X)
        else:
            raise Exception('Model should be fitted before prediction')

    @property
    def classes_(self):
        return self._fitted_model.classes_

    @property
    def feature_importances_(self):
        if self._feature_importances_ is not None:
            return self._feature_importances_
        else:
            raise Exception('model should be fitted before feature_importances_')

    @feature_importances_.setter
    def feature_importances_(self, value):
        self._feature_importances_ = value

    def save_model(self, fname):
        with open(fname, 'wb') as file:
            pickle.dump(self._fitted_model, file)

    def load_model(self, fname):
        with open(fname, 'rb') as file:
            self._fitted_model = pickle.load(file)
