import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import methods
import best_model_selection as bms
from nca_model import NCAMModel
import re
import sys
import os
import warnings


if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


regex = re.compile(r"\[|\]|<", re.IGNORECASE)
T = 'treatment'
Y = 'readmitted:<30'
KEY_COLUMNS = ['encounter_id', 'patient_nbr']
MED_COLUMNS = ['insulin', 'metformin', 'glyburide-metformin', 'glipizide-metformin',
               'glimepiride-pioglitazone', 'metformin-rosiglitazone',
               'metformin-pioglitazone' , 'no_med']
OUTCOME_COLUMNS = ['readmitted:>30', 'readmitted:NO']
DELETE_COLUMNS = KEY_COLUMNS + MED_COLUMNS + OUTCOME_COLUMNS + [T, Y]


def prepare_data(data, sample=None):
    data = data[(data[T] != 'other') & (data['age:[0-10)'] == 0)]
    if sample is not None:
        data = data.groupby(T).apply(lambda x: x.sample(frac=sample))
    X = data.drop(columns=DELETE_COLUMNS)
    t = (data[T] == 'insulin') * 1
    y = data[Y]
    X.columns = [regex.sub("_", col) if any(x in str(col) for x in {'[', ']', '<'}) else col for col in X.columns]
    y.name = 'readmitted'
    return X, t, y


def fit_best_models(X, t, y, model_num='', default_models=False):
    if default_models:
        model_num = model_num + 'd'
        kwargs = {'params':{'C': 1.0}}
        propensity_model = bms.BestModel('lr', **{'max_iter': 10000})
        s_learner_model = bms.BestModel('lr', **{'max_iter': 10000})
        t_learner_model0 = bms.BestModel('lr', **{'max_iter': 10000})
        t_learner_model1 = bms.BestModel('lr', **{'max_iter': 10000})
    else:
        kwargs = {'cv_splits':5, 'scoring':'balanced_accuracy'}
        propensity_model = bms.BestModel('classifier', init_points=50, n_iter=20,
                                         **{'max_iter': 100, 'n_neighbors': -1})
        s_learner_model = bms.BestModel('classifier', init_points=50, n_iter=20, **{'max_iter': 100, 'n_neighbors': -1})
        t_learner_model0 = bms.BestModel('classifier', init_points=50, n_iter=20,
                                         **{'max_iter': 100, 'n_neighbors': -1})
        t_learner_model1 = bms.BestModel('classifier', init_points=50, n_iter=20,
                                         **{'max_iter': 100, 'n_neighbors': -1})

        nca_model2 = NCAMModel(n_components=2, max_iter=300, verobose=1, n_neighbors=-1, n_jobs=None)
        nca_model2.fit(X, y)
        nca_model2.save_model(f'nca_model_model2.pkl')

        nca_model = NCAMModel(n_components=10, max_iter=100, verobose=1, n_neighbors=-1, n_jobs=None)
        nca_model.fit(X, y)
        nca_model.save_model(f'nca_model_model10.pkl')

        nca_model = NCAMModel(n_components=20, max_iter=100, verobose=1, n_neighbors=-1, n_jobs=None)
        nca_model.fit(X, y)
        nca_model.save_model(f'nca_model_model20.pkl')

        nca_model = NCAMModel(n_components=30, max_iter=100, verobose=1, n_neighbors=-1, n_jobs=None)
        nca_model.fit(X, y)
        nca_model.save_model(f'nca_model_model30.pkl')

    propensity_model.fit(X, t, **kwargs)
    propensity_model.save_model(f'models/propensity_model{model_num}.pkl')

    Xt = pd.concat([X, t], axis=1)
    s_learner_model.fit(Xt, y, **kwargs)
    s_learner_model.save_model(f'models/s_learner_model{model_num}.pkl')

    t_learner_model0.fit(X[t == 0], y[t == 0], **kwargs)
    t_learner_model0.save_model(f'models/t_learner_model0{model_num}.pkl')

    t_learner_model1.fit(X[t == 1], y[t == 1], **kwargs)
    t_learner_model1.save_model(f'models/t_learner_model1{model_num}.pkl')


def get_fitted_best_model(model_num=''):
    propensity_model = bms.BestModel()
    propensity_model.load_model(f'models/propensity_model{model_num}.pkl')
    s_learner_model = bms.BestModel()
    s_learner_model.load_model(f'models/s_learner_model{model_num}.pkl')
    t_learner_model0 = bms.BestModel()
    t_learner_model0.load_model(f'models/t_learner_model0{model_num}.pkl')
    t_learner_model1 = bms.BestModel()
    t_learner_model1.load_model(f'models/t_learner_model1{model_num}.pkl')
    nca_model2 = NCAMModel.load_model('models/nca_model_model2.pkl')
    nca_model10 = NCAMModel.load_model('models/nca_model_model10.pkl')
    nca_model20 = NCAMModel.load_model('models/nca_model_model20.pkl')
    nca_model30 = NCAMModel.load_model('models/nca_model_model30.pkl')
    return propensity_model, s_learner_model, t_learner_model0, t_learner_model1, \
           nca_model2, nca_model10, nca_model20, nca_model30


def train_models(data, iterations):
    for i in range(iterations):
        model_num = f'_{i}'
        data_train, data_test = train_test_split(data, test_size=0.5)
        print(data_train.shape)
        data_train.to_csv(f'data/data_train{model_num}.csv',index=False)
        data_test.to_csv(f'data/data_test{model_num}.csv', index=False)
        X, t, y = prepare_data(data_train)
        fit_best_models(X, t, y, model_num=model_num)


def get_models_results(model_num, train, X, t, y):
    results = []
    for d in [True, False]:
        propensity_model, s_learner_model, t_learner_model0, t_learner_model1, nca_model2, \
        nca_model10, nca_model20, nca_model30 = get_fitted_best_model('_' + model_num + ('d' if d else ''))
        template_results = {'iteration_num': model_num,  'train': train, 'size': X.shape[0], 'insulin': np.mean(t),
                            'readmitted': np.mean(y), 'default': d ,'method': '', 'model_acc': 0, 'model1_acc': 0,
                            'ate0': 0, 'ate1': 0, 'ate': 0}
        ipw_results = template_results.copy()
        e, acc = methods.propensity(X, t, propensity_model)
        ate1, ate0, ate = methods.ipw(t, y, e)
        ipw_results.update({'method': 'ipw', 'model_acc': acc, 'ate0': ate0, 'ate1': ate1, 'ate': ate})
        s_learner_results = template_results.copy()
        ate1, ate0, ate, ate1_star, ate0_star, ate_star, acc = methods.s_learner(X, t, y, s_learner_model)
        s_learner_results.update({'method': 's_learner', 'model_acc': acc, 'ate0': ate0, 'ate1': ate1, 'ate': ate})
        s_learner_star_results = template_results.copy()
        s_learner_star_results.update({'method': 's_learner_star', 'model_acc': acc,
                                       'ate0': ate0_star, 'ate1': ate1_star, 'ate': ate_star})
        t_learner_results = template_results.copy()
        ate1, ate0, ate, ate1_star, ate0_star, ate_star, acc0, acc1 = methods.t_learner(X, t, y, t_learner_model0, t_learner_model1)
        t_learner_results.update({'method': 't_learner', 'model_acc': acc0, 'model1_acc': acc1, 'ate0': ate0, 'ate1': ate1, 'ate': ate})
        t_learner_star_results = template_results.copy()
        t_learner_star_results.update({'method': 't_learner_star', 'model_acc': acc0, 'model1_acc': acc1,
                                       'ate0': ate0_star, 'ate1': ate1_star, 'ate': ate_star})
        results += [ipw_results, s_learner_results, s_learner_star_results, t_learner_results, t_learner_star_results]
    for k in [1, 3, 5, 7, 11, 15]:
        matching_euclidean_results = template_results.copy()
        ate1, ate0, ate, acc0, acc1 = methods.matching(X, t, y, k)
        matching_euclidean_results.update({'method': f'matching_euclidean_k:{k}', 'model_acc': acc0, 'model1_acc': acc1,
                                           'ate0': ate0, 'ate1': ate1, 'ate': ate})
        matching_propensity_results = template_results.copy()
        ate1, ate0, ate, acc0, acc1 = methods.matching(e.reshape(-1, 1), t, y, k)
        matching_propensity_results.update({'method': f'matching_propensity_k:{k}', 'model_acc': acc0, 'model1_acc': acc1,
                                           'ate0': ate0, 'ate1': ate1, 'ate': ate})
        matching_nca2_results = template_results.copy()
        ate1, ate0, ate, acc0, acc1 = methods.matching(X, t, y, k, nca_model2)
        matching_nca2_results.update({'method': f'matching_nca:2_:{k}', 'model_acc': acc0, 'model1_acc': acc1,
                                           'ate0': ate0, 'ate1': ate1, 'ate': ate})
        matching_nca10_results = template_results.copy()
        ate1, ate0, ate, acc0, acc1 = methods.matching(X, t, y, k, nca_model10)
        matching_nca10_results.update({'method': f'matching_nca:10_:{k}', 'model_acc': acc0, 'model1_acc': acc1,
                                           'ate0': ate0, 'ate1': ate1, 'ate': ate})
        matching_nca20_results = template_results.copy()
        ate1, ate0, ate, acc0, acc1 = methods.matching(X, t, y, k, nca_model20)
        matching_nca20_results.update({'method': f'matching_nca:20_:{k}', 'model_acc': acc0, 'model1_acc': acc1,
                                           'ate0': ate0, 'ate1': ate1, 'ate': ate})
        matching_nca30_results = template_results.copy()
        ate1, ate0, ate, acc0, acc1 = methods.matching(X, t, y, k, nca_model30)
        matching_nca30_results.update({'method': f'matching_nca:30_:{k}', 'model_acc': acc0, 'model1_acc': acc1,
                                           'ate0': ate0, 'ate1': ate1, 'ate': ate})
        results += [matching_euclidean_results, matching_propensity_results,
                         matching_nca2_results, matching_nca10_results, matching_nca20_results, matching_nca30_results]
    return pd.DataFrame(results)


def test_models(iterations):
    results = []
    for i in range(iterations):
        model_num = f'_{i}'
        data_train = pd.read_csv(f'data/data_train{model_num}.csv')
        data_test = pd.read_csv(f'data/data_test{model_num}.csv')
        print(data_train.shape, data_test.shape)
        X_train, t_train, y_train = prepare_data(data_train)
        results.append(get_models_results(str(i), True, X_train, t_train, y_train))
        X_test, t_test, y_test = prepare_data(data_test)
        results.append(get_models_results(str(i), False, X_test, t_test, y_test))
    return pd.concat(results, axis=0)


if __name__ == '__main__':
    data = pd.read_csv('casual_diabetes.csv')
    print(data.shape)
    data = data[(data[T] != 'other') & (data['age:[0-10)'] == 0)]
    print(data.shape)
    iterations = 3
    train_models(data, iterations)
    results = test_models(iterations)
    print(results.shape)
    results.to_csv('models_results.csv', index=False)