import time
from imblearn import under_sampling
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, mutual_info_classif
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, jaccard_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, RationalQuadratic, WhiteKernel

from genetic_selection import GeneticSelectionCV


if __name__ == "__main__":
    TARGETS = ['EGFR+HER2', 'ER-beta', 'NFKB', 'PR']

    data = joblib.load('../../../0-data/undersampled_data_non_corr.joblib')

    X_r = data['X_a_r']
    y_r = data['y_a_r']
    print(y_r.groupby('C4')['C4'].count())

    y_r = y_r['C4']

    X_cols = X_r.columns.to_numpy()
    y_r = y_r.to_numpy(dtype=int)
    X_r = X_r.to_numpy()

    scaler = StandardScaler().fit(X_r)
    X_r = scaler.transform(X_r)

    estimator = RandomForestClassifier(bootstrap=True, criterion='gini', n_estimators=100)
    selector = GeneticSelectionCV(
        estimator,
        cv=5,
        verbose=1,
        scoring='accuracy',
        max_features=64,
        n_population=200,
        crossover_proba=0.5,
        mutation_proba=0.2,
        n_generations=1000,
        crossover_independent_proba=0.5,
        mutation_independent_proba=0.1,
        tournament_size=3,
        n_gen_no_change=100,
        caching=True,
        n_jobs=10,
    )
    selector = selector.fit(X_r, y_r)
    features_mask = selector.get_support()
    selected_features = X_cols[features_mask]
    with open('ga-svm-features.txt', 'w') as f:
        print(selected_features, file=f)

# X_train, X_test, y_train, y_test = train_test_split(X_r, y_r, test_size=0.3, random_state=42, stratify=y_r)
# print(X_train.shape, X_test.shape)



# for name, (param, estimator) in classifiers.items():
#     print('starting the grid search with', name, 'classifier')
#     clf = GridSearchCV(estimator, param, n_jobs=16).fit(X_train, y_train)
#     with open(f'{name}.txt', 'w') as f:
#         print("Grid Search Results:", file=f)
#         print("---------------------", file=f)
#         for params, mean_score, std_score in zip(clf.cv_results_['params'], clf.cv_results_['mean_test_score'], clf.cv_results_['std_test_score']):
#             print(f"Parameters: {params}", file=f)
#             print(f"Mean Score: {mean_score:.4f}", file=f)
#             print(f"Standard Deviation: {std_score:.4f}", file=f)
#             print("----------------------------------", file=f)
#         print("============", file=f)
#         print(f"Best Params: {clf.best_params_}", file=f)
#         print(f"Best Score: {clf.best_score_}",  file=f)
#         y_pred = clf.best_estimator_.predict(X_test)
#         report = classification_report(y_test, y_pred, target_names=TARGETS)
#         print(report, file=f)

#     joblib.dump(clf, f'{name}.joblib')
#     print('the grid search with', name, 'classifier is finished.')


