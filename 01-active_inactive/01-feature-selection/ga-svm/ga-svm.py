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
    TARGETS = ['active', 'inactive']

    data = joblib.load('../../../0-data/undersampled_data_non_corr.joblib')

    X_r = data['X_r']
    y_r = data['y_r']
    print(y_r.groupby('activity')['activity'].count())

    y_r = y_r['activity']
    X_cols = X_r.columns.to_numpy()
    y_r = y_r.to_numpy(dtype=int)
    X_r = X_r.to_numpy()

    scaler = StandardScaler().fit(X_r)
    X_r = scaler.transform(X_r)

    estimator = SVC(C=10, decision_function_shape='ovo', gamma=0.01, kernel='rbf', max_iter=-1)
    selector = GeneticSelectionCV(
        estimator,
        cv=5,
        verbose=1,
        scoring='accuracy',
        max_features=128,
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

# lambda m, a, b: roc_auc_score(b, m.predict(a), average='macro', max_fpr=1.0, multi_class='ovo'),