import time
from imblearn import under_sampling
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, mutual_info_classif
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


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

selected_features = pd.read_csv('ga-svm-features.txt', header=None, delimiter=" ")
selected_features = selected_features.values.flatten()
feature_indices = [np.where(X_cols == feature)[0][0] for feature in selected_features]
X_r = X_r[:, feature_indices]

X_train, X_test, y_train, y_test = train_test_split(X_r, y_r, test_size=0.3, random_state=42, stratify=y_r)
print(X_train.shape, X_test.shape)

### Nearest Neighbors ###
KNN_grid_params = [
    {
        'algorithm': ['brute'],
        'n_neighbors': [5, 10, 15, 20, 25, 50, 75, 100],
        'p': [2],
        'weights': ['uniform', 'distance'],
        'metric': ['minkowski']
    },
    {
        'algorithm': ['ball_tree', 'kd_tree'],
        'n_neighbors': [5, 10, 15, 20, 25, 50, 75, 100],
        'leaf_size': [20, 30, 40, 50, 60],
        'p': [2],
        'weights': ['uniform', 'distance'],
        'metric': ['minkowski']
    }
]

KNN_estimator = KNeighborsClassifier()
# clf = GridSearchCV(KN_estimator, KN_grid_params, n_jobs=16).fit(X_train, y_train)

### SVC ('linear','poly','sigmoid','rbf') ###
SVC_grid_params = [
    {
        'C': [1, 10, 100, 1000], 
        'kernel': ['linear']
    },
    {
        'C': [1, 10, 100, 1000], 
        'gamma': [0.1, 0.01, 0.001, 0.0001], 
        'kernel': ['sigmoid','rbf']
    },
    # {
    #     'C': [1, 10, 100, 1000], 
    #     'gamma': [0.1, 0.01, 0.001, 0.0001],
    #     'degree': [3, 5, 7, 9],
    #     'kernel': ['poly']
    # }
]

SVC_estimator = SVC()
# clf = GridSearchCV(SVC_estimator, SVC_grid_params, n_jobs=16).fit(X_train, y_train)

### Decision Tree ###
DF_grid_params = {
    'criterion': ['gini', 'entropy', 'log_loss'], 
    'splitter': ['best', 'random'],
    'max_features': [8, 16, 32, 64]
}

DF_estimator = DecisionTreeClassifier()
# clf = GridSearchCV(DF_estimator, DF_grid_params, cv=5, verbose=1).fit(X_train, y_train)

### Random Forest ###

RF_grid_params = {
    'bootstrap': [True],
    'criterion': ['gini', 'entropy', 'log_loss'], 
    'max_features': [8, 16, 32, 64],
    'n_estimators': [100, 200, 400, 600, 1000]
}

RF_estimator = RandomForestClassifier()
# clf = GridSearchCV(RF_estimator, RF_grid_params, cv=5, verbose=1).fit(X_train, y_train)

### AdaBoost ###

AB_grid_params = {
    'n_estimators': [100, 200, 400, 600, 1000],
    'learning_rate': [1, 10, 100, 1000] # 0.0001, 0.001, 0.01, 0.1, 1.0
}

AB_estimator = AdaBoostClassifier()
# clf = GridSearchCV(AB_estimator, AB_grid_params, n_jobs=16).fit(X_train, y_train)

### Naive Bayes ###
NB_grid_params = {
    'var_smoothing': [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]
}
NB_estimator = GaussianNB()
# clf = GridSearchCV(NB_estimator, NB_grid_params, n_jobs=16).fit(X_train, y_train)

### QDA ###
QDA_grid_params = {
    'reg_param': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
}
QDA_estimator = QuadraticDiscriminantAnalysis()
# clf = GridSearchCV(QDA_estimator, QDA_grid_params, n_jobs=16).fit(X_train, y_train)

### LDA ###
LDA_grid_params = {
    'solver':['lsqr', 'eigen'], 
    'shrinkage': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
}

LDA_estimator = LinearDiscriminantAnalysis()
# clf = GridSearchCV(LDA_estimator, LDA_grid_params, n_jobs=16).fit(X_train, y_train)

classifiers = dict(
    KNN=(KNN_grid_params, KNN_estimator),
    SVC=(SVC_grid_params, SVC_estimator),
    DF=(DF_grid_params, DF_estimator),
    RF=(RF_grid_params, RF_estimator),
    AB=(AB_grid_params, AB_estimator),
    NB=(NB_grid_params, NB_estimator),
    QDA=(QDA_grid_params, QDA_estimator),
    LDA=(LDA_grid_params, LDA_estimator),
)

for name, (param, estimator) in classifiers.items():
    print('starting the grid search with', name, 'classifier')
    clf = GridSearchCV(estimator, param, n_jobs=10).fit(X_train, y_train)
    with open(f'{name}.txt', 'w') as f:
        print("Grid Search Results:", file=f)
        print("---------------------", file=f)
        for params, mean_score, std_score in zip(clf.cv_results_['params'], clf.cv_results_['mean_test_score'], clf.cv_results_['std_test_score']):
            print(f"Parameters: {params}", file=f)
            print(f"Mean Score: {mean_score:.4f}", file=f)
            print(f"Standard Deviation: {std_score:.4f}", file=f)
            print("----------------------------------", file=f)
        print("============", file=f)
        print(f"Best Params: {clf.best_params_}", file=f)
        print(f"Best Score: {clf.best_score_}",  file=f)
        y_pred = clf.best_estimator_.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=TARGETS)
        print(report, file=f)

    joblib.dump(clf, f'{name}.joblib')
    print('the grid search with', name, 'classifier is finished.')



# ### Gaussian Process ###

# from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, RationalQuadratic, WhiteKernel

# grid_params = {{'C': [1, 10, 100, 1000], 
# 			   'kernel': [1*RBF(), 1*DotProduct(), 1*Matern(),  1*RationalQuadratic(), 1*WhiteKernel()]
# 			   },
# 			   {'C': [1, 10, 100, 1000], 
# 			   'gamma': [0.1, 0.01, 0.001, 0.0001], 
# 			   'kernel': ['sigmoid','rbf']
# 			   },
# 			   {'C': [1, 10, 100, 1000], 
# 			   'gamma': [0.1, 0.01, 0.001, 0.0001],
# 			   'degree': [3, 5, 7, 9] 
# 			   'kernel': ['poly']}
# 			   }
# estimator = GaussianProcessClassifier()
# clf = GridSearchCV(estimator, grid_params, n_jobs=16).fit(X_train, y_train)