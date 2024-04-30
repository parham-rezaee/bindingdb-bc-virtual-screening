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

# rng = np.random.RandomState(304)
# qt = QuantileTransformer(
#     n_quantiles=500, output_distribution="normal", random_state=rng
# ).fit(X_r)
# X_r = qt.transform(X_r)

# selector = SelectKBest(k=128).fit(X_r, y_r)
# features_mask = selector.get_support()
# selected_features = X_cols[features_mask]
# X_r = selector.transform(X_r)

selected_features = pd.read_csv('ga-qda-features.txt', header=None, delimiter=" ")
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

### Label Propagation ('knn','rbf') ###
LP_grid_params = [
    {    
        'kernel': ['knn']
    },
    {
        'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 
        'kernel': ['rbf']
    }
]

LP_estimator = LabelPropagation()
# clf = GridSearchCV(LP_estimator, LP_grid_params, n_jobs=16).fit(X_train, y_train)

### Label Spreading ('knn','rbf') ###
LS_grid_params = [
    {    
        'alpha':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'kernel': ['knn']
    },
    {
        'alpha':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 
        'kernel': ['rbf']
    }
]

LS_estimator = LabelSpreading()
# clf = GridSearchCV(LS_estimator, LS_grid_params, n_jobs=16).fit(X_train, y_train)

### linear SVC ###
LSVC_grid_params = [
    {
        'penalty':['l2'],
        'C': [1, 10, 100, 1000],
        'multi_class':['crammer_singer','ovr'],
        'fit_intercept':[False, True],
        'max_iter':[100000],
        'dual':[True]
    },
        {
        'penalty':['l1'],
        'C': [1, 10, 100, 1000],
        'multi_class':['crammer_singer'],
        'fit_intercept':[False, True],
        'max_iter':[100000],
        'dual':[True]
    },
]

LSVC_estimator = LinearSVC()
# clf = GridSearchCV(LSVC_estimator, LSVC_grid_params, n_jobs=16).fit(X_train, y_train)

### SVC ('linear','poly','sigmoid','rbf') ###
SVC_grid_params = [
    {
        'C': [1, 10, 100, 1000], 
        'kernel': ['linear'],
        'max_iter':[-1],
        'decision_function_shape':['ovo'],
    },
    {
        'C': [1, 10, 100, 1000], 
        'gamma': [0.1, 0.01, 0.001, 0.0001], 
        'kernel': ['sigmoid','rbf'],
        'max_iter':[-1],
        'decision_function_shape':['ovo'],
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

### NuSVC ('linear','poly','sigmoid','rbf') ###
NUSVC_grid_params = [
    {
        'nu':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,],
        'kernel': ['linear'],
        'max_iter':[-1],
        'decision_function_shape':['ovo'],
    },
    {
        'nu':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,],
        'gamma': [0.1, 0.01, 0.001, 0.0001], 
        'kernel': ['sigmoid','rbf'],
        'max_iter':[-1],
        'decision_function_shape':['ovo'],
    },
    # {
    #     'C': [1, 10, 100, 1000], 
    #     'gamma': [0.1, 0.01, 0.001, 0.0001],
    #     'degree': [3, 5, 7, 9],
    #     'kernel': ['poly']
    # }
]

NUSVC_estimator = NuSVC()
# clf = GridSearchCV(NUSVC_estimator, NUSVC_grid_params, n_jobs=16).fit(X_train, y_train)

### Logistic Regression ###
LR_grid_params = [
    {
        'penalty':['l1', 'l2'],
        'fit_intercept':[False, True],
        'solver':['saga'],
        'max_iter':[100000],
        'C': [1, 10, 100, 1000], 
        'multi_class':['multinomial'],
    },
    {
        'penalty':['elasticnet'],  
        'fit_intercept':[False, True],
        'solver':['saga'],
        'max_iter':[100000],
        'C': [1, 10, 100, 1000], 
        'multi_class':['multinomial'],
        'l1_ratio':[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    }
]

LR_estimator = LogisticRegression()
# clf = GridSearchCV(LR_estimator, LR_grid_params, n_jobs=16).fit(X_train, y_train)

### Decision Tree ###
DT_grid_params = {
    'criterion': ['gini'], 
    'splitter': ['best', 'random'],
    'max_features': [32, 64]
}

DT_estimator = DecisionTreeClassifier()
# clf = GridSearchCV(DT_estimator, DT_grid_params, cv=5, verbose=1).fit(X_train, y_train)

### Random Forest ###

RF_grid_params = {
    'bootstrap': [True],
    'criterion': ['gini'], 
    'max_features': [32, 64],
    'n_estimators': [100, 200, 400, 600, 1000]
}

RF_estimator = RandomForestClassifier()
# clf = GridSearchCV(RF_estimator, RF_grid_params, cv=5, verbose=1).fit(X_train, y_train)

### Naive Bayes Bern###
NBB_grid_params = {
    'force_alpha':[True, False]
}
NBB_estimator = BernoulliNB()
# clf = GridSearchCV(NBB_estimator, NBB_grid_params, n_jobs=16).fit(X_train, y_train)

### Naive Bayes GaussianNB###
NBG_grid_params = {
    'var_smoothing': [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]
}
NBG_estimator = GaussianNB()
# clf = GridSearchCV(NBG_estimator, NBG_grid_params, n_jobs=16).fit(X_train, y_train)

### QDA ###
QDA_grid_params = {
    'reg_param': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
}
QDA_estimator = QuadraticDiscriminantAnalysis()
# clf = GridSearchCV(QDA_estimator, QDA_grid_params, n_jobs=16).fit(X_train, y_train)

### LDA ###
LDA_grid_params = {
    'solver':['lsqr'], 
    'shrinkage': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
}

LDA_estimator = LinearDiscriminantAnalysis()
# clf = GridSearchCV(LDA_estimator, LDA_grid_params, n_jobs=16).fit(X_train, y_train)

### Gaussian Process ###
GPC_grid_params = {
    'kernel': [1*RBF(), 1*DotProduct(), 1*Matern(),  1*RationalQuadratic(), 1*WhiteKernel()],
    'multi_class':['one_vs_one']
}

GPC_estimator = GaussianProcessClassifier()
# clf = GridSearchCV(GPC_estimator, GPC_grid_params, n_jobs=16).fit(X_train, y_train)

classifiers = dict(
    KNN=(KNN_grid_params, KNN_estimator),
    # LP=(LP_grid_params, LP_estimator),
    # LS=(LS_grid_params, LS_estimator),
    SVC=(SVC_grid_params, SVC_estimator),
    NUSVC=(NUSVC_grid_params, NUSVC_estimator),
    LSVC=(LSVC_grid_params, LSVC_estimator),
    LR=(LR_grid_params, LR_estimator),
    DT=(DT_grid_params, DT_estimator),
    RF=(RF_grid_params, RF_estimator),
    NBB=(NBB_grid_params, NBB_estimator),
    NBG=(NBG_grid_params, NBG_estimator),
    QDA=(QDA_grid_params, QDA_estimator),
    LDA=(LDA_grid_params, LDA_estimator),
    GPC=(GPC_grid_params, GPC_estimator),
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


