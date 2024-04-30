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

selector = SelectKBest(k=64).fit(X_r, y_r)
features_mask = selector.get_support()
selected_features = X_cols[features_mask]
with open('k-best-features.txt', 'w') as f:
        print(selected_features, file=f)