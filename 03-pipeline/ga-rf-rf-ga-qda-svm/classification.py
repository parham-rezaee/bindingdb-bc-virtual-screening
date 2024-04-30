import time
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
import itertools
import seaborn as sns
from sklearn.inspection import permutation_importance
from scipy.cluster import hierarchy
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, RationalQuadratic, WhiteKernel

class ExtractFeatures:
    def __init__(self, filename: str, columns: list[str]) -> list[str]:
        with open(filename) as f:
            features = f.read()

        features = features.strip().split(' ')
        features = [f.strip("'") for f in features if f != '']
        self.features = features
        self.columns = columns
        sf = set(features)
        self.mask = np.array([x in sf for x in columns])

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.mask]

    def transform_pd(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.features]


data = joblib.load('../../0-data/undersampled_data_non_corr.joblib')

X_a = data['X_r']
y_a = data['y_r'][['activity', 'C4A']]
X_t = data['X_a_r']
y_t = data['y_a_r']['C4']

fx_a = ExtractFeatures('activity-features.txt', X_a.columns.to_list())
fx_t = ExtractFeatures('target-features.txt', X_t.columns.to_list())

X_ar_train, X_ar_test, y_a_train, y_a_test = train_test_split(X_a.to_numpy(), y_a.to_numpy(dtype=int), test_size=0.3, random_state=42, stratify=y_a.to_numpy(dtype=int)[:, [0]])
X_tr_train, X_tr_test, y_t_train, y_t_test = train_test_split(X_t.to_numpy(), y_t.to_numpy(dtype=int), test_size=0.3, random_state=42, stratify=y_t.to_numpy(dtype=int))

X_a_train = fx_a.transform(X_ar_train)
X_a_test = fx_a.transform(X_ar_test)
X_t_train = fx_t.transform(X_tr_train)
X_t_test = fx_t.transform(X_tr_test)

scaler_a = StandardScaler().fit(X_a_train)
X_a_train = scaler_a.transform(X_a_train)
X_a_test = scaler_a.transform(X_a_test)

scaler_t = StandardScaler().fit(X_t_train)
X_t_train = scaler_t.transform(X_t_train)
X_t_test = scaler_t.transform(X_t_test)

clf_a = RandomForestClassifier(bootstrap=True, criterion='gini', max_features=32, n_estimators=400, random_state=42)
clf_a.fit(X_a_train, y_a_train[:, [0]].flatten())
y_a_pred = clf_a.predict(X_a_test)
y_a_pred_score = clf_a.predict_proba(X_a_test)
report_a = classification_report(y_a_test[:, [0]].flatten(), y_a_pred, target_names=['active', 'inactive'])
print('ACTIVITY CLASSIFICATION TEST REPORT\n')
print(report_a)
print(20 * '=')

# clf_t = GaussianProcessClassifier(kernel= 1**2 * Matern(length_scale=1, nu=1.5), multi_class='one_vs_rest', random_state=42
clf_t = SVC(C=10, decision_function_shape= 'ovo', gamma=0.01, kernel='rbf', probability=True, random_state=42, max_iter=-1)
clf_t.fit(X_t_train, y_t_train)
y_t_pred = clf_t.predict(X_t_test)
report_t = classification_report(y_t_test, y_t_pred, target_names=['EGFR+HER2', 'ER-beta', 'NFKB', 'PR'])

print('TARGET CLASSIFICATION TEST REPORT\n')
print(report_t)
print(20 * '=')

# COMBINED MODEL
X_at = fx_t.transform_pd(pd.DataFrame(X_ar_test, columns=X_a.columns)).to_numpy()
X_at = scaler_t.transform(X_at)
y_at_pred = clf_t.predict(X_at)
y_at_pred_score = clf_t.predict_proba(X_at)

y_at_pred = -y_at_pred*(y_a_pred - 2)
y_at = y_a_test[:, [1]].flatten()

y_at_pred_score = np.concatenate((y_a_pred_score[:, 1][:, np.newaxis], y_at_pred_score), axis=1)
for i in range(1, 5):
    y_at_pred_score[:, i] =  y_a_pred_score[:, 0] * y_at_pred_score[:, i]

y_at[y_at > 4] = 0

report_at = classification_report(y_at, y_at_pred, target_names=['N/A', 'EGFR+HER2', 'ER-beta', 'NFKB', 'PR'])
print('COMBINED CLASSIFICATION TEST REPORT\n')
print(report_at)
print(20 * '=')


auc_r = metrics.roc_auc_score(y_at, y_at_pred_score, multi_class='ovo', average='macro')
print('Combined AUC: ', auc_r)

pred_active = np.where(y_a_pred == 1)[0]
y_ata_pred = y_at_pred[pred_active]

y_ata = y_a_test[pred_active, [1]].flatten()
y_ata = np.where(y_ata > 4, y_ata - 4, y_ata)

report_ata = classification_report(y_ata, y_ata_pred)
print('COMBINED CLASSIFIVATION [ONLY ACTIVES] TEST REPORT\n')
print(report_ata)
print(20 * '=')
