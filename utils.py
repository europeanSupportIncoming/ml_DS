import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import itertools

main_path = r'C:\Users\paulk\Desktop\ml1\ml1_assignment3'
df = pd.read_csv(os.path.join(main_path, 'D.csv'))
ids = df.pop('id')
labels = df.pop('label')
X = df.copy()
y = labels.copy()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize scaler
scaler = StandardScaler()

# Fit and transform
X_normalized = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns,
    index=X.index
)


def micro_mcc(y_true, y_pred):
    """
    Calculate Micro-averaged Matthews Correlation Coefficient

    Parameters:
    - y_true: array-like of true labels
    - y_pred: array-like of predicted labels

    Returns:
    - micro_mcc: micro-averaged MCC value

    Formula: miM = (r * sum(pi_ii) - 1) / (r - 1)
    where r = number of classes, pi_ii = diagonal of confusion matrix
    """
    # Get confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    r = cm.shape[0]  # number of classes

    # Sum of diagonal (correct predictions)
    sum_TP = np.trace(cm)

    # Total number of samples
    n = np.sum(cm)

    # Using the simplified formula
    micro_mcc_value = (r * sum_TP - n) / (n * (r - 1))

    # Alternative: Using the full formula
    # sum_TP = np.trace(cm)
    # sum_FN = n - sum_TP
    # sum_FP = n - sum_TP
    # sum_TN = n * r - 2 * n + sum_TP  # Simplified: r*n - 2n + sum_TP

    # numerator = sum_TP * sum_TN - sum_FP * sum_FN
    # denominator = np.sqrt((sum_TP + sum_FP) * (sum_TP + sum_FN) *
    #                       (sum_TN + sum_FP) * (sum_TN + sum_FN))
    # micro_mcc_value = numerator / denominator

    return micro_mcc_value


def micro_mcc_from_cm(cm):
    """
    Calculate micro MCC directly from confusion matrix

    Parameters:
    - cm: confusion matrix (r x r)
    """
    r = cm.shape[0]
    n = np.sum(cm)
    sum_TP = np.trace(cm)

    # Simplified formula
    micro_mcc_value = (r * sum_TP - n) / (n * (r - 1))

    return micro_mcc_value


scorer = make_scorer(micro_mcc)

hyperparams = {
    'n_estimators': [2, 5, 10, 25, 50, 75, 100, 150, 250, 500],
    'num_class': [2, 3, 4, 5, 6, 8, 10, 12, 15, 20],
    'hidden_layers': [(4,), (8,4), (16,8), (32,16), (64,32), (128,64), (256,128), (512,256), (64,32,16), (128,64,32)],
    'components_pca': [2, 3, 4, 5, 6, 7, 8, 9, 10, 12]
}


for it in range(10):
    print([hyperparams[key][it] for key in hyperparams.keys()])
    models = {
        'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=hyperparams['n_estimators'][it], random_state=42),
        'XGBoost': xgb.XGBClassifier(objective='multi:softmax', num_class=hyperparams['num_class'][it], random_state=42),
        'MLP': MLPClassifier(hidden_layer_sizes=hyperparams['hidden_layers'][it], max_iter=1000, random_state=42),
        f'SVM with PCA': Pipeline([
            ('pca', PCA(n_components=hyperparams['components_pca'][it])),
            ('svm', SVC(class_weight='balanced', random_state=42))
        ])
    }

    for name, model in models.items():
        scores = cross_val_score(model, X_normalized, y, cv=cv, scoring=scorer, n_jobs=-1)
        print(f"{name:35} | MCC = {scores.mean():.4f} (+/- {scores.std():.4f})")
