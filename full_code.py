import pandas as pd
import numpy as np
import os
from collections import Counter
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, f1_score, precision_recall_fscore_support, \
    matthews_corrcoef
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, cross_val_predict
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import xgboost as xgb
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek

main_path = r'C:\Users\paulk\Desktop\ml1\ml1_assignment3'
df = pd.read_csv(os.path.join(main_path, 'D.csv'))
X = df.copy()
ids = X.pop('id')
labels = X.pop('label')
y = labels.copy()

# Initialize scaler
scaler = StandardScaler()
X_normalized = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns,
    index=X.index
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def macro_mcc(y_true, y_pred):
    """Calculate macro-averaged Matthews Correlation Coefficient"""
    cm = confusion_matrix(y_true, y_pred)
    r = cm.shape[0]
    mcc_per_class = []

    for i in range(r):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        numerator = (TP * TN) - (FP * FN)
        denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

        if denominator == 0:
            mcc = 0
        else:
            mcc = numerator / denominator
        mcc_per_class.append(mcc)

    return np.mean(mcc_per_class)


scorer = make_scorer(macro_mcc)


def evaluate_model(model, X_data, y_data, model_name, sampling_strategy=None, params=None):
    """Evaluate model and return metrics"""
    param_str = f"_{'_'.join([f'{k}={v}' for k, v in params.items()])}" if params else ""

    if sampling_strategy:
        print(f"\n{'=' * 80}")
        print(f"MODEL: {model_name}{param_str} with {sampling_strategy['name']}")
        print(f"{'=' * 80}")

        sampler = sampling_strategy['sampler']
        X_resampled, y_resampled = sampler.fit_resample(X_data, y_data)
        print(f"Original class distribution: {Counter(y_data)}")
        print(f"Resampled class distribution: {Counter(y_resampled)}")

        y_pred = cross_val_predict(model, X_resampled, y_resampled, cv=cv, n_jobs=-1)
        y_true_eval = y_resampled
    else:
        print(f"\n{'=' * 80}")
        print(f"MODEL: {model_name}{param_str} (Baseline)")
        print(f"{'=' * 80}")
        y_pred = cross_val_predict(model, X_data, y_data, cv=cv, n_jobs=-1)
        y_true_eval = y_data

    precision, recall, f1, support = precision_recall_fscore_support(y_true_eval, y_pred, average=None,
                                                                     labels=[0, 1, 2, 3, 4])
    accuracy = accuracy_score(y_true_eval, y_pred)
    macro_f1 = f1_score(y_true_eval, y_pred, average='macro')
    mcc = matthews_corrcoef(y_true_eval, y_pred)
    macro_mcc_val = macro_mcc(y_true_eval, y_pred)

    print(f"\n  Accuracy: {accuracy:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  MCC: {mcc:.4f}")
    print(f"  Macro MCC: {macro_mcc_val:.4f}")
    print(f"  Per-class Precision: {[f'{x:.4f}' for x in precision]}")
    print(f"  Per-class Recall: {[f'{x:.4f}' for x in recall]}")
    print(f"  Per-class F1: {[f'{x:.4f}' for x in f1]}")

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'mcc': mcc,
        'macro_mcc': macro_mcc_val,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support
    }


sampling_strategies = [
    None,
    {'name': 'SMOTE', 'sampler': SMOTE(sampling_strategy='auto', random_state=42)},
    {'name': 'ADASYN', 'sampler': ADASYN(sampling_strategy='auto', random_state=42)},
    {'name': 'SMOTE-Tomek', 'sampler': SMOTETomek(sampling_strategy='auto', random_state=42)}
]

# Define models with their hyperparameter grids
model_configs = {
    'KNN': {
        'model': KNeighborsClassifier(),
        'use_normalized': False,
        'params': {
            'n_neighbors': [5, 10, 14, 21, 31]
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(class_weight='balanced', random_state=42),
        'use_normalized': False,
        'params': {
            'n_estimators': [100, 250, 500],
            'max_depth': [None, 10, 20]
        }
    },
    'XGBoost': {
        'model': xgb.XGBClassifier(objective='multi:softmax', num_class=5, random_state=42),
        'use_normalized': False,
        'params': {
            'n_estimators': [100, 250, 500],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1]
        }
    },
    'MLP': {
        'model': MLPClassifier(max_iter=800, random_state=42),
        'use_normalized': True,
        'params': {
            'hidden_layer_sizes': [(64, 32, 16), (128, 64, 32), (256, 128, 64)],
            'alpha': [0.0001, 0.001]
        }
    },
    'SVM with PCA': {
        'model': Pipeline([
            ('pca', PCA(random_state=42)),
            ('svm', SVC(class_weight='balanced', random_state=42))
        ]),
        'use_normalized': True,
        'params': {
            'pca__n_components': [5, 7, 10],
            'svm__C': [0.1, 1, 10],
            'svm__kernel': ['rbf', 'linear']
        }
    }
}

all_results = {}

for model_name, config in model_configs.items():
    # Select data based on model type
    if config['use_normalized']:
        X_data = X_normalized
    else:
        X_data = X

    print(f"\n{'=' * 80}")
    print(f"EVALUATING: {model_name}")
    print(f"{'=' * 80}")

    base_model = config['model']
    param_grid = config['params']

    # Get all hyperparameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    from itertools import product

    param_combinations = list(product(*param_values))

    for param_combo in param_combinations:
        # Create a fresh copy of the model
        from copy import deepcopy

        model = deepcopy(base_model)

        # Set hyperparameters
        params_dict = {}
        for param_name, param_val in zip(param_names, param_combo):
            if '__' in param_name:
                # Handle pipeline parameters
                model.set_params(**{param_name: param_val})
            else:
                model.set_params(**{param_name: param_val})
            params_dict[param_name] = param_val

        param_str = ', '.join([f"{k}={v}" for k, v in params_dict.items()])
        print(f"\n  Hyperparameters: {param_str}")

        for strategy in sampling_strategies:
            strategy_name = strategy['name'] if strategy else 'Baseline'

            results = evaluate_model(model, X_data, y, model_name, sampling_strategy=strategy, params=params_dict)

            # Store results with unique key
            key = f"{model_name}_{param_str}"
            if key not in all_results:
                all_results[key] = {}
            all_results[key][strategy_name] = results

# Summary table
print("\n" + "=" * 80)
print("SUMMARY COMPARISON TABLE")
print("=" * 80)
print("\n\\begin{table}[h]")
print("    \\centering")
print("    \\begin{tabular}{|l|l|c|c|c|c|}")
print("    \\hline")
print("    Model & Hyperparams & Method & Accuracy & Macro F1 & MCC \\\\")
print("    \\hline")

for model_key, strategies in all_results.items():
    for method_name, results in strategies.items():
        # Truncate long hyperparameter strings
        short_key = model_key[:40] if len(model_key) > 40 else model_key
        print(
            f"    {short_key} & {method_name} & {results['accuracy']:.4f} & {results['macro_f1']:.4f} & {results['mcc']:.4f} \\\\")
print("    \\hline")
print("    \\end{tabular}")
print("    \\caption{Comparison of models with different hyperparameters and sampling strategies}")
print("    \\label{tab:sampling_comparison}")
print("\\end{table}")

# Find best overall
print("\n" + "=" * 80)
print("BEST OVERALL PERFORMANCE")
print("=" * 80)

best_mcc = -1
best_config = None
for model_key, strategies in all_results.items():
    for method_name, results in strategies.items():
        if results['mcc'] > best_mcc:
            best_mcc = results['mcc']
            best_config = (model_key, method_name, results)

if best_config:
    print(f"Best: {best_config[0]} with {best_config[1]}")
    print(f"  Accuracy: {best_config[2]['accuracy']:.4f}")
    print(f"  Macro F1: {best_config[2]['macro_f1']:.4f}")
    print(f"  MCC: {best_config[2]['mcc']:.4f}")
    print(f"  Macro MCC: {best_config[2]['macro_mcc']:.4f}")

# Find best per model
print("\n" + "=" * 80)
print("BEST PER MODEL")
print("=" * 80)

for model_key, strategies in all_results.items():
    best_for_model = max(strategies.items(), key=lambda x: x[1]['mcc'])
    print(f"{model_key}: {best_for_model[0]} -> MCC = {best_for_model[1]['mcc']:.4f}")