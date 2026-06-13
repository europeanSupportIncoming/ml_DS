import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from collections import Counter
import xgboost as xgb
from sklearn.metrics import make_scorer, confusion_matrix

main_path = r'C:\Users\paulk\Desktop\ml1\ml1_assignment3'
df = pd.read_csv(os.path.join(main_path, 'D.csv'))
X = df.copy()
ids = X.pop('id')
labels = X.pop('label')
y = labels.copy()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Normalize data for MLP and SVM
scaler = StandardScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)


def micro_mcc(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    r = cm.shape[0]
    n = np.sum(cm)
    sum_TP = np.trace(cm)
    return (r * sum_TP - n) / (n * (r - 1))
scorer = make_scorer(micro_mcc)


def evaluate_model(model, X_data, y_data, model_name, sampling_strategy=None):
    """Evaluate model with optional sampling and print LaTeX table"""

    if sampling_strategy:
        print(f"\n{'=' * 80}")
        print(f"MODEL: {model_name} with {sampling_strategy['name']}")
        print(f"{'=' * 80}")

        # Apply sampling
        sampler = sampling_strategy['sampler']
        X_resampled, y_resampled = sampler.fit_resample(X_data, y_data)
        print(f"Original class distribution: {Counter(y_data)}")
        print(f"Resampled class distribution: {Counter(y_resampled)}")

        # Use cross-validation on resampled data
        y_pred = cross_val_predict(model, X_resampled, y_resampled, cv=cv, n_jobs=-1)
        y_true_eval = y_resampled
    else:
        print(f"\n{'=' * 80}")
        print(f"MODEL: {model_name} (Baseline)")
        print(f"{'=' * 80}")
        y_pred = cross_val_predict(model, X_data, y_data, cv=cv, n_jobs=-1)
        y_true_eval = y_data
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true_eval, y_pred, average=None,
                                                                     labels=[0, 1, 2, 3, 4])
    accuracy = accuracy_score(y_true_eval, y_pred)
    macro_f1 = f1_score(y_true_eval, y_pred, average='macro')
    mcc = micro_mcc(y_true_eval, y_pred)

    # Print LaTeX table
    print("\n\\begin{table}[h]")
    print("    \\centering")
    print("    \\begin{tabular}{|c|c|c|c|c|c|}")
    print("    \\hline")
    print("    Value   & class 0 & class 1 & class 2 & class 3 & class 4 \\\\")
    print("    \\hline")
    print(
        f"    Precision & {precision[0]:.4f} & {precision[1]:.4f} & {precision[2]:.4f} & {precision[3]:.4f} & {precision[4]:.4f} \\\\")
    print("    \\hline")
    print(f"    Recall & {recall[0]:.4f} & {recall[1]:.4f} & {recall[2]:.4f} & {recall[3]:.4f} & {recall[4]:.4f} \\\\")
    print("    \\hline")
    print(f"    F1  & {f1[0]:.4f} & {f1[1]:.4f} & {f1[2]:.4f} & {f1[3]:.4f} & {f1[4]:.4f} \\\\")
    print("    \\hline")
    print("    \\end{tabular}")
    print(f"    \\caption{{{model_name} with {sampling_strategy['name'] if sampling_strategy else 'Baseline'}}}")
    print(
        f"    \\label{{tab:{model_name.lower().replace(' ', '_')}_{sampling_strategy['name'].lower().replace(' ', '_') if sampling_strategy else 'baseline'}}}")
    print("\\end{table}")
    print()
    print("\\begin{itemize}")
    print(f"    \\item overall accuracy {accuracy:.4f}")
    print(f"    \\item macro averaged F1 score: {macro_f1:.4f}")
    print(f"    \\item average MCC {mcc:.4f}")
    print("\\end{itemize}")

    return {'accuracy': accuracy, 'macro_f1': macro_f1, 'mcc': mcc, 'precision': precision, 'recall': recall, 'f1': f1}


# Define sampling strategies
sampling_strategies = [
    None,  # Baseline
    {
        'name': 'SMOTE',
        'sampler': SMOTE(sampling_strategy='auto', random_state=42)
    },
    {
        'name': 'ADASYN',
        'sampler': ADASYN(sampling_strategy='auto', random_state=42)
    },
    {
        'name': 'SMOTE-Tomek',
        'sampler': SMOTETomek(sampling_strategy='auto', random_state=42)
    }
]

# Define models
models = {
    'KNN (k=14)': KNeighborsClassifier(n_neighbors=14),
    'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=250, random_state=42),
    'XGBoost': xgb.XGBClassifier(objective='multi:softmax', num_class=5, random_state=42),
    'MLP': MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=800, random_state=42),
    'SVM with PCA': Pipeline([
        ('pca', PCA(n_components=7)),
        ('svm', SVC(class_weight='balanced', random_state=42))
    ])
}

# Run all combinations
all_results = {}

for model_name, model in models.items():
    # Determine which data to use
    if model_name in ['MLP', 'SVM with PCA']:
        X_data = X_normalized
    else:
        X_data = X

    all_results[model_name] = {}

    for strategy in sampling_strategies:
        if strategy is None:
            results = evaluate_model(model, X_data, y, model_name, sampling_strategy=None)
            all_results[model_name]['Baseline'] = results
        else:
            results = evaluate_model(model, X_data, y, model_name, sampling_strategy=strategy)
            all_results[model_name][strategy['name']] = results

# Summary comparison table
print("\n" + "=" * 80)
print("SUMMARY COMPARISON TABLE")
print("=" * 80)
print("\n\\begin{table}[h]")
print("    \\centering")
print("    \\begin{tabular}{|l|c|c|c|c|}")
print("    \\hline")
print("    Model & Method & Accuracy & Macro F1 & MCC \\\\")
print("    \\hline")

for model_name, strategies in all_results.items():
    for method_name, results in strategies.items():
        print(
            f"    {model_name} & {method_name} & {results['accuracy']:.4f} & {results['macro_f1']:.4f} & {results['mcc']:.4f} \\\\")
print("    \\hline")
print("    \\end{tabular}")
print("    \\caption{Comparison of models with different sampling strategies}")
print("    \\label{tab:sampling_comparison}")
print("\\end{table}")

# Find best overall
print("\n" + "=" * 80)
print("BEST OVERALL PERFORMANCE")
print("=" * 80)

best_mcc = -1
best_config = None
for model_name, strategies in all_results.items():
    for method_name, results in strategies.items():
        if results['mcc'] > best_mcc:
            best_mcc = results['mcc']
            best_config = (model_name, method_name, results)

print(f"Best: {best_config[0]} with {best_config[1]}")
print(f"  Accuracy: {best_config[2]['accuracy']:.4f}")
print(f"  Macro F1: {best_config[2]['macro_f1']:.4f}")
print(f"  MCC: {best_config[2]['mcc']:.4f}")

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import cross_val_predict, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from collections import Counter
import xgboost as xgb

main_path = r'C:\Users\paulk\Desktop\ml1\ml1_assignment3'
df = pd.read_csv(os.path.join(main_path, 'D.csv'))
ids = df.pop('id')
labels = df.pop('label')
X = df.copy()
y = labels.copy()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Normalize data for MLP and SVM
scaler = StandardScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)


def micro_mcc(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    r = cm.shape[0]
    n = np.sum(cm)
    sum_TP = np.trace(cm)
    return (r * sum_TP - n) / (n * (r - 1))


def evaluate_model(model, X_data, y_data, model_name, sampling_strategy=None):
    """Evaluate model with optional sampling and print LaTeX table"""

    if sampling_strategy:
        print(f"\n{'=' * 80}")
        print(f"MODEL: {model_name} with {sampling_strategy['name']}")
        print(f"{'=' * 80}")

        # Apply sampling
        sampler = sampling_strategy['sampler']
        X_resampled, y_resampled = sampler.fit_resample(X_data, y_data)
        print(f"Original class distribution: {Counter(y_data)}")
        print(f"Resampled class distribution: {Counter(y_resampled)}")

        # Use cross-validation on resampled data
        y_pred = cross_val_predict(model, X_resampled, y_resampled, cv=cv, n_jobs=-1)
        y_true_eval = y_resampled
    else:
        print(f"\n{'=' * 80}")
        print(f"MODEL: {model_name} (Baseline)")
        print(f"{'=' * 80}")
        y_pred = cross_val_predict(model, X_data, y_data, cv=cv, n_jobs=-1)
        y_true_eval = y_data

    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true_eval, y_pred, average=None,
                                                                     labels=[0, 1, 2, 3, 4])
    accuracy = accuracy_score(y_true_eval, y_pred)
    macro_f1 = f1_score(y_true_eval, y_pred, average='macro')
    mcc = matthews_corrcoef(y_true_eval, y_pred)

    # Print LaTeX table
    print("\n\\begin{table}[h]")
    print("    \\centering")
    print("    \\begin{tabular}{|c|c|c|c|c|c|}")
    print("    \\hline")
    print("    Value   & class 0 & class 1 & class 2 & class 3 & class 4 \\\\")
    print("    \\hline")
    print(
        f"    Precision & {precision[0]:.4f} & {precision[1]:.4f} & {precision[2]:.4f} & {precision[3]:.4f} & {precision[4]:.4f} \\\\")
    print("    \\hline")
    print(f"    Recall & {recall[0]:.4f} & {recall[1]:.4f} & {recall[2]:.4f} & {recall[3]:.4f} & {recall[4]:.4f} \\\\")
    print("    \\hline")
    print(f"    F1  & {f1[0]:.4f} & {f1[1]:.4f} & {f1[2]:.4f} & {f1[3]:.4f} & {f1[4]:.4f} \\\\")
    print("    \\hline")
    print("    \\end{tabular}")
    print(f"    \\caption{{{model_name} with {sampling_strategy['name'] if sampling_strategy else 'Baseline'}}}")
    print(
        f"    \\label{{tab:{model_name.lower().replace(' ', '_')}_{sampling_strategy['name'].lower().replace(' ', '_') if sampling_strategy else 'baseline'}}}")
    print("\\end{table}")
    print()
    print("\\begin{itemize}")
    print(f"    \\item overall accuracy {accuracy:.4f}")
    print(f"    \\item macro averaged F1 score: {macro_f1:.4f}")
    print(f"    \\item average MCC {mcc:.4f}")
    print("\\end{itemize}")

    return {'accuracy': accuracy, 'macro_f1': macro_f1, 'mcc': mcc, 'precision': precision, 'recall': recall, 'f1': f1}


# Define sampling strategies
sampling_strategies = [
    None,  # Baseline
    {
        'name': 'SMOTE',
        'sampler': SMOTE(sampling_strategy='auto', random_state=42)
    },
    {
        'name': 'ADASYN',
        'sampler': ADASYN(sampling_strategy='auto', random_state=42)
    },
    {
        'name': 'SMOTE-Tomek',
        'sampler': SMOTETomek(sampling_strategy='auto', random_state=42)
    }
]

# Define models
models = {
    'KNN (k=14)': KNeighborsClassifier(n_neighbors=14),
    'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=250, random_state=42),
    'XGBoost': xgb.XGBClassifier(objective='multi:softmax', num_class=5, random_state=42),
    'MLP': MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=800, random_state=42),
    'SVM with PCA': Pipeline([
        ('pca', PCA(n_components=7)),
        ('svm', SVC(class_weight='balanced', random_state=42))
    ])
}

# Run all combinations
all_results = {}

for model_name, model in models.items():
    # Determine which data to use
    if model_name in ['MLP', 'SVM with PCA']:
        X_data = X_normalized
    else:
        X_data = X

    all_results[model_name] = {}

    for strategy in sampling_strategies:
        if strategy is None:
            results = evaluate_model(model, X_data, y, model_name, sampling_strategy=None)
            all_results[model_name]['Baseline'] = results
        else:
            results = evaluate_model(model, X_data, y, model_name, sampling_strategy=strategy)
            all_results[model_name][strategy['name']] = results

# Summary comparison table
print("\n" + "=" * 80)
print("SUMMARY COMPARISON TABLE")
print("=" * 80)
print("\n\\begin{table}[h]")
print("    \\centering")
print("    \\begin{tabular}{|l|c|c|c|c|}")
print("    \\hline")
print("    Model & Method & Accuracy & Macro F1 & MCC \\\\")
print("    \\hline")

for model_name, strategies in all_results.items():
    for method_name, results in strategies.items():
        print(
            f"    {model_name} & {method_name} & {results['accuracy']:.4f} & {results['macro_f1']:.4f} & {results['mcc']:.4f} \\\\")
print("    \\hline")
print("    \\end{tabular}")
print("    \\caption{Comparison of models with different sampling strategies}")
print("    \\label{tab:sampling_comparison}")
print("\\end{table}")

# Find best overall
print("\n" + "=" * 80)
print("BEST OVERALL PERFORMANCE")
print("=" * 80)

best_mcc = -1
best_config = None
for model_name, strategies in all_results.items():
    for method_name, results in strategies.items():
        if results['mcc'] > best_mcc:
            best_mcc = results['mcc']
            best_config = (model_name, method_name, results)

print(f"Best: {best_config[0]} with {best_config[1]}")
print(f"  Accuracy: {best_config[2]['accuracy']:.4f}")
print(f"  Macro F1: {best_config[2]['macro_f1']:.4f}")
print(f"  MCC: {best_config[2]['mcc']:.4f}")
for i in range(11):
    print(loadings_df[f'PC{i + 1}'])
