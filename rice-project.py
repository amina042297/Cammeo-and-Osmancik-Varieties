import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
rice_cammeo_and_osmancik = fetch_ucirepo(id=545) 
  
# data (as pandas dataframes) 
X = rice_cammeo_and_osmancik.data.features 
y = rice_cammeo_and_osmancik.data.targets 
  
# metadata 
print(rice_cammeo_and_osmancik.metadata) 
  
# variable information 
print(rice_cammeo_and_osmancik.variables) 

# Statistical summary
data = pd.concat([X, y], axis=1)
#data.to_excel('/Users/aminabauyrzan/Desktop/project_rice/data.xlsx', index=True)

grouped_stats = data.groupby('Class').agg(['mean', 'std'])
grouped_stats = grouped_stats.round(2)
feature_stats = X.agg(['mean', 'std']).round(2)
feature_stats_transposed = feature_stats.transpose().round(2)
overall_stats_flat = feature_stats_transposed.values.flatten()
overall_stats_df = pd.DataFrame([overall_stats_flat], index=['Overall'], columns=grouped_stats.columns)
grouped_stats_with_overall = pd.concat([grouped_stats, overall_stats_df])
print(grouped_stats_with_overall)
#grouped_stats_with_overall.to_excel('/Users/aminabauyrzan/Desktop/project_rice/grouped_stats_with_overall.xlsx', index=True)




data['Class'] = data['Class'].map({'Cammeo': 0, 'Osmancik': 1})

X = data.drop(['Class'], axis=1)
y = data['Class']
data = pd.concat([X, y], axis=1)

sns.set(style="ticks", color_codes=True)

# Pairplot for class 0
sns.pairplot(data[data['Class'] == 0], hue='Class')
#plt.savefig("/Users/aminabauyrzan/Desktop/project_rice/Cammeo_pairplot.pdf")

plt.close()

# Pairplot for class 1
sns.pairplot(data[data['Class'] == 1], hue='Class')
#plt.savefig("/Users/aminabauyrzan/Desktop/project_rice/Osmancik_pairplot.pdf")

plt.close()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=677)
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

# KNN

k_values = [3, 5, 7, 9, 11]

accuracies = {}

for k in k_values:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[k] = accuracy

for k, accuracy in accuracies.items():
    print(f"k = {k}: Accuracy = {accuracy}")

k_values = list(accuracies.keys())
accuracies_values = list(accuracies.values())

plt.plot(k_values, accuracies_values, marker='o')
plt.title('Accuracy vs. k')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid(True)
#plt.savefig('/Users/aminabauyrzan/Desktop/project_rice/KNN_accuracy_vs_k.pdf')
plt.show()
plt.close()


knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
print("KNN classifier:")
print("Accuracy:", accuracy_knn)
print("Confusion Matrix:\n", conf_matrix_knn)


# Feature selection with knn

accuracies_truncated = {}

def drop_feature_and_evaluate(feature_name):
    # Drop the specified feature from X_train and X_test
    X_train_truncated = X_train.drop(feature_name, axis=1)
    X_test_truncated = X_test.drop(feature_name, axis=1)
    knn_classifier_truncated = KNeighborsClassifier(n_neighbors=5)
    knn_classifier_truncated.fit(X_train_truncated, y_train)
    y_pred_truncated = knn_classifier_truncated.predict(X_test_truncated)
    accuracy_truncated = accuracy_score(y_test, y_pred_truncated)
    return accuracy_truncated

# Iterate over each feature and drop it to evaluate accuracy
for feature_name in X_train.columns:
    accuracy_truncated = drop_feature_and_evaluate(feature_name)
    accuracies_truncated[feature_name] = accuracy_truncated.round(2)

for feature_name, accuracy_truncated in accuracies_truncated.items():
    print(f"Scenario: Just {feature_name} is missing, Accuracy: {accuracy_truncated}")
    


X_train = X_train.drop('Area', axis=1)
X_test = X_test.drop('Area', axis =1)


# KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
y_pred_knn = knn_classifier.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
print("KNN classifier:")
print("Accuracy:", accuracy_knn)
print("Confusion Matrix:\n", conf_matrix_knn)

# Gaussian Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_nb = gnb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
conf_matrix_nb = confusion_matrix(y_test, y_pred_nb)
print("Naive Bayes Classifier:")
print("Accuracy:", accuracy_nb)
print("Confusion Matrix:\n", conf_matrix_nb)


#Decision tree
dt_classifier = DecisionTreeClassifier(random_state=677)
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
print("Decision Tree Classifier:")
print("Accuracy:", accuracy_dt)
print("Confusion Matrix:\n", conf_matrix_dt)


# Random Forest Classifier
error_rates = []
best_error_rate = 100
best_N = 0
best_d = 0

# Range of values for N and d
N_values = range(1, 11)
d_values = range(1, 6)
np.random.seed(677)
# Iterate over each combination of N and d
for N in N_values:
    for d in d_values:
        rf_classifier = RandomForestClassifier(criterion = 'entropy', n_estimators=N, max_depth=d, random_state=677).fit(X_train, y_train)
        y_pred_rf = rf_classifier.predict(X_test)
        error_rate = 1 - accuracy_score(y_test, y_pred_rf)
        error_rates.append(error_rate)
        if error_rate < best_error_rate:
            best_error_rate = error_rate
            best_N = N
            best_d = d

error_rates = np.array(error_rates).reshape(len(N_values), len(d_values))

# Plot the error rates for different values of N and d
plt.figure(figsize=(10, 6))
plt.imshow(error_rates, cmap='viridis', origin='lower', extent=[1, 5, 1, 10], aspect='auto')
plt.colorbar(label='Error Rate')
plt.title('Error Rates for Different Values of N and d')
plt.xlabel('Max Depth (d)')
plt.ylabel('Number of Trees (N)')
plt.xticks(range(1, 6))
plt.yticks(range(1, 11))
plt.grid(visible=True)
thresh = error_rates.max() / 2.
for i in range(error_rates.shape[0]):
    for j in range(error_rates.shape[1]):
        plt.text(j + 1, i + 1, f'{error_rates[i, j]:.4f}',
                 horizontalalignment='center',
                 color='white' if error_rates[i, j] > thresh else 'black',
                 fontsize=12)
#plt.savefig('/Users/aminabauyrzan/Desktop/project_rice/RF_error_rates.pdf')
plt.show()

print("Best combination of N and d: N =", best_N, ", d =", best_d)

np.random.seed(677)
rf_classifier_best = RandomForestClassifier(criterion = 'entropy', n_estimators=best_N, max_depth=best_d, random_state=677).fit(X_train, y_train)
y_pred_best = rf_classifier_best.predict(X_test)

# accuracy of RF model
accuracy_rf = accuracy_score(y_test, y_pred_best)
conf_matrix_rf = confusion_matrix(y_test, y_pred_best)
print("Random Forest:")
print("Accuracy:", accuracy_rf)
print("Confusion Matrix:\n")
print(conf_matrix_rf)
# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train))
X_test = pd.DataFrame(scaler.transform(X_test))

# Logistic Regression
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
y_pred_logistic = logistic_regression.predict(X_test)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
conf_matrix_logistic = confusion_matrix(y_test, y_pred_logistic)
print("Logistic Regression:")
print("Accuracy of Logistic Regression Classifier:", accuracy_logistic)
print("Confusion Matrix:\n", conf_matrix_logistic)


# Linear Kernel SVM
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_linear)
conf_matrix_svm = confusion_matrix(y_test, y_pred_linear)
print("Linear Kernel SVM:")
print("Accuracy:", accuracy_svm)
print("Confusion Matrix:\n", conf_matrix_svm)

# Result table
model_names = ['KNN', 'Logistic Regression', 'Naive Bayes', 'Decision Tree', 'Random Forest', 'Linear SVM']

conf_matrices = [
    conf_matrix_knn,
    conf_matrix_logistic,
    conf_matrix_nb,
    conf_matrix_dt,
    conf_matrix_rf,
    conf_matrix_svm
]

accuracies = [
    accuracy_knn,
    accuracy_logistic,
    accuracy_nb,
    accuracy_dt,
    accuracy_rf,  
    accuracy_svm
]

metrics = []

for conf_matrix in conf_matrices:
    TP = conf_matrix[1, 1]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]
    TN = conf_matrix[0, 0]

    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    metrics.append([TP, FP, TN, FN, TPR, TNR])

combined_metrics = [metrics[i] + [accuracies[i]] for i in range(len(metrics))]
results_df = pd.DataFrame(combined_metrics, columns=['TP', 'FP', 'TN', 'FN', 'TPR', 'TNR', 'Accuracy'], index=model_names)
print(results_df)

#results_df.to_excel('/Users/aminabauyrzan/Desktop/project_rice/results_df.xlsx', index=True)









