import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC

training = pd.read_csv('Datasets/Training.csv')
testing = pd.read_csv('Datasets/Testing.csv')
cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training['prognosis']
y1 = y

reduced_data = training.groupby(training['prognosis']).max()

le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)

clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)

print("Decision Tree Classifier:")
print(clf.score(x_train, y_train))
scores = cross_val_score(clf, x_test, y_test, cv=3)
print("Cross-Validation Scores for Decision Tree Classifier:", scores)
print("Mean CV Score:", scores.mean())

# SVM
model = SVC(C=1.0, kernel='rbf', gamma='scale')
model.fit(x_train, y_train)

print("\nSVM:")
print("Training accuracy:", model.score(x_train, y_train))
print("Testing accuracy:", model.score(x_test, y_test))

# Random Forest
rfc_classifier = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42)
rfc_classifier.fit(x_train, y_train)

print("\nRandom Forest Classifier:")
Y_pred = rfc_classifier.predict(x_test)
accuracy = accuracy_score(y_test, Y_pred)
precision = precision_score(y_test, Y_pred, average='macro')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')

# GuassianClassifier
NB_classifier = GaussianNB()
NB_classifier.fit(x_train, y_train)
Y_pred = NB_classifier.predict(x_test)
precision = precision_score(y_test, Y_pred, average='macro')

print("\nGaussian NB Classifier:")
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')

# Visualizing accuracy and precision results
results = {
    'Algorithm': ['Decision Tree', 'SVM', 'Random Forest', 'Gaussian NB'],
    'Accuracy': [clf.score(x_train, y_train), model.score(x_test, y_test), accuracy_score(y_test, Y_pred), accuracy_score(y_test, Y_pred)],
    'Precision': [scores.mean(), precision_score(y_test, Y_pred, average='macro'), precision, precision]
}

results_df = pd.DataFrame(results)

# Plotting
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
sns.barplot(x='Algorithm', y='Accuracy', data=results_df)
plt.title('Accuracy Comparison')
plt.ylim(0.95, 1.0)  # Adjust y-axis limits to highlight small differences

plt.subplot(1, 2, 2)
sns.barplot(x='Algorithm', y='Precision', data=results_df)
plt.title('Precision Comparison')
plt.ylim(0.95, 1.0)  # Adjust y-axis limits to highlight small differences

plt.tight_layout()
plt.show()
