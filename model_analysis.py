import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv('archive/dataset.csv')
df = shuffle(df,random_state=42)
df.head()

for col in df.columns:
    df[col] = df[col].str.replace('_',' ')
df.head()    

df.describe()

null_checker = df.apply(lambda x: sum(x.isnull())).to_frame(name='count')
#print(null_checker)

'''plt.figure(figsize=(10,5))
plt.plot(null_checker.index, null_checker['count'])
plt.xticks(null_checker.index, null_checker.index, rotation=45,
horizontalalignment='right')
plt.title('Before removing Null values')
plt.xlabel('column names')
plt.margins(0.1)
plt.show()'''

cols = df.columns
data = df[cols].values.flatten()

s = pd.Series(data)
s = s.str.strip()
s = s.values.reshape(df.shape)
df = pd.DataFrame(s, columns=df.columns)
df.head()

df = df.fillna(0)
df.head()

df1 = pd.read_csv('archive/Symptom-severity.csv')
df1['Symptom'] = df1['Symptom'].str.replace('_',' ')
df1.head()

df1['Symptom'].unique()

vals = df.values
symptoms = df1['Symptom'].unique()
#print(symptoms)

for i in range(len(symptoms)):
    vals[vals == symptoms[i]] = df1[df1['Symptom'] == symptoms[i]]['weight'].values[0]
    
d = pd.DataFrame(vals, columns=cols)
d.head()

d = d.replace('dischromic  patches', 0)
d = d.replace('spotting  urination',0)
df = d.replace('foul smell of urine',0)
df.head(10)

null_checker = df.apply(lambda x: sum(x.isnull())).to_frame(name='count')
#print(null_checker)

'''plt.figure(figsize=(10,5))
plt.plot(null_checker.index, null_checker['count'])
plt.xticks(null_checker.index, null_checker.index, rotation=45,
horizontalalignment='right')
plt.title('After removing Null values')
plt.xlabel('column names')
plt.margins(0.01)
plt.show()

print("Number of symptoms used to identify the disease ",len(df1['Symptom'].unique()))
print("Number of diseases that can be identified ",len(df['Disease'].unique()))'''

df['Disease'].unique()

data = df.iloc[:,1:].values
labels = df['Disease'].values

x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size = 0.8,random_state=42)

tree =DecisionTreeClassifier(criterion='gini',random_state=42,max_depth=13)
tree.fit(x_train, y_train)
preds=tree.predict(x_test)
conf_mat = confusion_matrix(y_test, preds)
df_cm = pd.DataFrame(conf_mat, index=df['Disease'].unique(), columns=df['Disease'].unique())
print('F1-score% =', f1_score(y_test, preds, average='macro')*100, '|', 'Accuracy% =', accuracy_score(y_test, preds)*100)
sns.heatmap(df_cm)

rfc=RandomForestClassifier(random_state=42)
rnd_forest = RandomForestClassifier(random_state=42, max_features='sqrt', n_estimators= 500, max_depth=13)
rnd_forest.fit(x_train,y_train)
preds=rnd_forest.predict(x_test)
conf_mat = confusion_matrix(y_test, preds)
df_cm = pd.DataFrame(conf_mat, index=df['Disease'].unique(), columns=df['Disease'].unique())
print('F1-score% =', f1_score(y_test, preds, average='macro')*100, '|', 'Accuracy% =', accuracy_score(y_test, preds)*100)
sns.heatmap(df_cm)

svm_classifier = SVC()
svm_classifier.fit(x_train, y_train)
svm_preds = svm_classifier.predict(x_test)
svm_f1 = f1_score(y_test, svm_preds, average='macro') * 100
svm_accuracy = accuracy_score(y_test, svm_preds) * 100

print('SVM - F1-score% =', svm_f1, '|', 'Accuracy% =', svm_accuracy)

gnb_classifier = GaussianNB()
gnb_classifier.fit(x_train, y_train)
gnb_preds = gnb_classifier.predict(x_test)
gnb_f1 = f1_score(y_test, gnb_preds, average='macro') * 100
gnb_accuracy = accuracy_score(y_test, gnb_preds) * 100

print('Gaussian Naive Bayes - F1-score% =', gnb_f1, '|', 'Accuracy% =', gnb_accuracy)

discrp = pd.read_csv("archive/symptom_Description.csv")
discrp.head()

ektra7at = pd.read_csv("archive/symptom_precaution.csv")
ektra7at.head()

# Decision Tree
tree = DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=13)
tree.fit(x_train, y_train)
tree_preds = tree.predict(x_test)
tree_f1 = f1_score(y_test, tree_preds, average='macro') * 100
tree_accuracy = accuracy_score(y_test, tree_preds) * 100

print('Decision Tree - F1-score% =', tree_f1, '|', 'Accuracy% =', tree_accuracy)

# Decision Tree
tree = DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=13)
tree.fit(x_train, y_train)
tree_preds = tree.predict(x_test)
tree_f1 = f1_score(y_test, tree_preds, average='macro') * 100
tree_accuracy = accuracy_score(y_test, tree_preds) * 100

# Random Forest
rfc = RandomForestClassifier(random_state=42)
rnd_forest = RandomForestClassifier(random_state=42, max_features='sqrt', n_estimators=500, max_depth=13)
rnd_forest.fit(x_train, y_train)
rnd_forest_preds = rnd_forest.predict(x_test)
rnd_forest_f1 = f1_score(y_test, rnd_forest_preds, average='macro') * 100
rnd_forest_accuracy = accuracy_score(y_test, rnd_forest_preds) * 100

# Plotting all classifiers' F1-score and accuracy
plt.figure(figsize=(10, 7))

# Data for plotting
classifiers = ['Decision Tree', 'Random Forest', 'SVM', 'Gaussian Naive Bayes']
accuracy_scores = [tree_accuracy, rnd_forest_accuracy, svm_accuracy, gnb_accuracy]
f1_scores = [tree_f1, rnd_forest_f1, svm_f1, gnb_f1]

# Plotting the bar graph
# Plotting the bar graph
plt.figure(figsize=(10, 7))

# Plotting accuracy scores
plt.bar(np.arange(len(classifiers))-0.2, accuracy_scores, width=0.4, label='Accuracy', color='blue', align='center')

# Plotting F1-scores
plt.bar(np.arange(len(classifiers))+0.2, f1_scores, width=0.4, label='F1-score', color='green', align='center')

# Labels and titles
plt.title('Classifier Performance')
plt.xlabel('Classifiers')
plt.ylabel('Scores')
plt.xticks(np.arange(len(classifiers)), classifiers)
plt.legend()

# Set y-axis limits
plt.ylim(85, 100)

# Display the plot
plt.show()


# Plotting confusion matrix for Decision Tree
plt.figure(figsize=(8, 6))
sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Decision Tree Confusion Matrix')
plt.show()

# Plotting confusion matrix for Random Forest
plt.figure(figsize=(8, 6))
sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Random Forest Confusion Matrix')
plt.show()

# Plotting confusion matrix for SVM
plt.figure(figsize=(8, 6))
sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('SVM Confusion Matrix')
plt.show()

# Plotting confusion matrix for Gaussian Naive Bayes
plt.figure(figsize=(8, 6))
sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Gaussian Naive Bayes Confusion Matrix')
plt.show()