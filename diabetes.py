import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
diabetes_dataset = pd.read_csv(r'C:\Users\pedam\Downloads\heart.csv')
print(diabetes_dataset.describe())
diabetes_dataset['target'].value_counts()
diabetes_dataset.groupby('target').mean()
# separating the data and labels
X = diabetes_dataset.drop(columns = 'target', axis=1)
Y = diabetes_dataset['target']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)