import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
diabetes_dataset = pd.read_csv('C:\Users\pedam\Downloads\heart.csv')
diabetes_dataset.head() 