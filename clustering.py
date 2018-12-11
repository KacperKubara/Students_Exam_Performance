import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

def score_labels(x):
    if x<35:
        return "very low"
    if x>=35 and x<55:
        return "low"
    if x>=55 and x<65:
        return "average"
    if x>=65 and x<75:
        return "good"
    if x>=75 and x<85:
        return "high"
    if x>=85 and x<=100:
        return "very high"    
sns.set()
# Read the data
dataset = pd.read_csv("StudentsPerformance.csv")
"""
Create classes for the exam scores
0-35%    - very low
35-55%   - low
55-65%   - average
65%-75%  - good
75-85%   - high
85%-100% - very high
"""
# Make an average score from 3 exams and label them as above
average_score = (dataset["math score"] + dataset["reading score"]
                 + dataset["writing score"])/3
dataset_average = dataset.drop(labels = ["writing score", "math score", "reading score"],
                       axis = 1)
dataset_average["average score"] = average_score
dataset_average["average score"] = dataset_average["average score"].map(score_labels)

"""
X_train = dataset_average.iloc[:, :-1].values
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
explained_variance = pca.explained_variance_ratio_
"""
x = dataset_average.iloc[:, :-1]
# x = np.transpose(x)
y = dataset_average.iloc[:,-1]

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
hot_enc_x   = OneHotEncoder()
hot_enc_y   = OneHotEncoder(categorical_features = [0])
label_enc_x = LabelEncoder()
label_enc_y = LabelEncoder()

x = x.apply(label_enc_x.fit_transform)
y = label_enc_y.fit_transform(y)

x = hot_enc_x.fit_transform(x).toarray()
y = hot_enc_y.fit_transform(y.reshape(-1, 1)).toarray()

#ToDo: PCA, K-means clustering analysis
