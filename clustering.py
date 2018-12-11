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
average_score = dataset.iloc[:,-3:]
average_score = average_score.applymap(score_labels)


x = average_score
y = dataset.iloc[:,:-3]

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
hot_enc_x   = OneHotEncoder()
label_enc_x = LabelEncoder()
label_enc_y = LabelEncoder()

x = x.apply(label_enc_x.fit_transform)
x = hot_enc_x.fit_transform(x).toarray()

#ToDo: PCA, K-means clustering 
# PCA 
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
x = pca.fit_transform(x)
explained_variance = pca.explained_variance_ratio_

# Elbow Method here
# Fitting K-Means to the dataset
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 8, init = 'k-means++')
y_kmeans = kmeans.fit_predict(x)
# Visualising the clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(x[y_kmeans == 5, 0], x[y_kmeans == 5, 1], s = 100, c = 'black', label = 'Cluster 6')
plt.scatter(x[y_kmeans == 6, 0], x[y_kmeans == 6, 1], s = 100, c = 'orange', label = 'Cluster 7')
plt.scatter(x[y_kmeans == 7, 0], x[y_kmeans == 7, 1], s = 100, c = 'brown', label = 'Cluster 8')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Var1')
plt.ylabel('Var2')
plt.legend()
plt.show()
