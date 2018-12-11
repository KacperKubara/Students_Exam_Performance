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
x_num =  dataset.iloc[:,-3:]
average_score = average_score.applymap(score_labels)
x = average_score
x_copy = x.copy()

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
hot_enc_x   = OneHotEncoder()
label_enc_x = LabelEncoder()

x = x.apply(label_enc_x.fit_transform)
x = hot_enc_x.fit_transform(x).toarray()

#ToDo: PCA, K-means clustering 
"""# PCA 
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
x = pca.fit_transform(x)
explained_variance = pca.explained_variance_ratio_
"""

# Elbow Method here
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
# n_clusters should be 5

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++')
y_kmeans = kmeans.fit_predict(x)
x_num["cluster"] = y_kmeans
# Visualising the clusters
from mpl_toolkits.mplot3d import axes3d
fig6 = plt.figure()
ax11 = fig6.add_subplot(111, projection='3d')

ax11.scatter((x_num[x_num.cluster == 0])["math score"].values, (x_num[x_num.cluster == 0])["reading score"].values, (x_num[x_num.cluster == 0])["writing score"].values, s = 100, c = 'red', label = 'Cluster 1')
ax11.scatter((x_num[x_num.cluster == 1])["math score"].values, (x_num[x_num.cluster == 1])["reading score"].values, (x_num[x_num.cluster == 1])["writing score"].values, s = 100, c = 'blue', label = 'Cluster 2')
ax11.scatter((x_num[x_num.cluster == 2])["math score"].values, (x_num[x_num.cluster == 2])["reading score"].values, (x_num[x_num.cluster == 2])["writing score"].values, s = 100, c = 'green', label = 'Cluster 3')
ax11.scatter((x_num[x_num.cluster == 3])["math score"].values, (x_num[x_num.cluster == 3])["reading score"].values, (x_num[x_num.cluster == 3])["writing score"].values, s = 100, c = 'cyan', label = 'Cluster 4')
ax11.scatter((x_num[x_num.cluster == 4])["math score"].values, (x_num[x_num.cluster == 4])["reading score"].values, (x_num[x_num.cluster == 4])["writing score"].values, s = 100, c = 'magenta', label = 'Cluster 5')

ax11.set_title('Clusters of customers')
ax11.set_xlabel('Math')
ax11.set_ylabel('Reading')
ax11.set_zlabel('Writing')
ax11.legend()
