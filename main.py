import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
sns.set()
# Read the data
dataset = pd.read_csv("StudentsPerformance.csv")

# Create figures and axes
fig0, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize = [12.8, 9.6])

# Plot the histograms for exam scores distribiution
sns.distplot(dataset["math score"], kde = False, label = "Maths", ax = ax0, color = 'b')
ax0.set_xlabel("Math")
sns.distplot(dataset["reading score"], kde = False, label = "Reading", ax = ax1, color = 'g')
ax1.set_xlabel("Reading")
sns.distplot(dataset["writing score"], kde = False, label = "Writing", ax = ax2, color = 'y')
ax2.set_xlabel("Writing")

# Create figures and axes
fig1, ((ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(3, 2, figsize = [12.8, 9.6])

# Plot the histograms for exam scores distribiution based on gender
dataset_male = dataset[dataset["gender"] == "male"]
dataset_female = dataset[dataset["gender"] == "female"]

sns.distplot(dataset_male["math score"], kde = False, label = "Maths", ax = ax3, color = 'b')
ax3.set_xlabel("Math_male")
sns.distplot(dataset_female["math score"], kde = False, label = "Maths", ax = ax4, color = 'b')
ax4.set_xlabel("Math_female")
sns.distplot(dataset_male["reading score"], kde = False, label = "Reading", ax = ax5, color = 'g')
ax5.set_xlabel("Reading_male")
sns.distplot(dataset["reading score"], kde = False, label = "Reading", ax = ax6, color = 'g')
ax6.set_xlabel("Reading_female")
sns.distplot(dataset_male["writing score"], kde = False, label = "Writing", ax = ax7, color = 'y')
ax7.set_xlabel("Writing_male")
sns.distplot(dataset_female["writing score"], kde = False, label = "writing", ax = ax8, color = 'y')
ax8.set_xlabel("Writing_female")
# Visualise the average score based on gender
male_mean = dataset_male[["math score", "reading score", "writing score"]].mean()
female_mean = dataset_female[["math score", "reading score", "writing score"]].mean()
mean_scores_by_gender = pd.concat([male_mean, female_mean], axis = 1, names = ["test", "lol"])
mean_scores_by_gender.columns = ["Male Mean", "Female Mean"] 
display(mean_scores_by_gender)

# Results based on parental level of education
display(dataset["parental level of education"].unique())
dataset["parental level of education"] = dataset["parental level of education"].map(lambda x: "high school" if x == "some high school" else x)
dataset["parental level of education"] = dataset["parental level of education"].map(lambda x: "college" if x == "some college" else x)
education_level_list = dataset["parental level of education"].unique()
df_mean = pd.Series()
fig2 , ax = plt.subplots(3, 1, figsize = [10, 15], sharex= True)

# Create neat table for mean values
for i, education_level in enumerate(education_level_list):
    mean = dataset[dataset["parental level of education"] == education_level].mean()
    mean = mean.rename(education_level)
    df_mean = pd.concat([df_mean, mean], axis = 1, sort = False)

df_mean = df_mean.drop(df_mean.columns[0], axis = 1)

# Plot the exam score based on parental education
ax[0] = sns.barplot(x = "parental level of education", y = "math score", 
                    data = dataset, estimator = np.mean, ax = ax[0])
ax[1] = sns.barplot(x = "parental level of education", y = "reading score", 
                    data = dataset, estimator = np.mean, ax = ax[1])
ax[2] = sns.barplot(x = "parental level of education", y = "writing score", 
                    data = dataset, estimator = np.mean, ax = ax[2])

# Display the mean table
display(df_mean)

# Display a heatmap with the numeric values in each cell
fig4, ax9 = plt.subplots(figsize=(12, 5))
sns.heatmap(df_mean,linewidths=.1, ax=ax9)

# Results based on the lunch type
dataset_lunch = dataset[["lunch", "math score", "reading score", "writing score"]].copy()
dataset_lunch = dataset_lunch.groupby(by = ["lunch"]).mean()
# Display the table and the heatmap
display(dataset_lunch)
fig5, ax10 = plt.subplots(figsize=(12, 5))
sns.heatmap(dataset_lunch,linewidths=.1, ax=ax10)
