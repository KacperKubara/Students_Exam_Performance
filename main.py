import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
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