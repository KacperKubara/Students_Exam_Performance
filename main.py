import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Read the data
dataset = pd.read_csv("StudentsPerformance.csv")

# Create figures and axes
fig0, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize = [12.8, 9.6])

# Plot the histograms
sns.distplot(dataset["math score"], kde = False, label = "Maths", ax = ax0, color = 'b')
ax0.set_title("Math") 
ax0.set_xlabel("") # Remove xlabel
sns.distplot(dataset["reading score"], kde = False, label = "Reading", ax = ax1, color = 'g')
ax1.set_title("Reading")
ax1.set_xlabel("")
sns.distplot(dataset["writing score"], kde = False, label = "Writing", ax = ax2, color = 'y')
ax2.set_title("Writing")
ax2.set_xlabel("")
