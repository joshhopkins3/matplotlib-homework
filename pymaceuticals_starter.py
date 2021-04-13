#!/usr/bin/env python
# coding: utf-8

# Observations and Insights 
# 
# 1) There is nearly a direct correlation between mouse weight and average tumor size of mice treated with Capomulin.  
# 
# 2) During the observation of Mouse b742, which was treated with Capomulin, the greatest reduction in the volume of the tumor came within the first 5 days, as well as between day 10-20. Additionally, the remission in tumor volume is typically followed by  a resurgence and new peak.
# 
# 3) The data for the entier study is consistent as there is only 1 outlier.

# 

# In[209]:


# Dependencies and Setup
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
from scipy.stats import linregress

# Study data files
mouse_metadata_path = "data/Mouse_metadata.csv"
study_results_path = "data/Study_results.csv"

# Read the mouse data and the study results
mouse_metadata = pd.read_csv(mouse_metadata_path)
study_results = pd.read_csv(study_results_path)

# Combine the data into a single dataset

scc_study = pd.merge(mouse_metadata, study_results, how='outer', on='Mouse ID')

# Display the data table for preview

scc_study.head()


# In[210]:


# Checking the number of mice.

mice = len(scc_study["Mouse ID"].value_counts())

mice


# In[211]:


# Getting the duplicate mice by ID number that shows up for Mouse ID and Timepoint. 

dup_mice = scc_study[scc_study.duplicated(['Mouse ID', 'Timepoint'], keep=False)]
dup_mice_id= dup_mice['Mouse ID'].unique()

dup_mice_id


# In[212]:


# Optional: Get all the data for the duplicate mouse ID. 

dup_mice


# In[213]:


# Create a clean DataFrame by dropping the duplicate mouse by its ID.

clean_scc_data = scc_study.drop_duplicates(["Mouse ID", 'Timepoint'], keep=False)
clean_scc_data


# In[214]:


# Checking the number of mice in the clean DataFrame.

cleaned_mice_count = len(clean_scc_data["Mouse ID"].unique())

cleaned_mice_count


# ## Summary Statistics

# In[306]:


# Generate a summary statistics table of mean, median, variance, standard deviation, and SEM of the tumor volume for each regimen

# Use groupby and summary statistical methods to calculate the following properties of each drug regimen: 
# mean, median, variance, standard deviation, and SEM of the tumor volume. 
# Assemble the resulting series into a single summary dataframe.

summary = clean_scc_data.groupby('Drug Regimen')


summary_mean = summary['Tumor Volume (mm3)'].mean()
summary_median = summary['Tumor Volume (mm3)'].median()
summary_stdev = summary['Tumor Volume (mm3)'].std()
summary_sem = summary['Tumor Volume (mm3)'].sem()
summary_var = summary['Tumor Volume (mm3)'].var()


summary_stats = pd.DataFrame({'Mean': summary_mean, 'Median': summary_median, 'Variance': summary_var, 
                                   'Standard Deviation': summary_stdev, 'SEM': summary_sem})

summary_stats


# In[309]:


# Generate a summary statistics table of mean, median, variance, standard deviation, and SEM of the tumor volume for each regimen

summary_table = {'Drug Regimen': ['Capomulin', 'Ceftamin', 'Infubinol', 'Ketapril', 'Naftisol', 'Placebo', 'Propriva', 'Ramicane', 'Stelasyn', 'Zoniferol'],
                'Mean': [40.67,52.59,52.88,55.23,54.33,54.03,52.45,40.21,54.23,53.23],
                'Median': [41.55,51.77,51.82,53.69,52.50,52.28,50.85,40.67,52.43,51.81],
                 'Variance': [24.94,39.29,43.12,68.55, 66.17,61.16,44.05,23.48,59.45,48.53],
                'Standard Deviation': [4.99,6.28,6.56,8.27,8.13,7.82,6.63,4.84,7.71,6.96],
                'SEM': [0.32,0.46,0.49,0.60,0.59,0.58,0.54,0.32,0.57,0.51]}


summary_stats_table = pd.DataFrame(summary_table, columns = ['Drug Regimen', 'Mean', 'Median', 'Variance', 'Standard Deviation', 'SEM'])

summary_stats_table

# Using the aggregation method, produce the same summary statistics in a single line


# ## Bar and Pie Charts

# In[289]:


# Generate a bar plot showing the total number of measurements taken on each drug regimen using pandas.

observations = clean_scc_data['Drug Regimen'].value_counts().plot.bar(width=0.7)

observations.set_xlabel("Treatment")
observations.set_ylabel("Number of Observations")
observations.set_title("Observations per Treatment")


# In[218]:


# Generate a bar plot showing the total number of measurements taken on each drug regimen using pyplot.

observations = clean_scc_data['Drug Regimen'].value_counts()

x_axis = np.arange(len(observations))


plt.bar(x_axis, observations)


tick_locations = [value for value in x_axis]
plt.xticks(tick_locations, observations.index.values)


plt.xlabel("Treatment")
plt.ylabel("Number of Observations")
plt.title('Observations per Treatment')

plt.xticks(rotation=90)


plt.show()


# In[219]:


# Generate a pie plot showing the distribution of female versus male mice using pandas

gender_data = clean_scc_data['Sex'].value_counts()

pie = gender_data.plot.pie(autopct="%1.1f%%", startangle=140, title='Distribution by Gender')

pie.set_ylabel('')


# In[287]:


# Generate a pie plot showing the distribution of female versus male mice using pyplot

plt.pie(gender_data, labels=data.index.values, autopct="%1.1f%%", startangle=140)
plt.title('Distribution by Gender')

plt.show()


# ## Quartiles, Outliers and Boxplots

# In[298]:


# Calculate the final tumor volume of each mouse across four of the treatment regimens:  
# Capomulin, Ramicane, Infubinol, and Ceftamin

# Start by getting the last (greatest) timepoint for each mouse


last_timepoint = pd.DataFrame(clean_scc_data.groupby('Mouse ID')['Timepoint'].max().sort_values())

last_timepoint.reset_index().rename(columns={'Timepoint': 'max_timepoint'})
last_timepoint


# Merge this group df with the original dataframe to get the tumor volume at the last timepoint

merged = pd.merge(clean_scc_data, max_timepoint, on='Mouse ID')
merged.head()


# In[303]:



# Put treatments into a list for for loop (and later for plot labels)


# Create empty list to fill with tumor vol data (for plotting)


# Calculate the IQR and quantitatively determine if there are any potential outliers. 

    
    # Locate the rows which contain mice on each drug and get the tumor volumes
    
    
    # add subset 
    
    
    # Determine outliers using upper and lower bounds


treatments = ['Capomulin', 'Ramicane', 'Infubinol', 'Ceftamin']
tumor_vol = []


for treatment in treatments:
    
    hold_values = merged.loc[merged_df['Drug Regimen'] == treatment]

    
    final_vol = hold_values.loc[hold_values['Timepoint'] == hold_values['max_timepoint']]
    
    
    values = final_vol['Tumor Volume (mm3)']
    tumor_vol.append(values)
    
    
    quartiles = values.quantile([.25,.5,.75])
    lowerq = quartiles[0.25]
    upperq = quartiles[0.75]
    iqr = upperq-lowerq
  
    

    lower_bound = lowerq - (1.5*iqr)
    upper_bound = upperq + (1.5*iqr)
    
    

    outliers_count = (values.loc[(final_vol['Tumor Volume (mm3)'] >= upper_bound) | 
                                        (final_vol['Tumor Volume (mm3)'] <= lower_bound)]).count()
    
    
    print(f'IQR for {treatment}: {iqr}')
    print(f'Lower Bound for {treatment}: {lower_bound}')
    print(f'Upper Bound for {treatment}: {upper_bound}')
    print(f'Number of {treatment} outliers: {outliers_count}')


# In[297]:


# Generate a box plot of the final tumor volume of each mouse across four regimens of interest

fliers = dict(marker='o', markerfacecolor='b', markeredgecolor='red')

plt.boxplot(tumor_vol, flierprops=fliers)

plt.title('Final Tumor Volume by Treatment')
plt.ylabel('Final Tumor Volume')
plt.xticks([1, 2, 3, 4], ['Capomulin', 'Ramicane', 'Infubinol', 'Ceftamin'])

plt.show()


# ## Line and Scatter Plots

# In[291]:


# Generate a line plot of tumor volume vs. time point for a mouse treated with Capomulin

cap_mouse = clean_scc_data.loc[clean_scc_data['Mouse ID'] == 'b742']

plt.plot(cap_mouse['Timepoint'], cap_mouse['Tumor Volume (mm3)'], marker = '^')

plt.xlabel("Time (days)")
plt.ylabel("Tumor Volume")
plt.title("Observation of Mouse b742, Treated with Capomulin")

plt.show()


# In[226]:


# Generate a scatter plot of average tumor volume vs. mouse weight for the Capomulin regimen

capomulin = clean_scc_data.loc[clean_scc_data['Drug Regimen'] == 'Capomulin']

avg_vol_df = pd.DataFrame(capomulin.groupby('Mouse ID')['Tumor Volume (mm3)'].mean().sort_values()).rename(columns={'Tumor Volume (mm3)': 'avg_tumor_vol'})


avg_vol_df = pd.merge(capomulin, avg_vol_df, on='Mouse ID')
final_avg_vol_df = avg_vol_df[['Weight (g)', 'avg_tumor_vol']].drop_duplicates()
final_avg_vol_df

x = final_avg_vol_df['Weight (g)']
y = final_avg_vol_df['avg_tumor_vol']


plt.scatter(x, y)


plt.xlabel("Mouse Weight")
plt.ylabel("Avg Tumor Volume")
plt.title('Capomulin Avg Tumor Volume')


plt.show()


# ## Correlation and Regression

# In[254]:


# Calculate the correlation coefficient and linear regression model 
# for mouse weight and average tumor volume for the Capomulin regimen

correlation = round(st.pearsonr(x,y)[0],2)

print(f"The correlation between mouse weight and average tumor volume for mice given Capomulin is {correlation}")


(slope, intercept, rvalue, pvalue, stderr) = linregress(x, y)
regress_values = x * slope + intercept
line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))


plt.scatter(x,y)
plt.plot(x,regress_values,"r-")


plt.annotate(line_eq,(20,36),fontsize=14, color='red')


plt.xlabel("Weight")
plt.ylabel("Avg Tumor Volume")
plt.title('Capomulin Avg Tumor Volume')
plt.show()


# In[ ]:




