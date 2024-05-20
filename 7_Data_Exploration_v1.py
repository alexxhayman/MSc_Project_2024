#!/usr/bin/env python
# coding: utf-8

# # Caluclate the percentage of subscateogry 

# In[8]:


# caluclate distribution before manipulating data

import pandas as pd

# Load the CSV file into a DataFrame
file_path = 'filtered_combined_road_bikes_2020-2024.csv'  # Update this with your file path
df = pd.read_csv(file_path)

# Count the occurrences of each unique value in the 'subcategory' column
subcategory_counts = df['subcategory'].value_counts()

# Calculate the percentage of each unique instance
total_instances = len(df)
percentage_per_instance = (subcategory_counts / total_instances) * 100

# Print the results
print("Percentage of unique instances in the 'subcategory' column:")
print(percentage_per_instance)


# In[9]:


# caluclate distribution after manipulating data


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file into a DataFrame
file_path = 'std_all_road_bikes_with_ratios_2020-2024.csv'  # Update this with your file path
df = pd.read_csv(file_path)

# Count the occurrences of each unique value in the 'subcategory' column
subcategory_counts = df['subcategory'].value_counts()

# Calculate the percentage of each unique instance
total_instances = len(df)
percentage_per_instance = (subcategory_counts / total_instances) * 100

# Print the results
print("Percentage of unique instances in the 'subcategory' column:")
print(percentage_per_instance)

# Create a line chart for the percentage of bikes in each subcategory
plt.figure(figsize=(12, 8))
sns.lineplot(x=percentage_per_instance.index, y=percentage_per_instance.values, marker='o')
plt.title('Percentage of Bikes in Each Subcategory')
plt.xlabel('Subcategory')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[3]:


# Actual values vs clustering algorithm

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Given values for subcategories
subcategory_percentages = {
    'gravel': 38.007254,
    'race': 23.754177,
    'endurance': 19.070737,
    'aero': 8.565839,
    'triathlon': 4.029471,
    'cyclocross': 3.705343,
    'touring': 1.914784,
    'track': 0.485478,
    'general-road': 0.466916
}

# Given values for clusters
cluster_percentages = [
    33.924951,
    28.206300,
    22.316304,
    10.350687,
    4.926179,
    0.137076,
    0.089956,
    0.041408,
    0.007139
]

# Convert the subcategory percentages to a DataFrame
subcategory_df = pd.DataFrame(list(subcategory_percentages.items()), columns=['subcategory', 'percentage'])

# Convert the cluster percentages to a DataFrame
cluster_df = pd.DataFrame(cluster_percentages, columns=['percentage'])

# Ensure both DataFrames have the same length
if len(subcategory_df) != len(cluster_df):
    raise ValueError("The number of subcategories does not match the number of clusters.")

# Subtract cluster percentages from subcategory percentages
subcategory_df['adjusted_percentage'] = subcategory_df['percentage'] - cluster_df['percentage']

# Print the adjusted percentages
print(subcategory_df)

# Create a line chart for the adjusted percentage of bikes in each subcategory
plt.figure(figsize=(12, 8))
sns.lineplot(x=subcategory_df['subcategory'], y=subcategory_df['adjusted_percentage'], marker='o')
plt.title('Adjusted Percentage of Bikes in Each Subcategory')
plt.xlabel('Subcategory')
plt.ylabel('Adjusted Percentage (%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[62]:


import pandas as pd

# Load the CSV file into a DataFrame
file_path = 'filtered_combined_road_bikes_2020-2024.csv'  # Update this with your file path
df = pd.read_csv(file_path)

# Count the number of unique values in the 'id' column
unique_ids = df['id'].nunique()

# Print the result
print("Number of unique values in the 'id' column:", unique_ids)


# # Calculating Missing Values

# In[1]:


# This function calculates the total percentage of rows that contain at least one missing value

import pandas as pd

def count_rows_with_missing_values(file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Total number of rows in the DataFrame
    total_rows = len(df)
    
    # Check for any missing values in each row
    missing_rows = df.isnull().any(axis=1)
    
    # Count the number of rows with at least one missing value
    count_missing_rows = missing_rows.sum()
    
    # Calculate the percentage of rows with at least one missing value
    percentage_missing = (count_missing_rows / total_rows) * 100
    
    return count_missing_rows, total_rows, percentage_missing

# Specify the path to your CSV file
file_path = 'encoded_road_bikes_2020-2024.csv'

# Get the count of missing rows, total rows, and the percentage
missing_count, total_count, missing_percentage = count_rows_with_missing_values(file_path)
print(f'Total number of rows: {total_count}')
print(f'Number of rows with at least one missing value: {missing_count}')
print(f'Percentage of rows with at least one missing value: {missing_percentage:.2f}%')


# In[5]:


# Calculates the percentage each column contributes to entire dataset. 

import pandas as pd

def calculate_missing_value_percentages(file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Calculate the total number of missing values in the DataFrame
    total_missing_values = df.isnull().sum().sum()
    
    # Calculate the number of missing values for each column
    missing_values_per_column = df.isnull().sum()
    
    # Calculate the percentage contribution of each column to the total missing values
    missing_percentage_per_column = (missing_values_per_column / total_missing_values) * 100
    
    return missing_percentage_per_column

# Specify the path to your CSV file
file_path = 'imputed_road_bikes_2020-2024.csv'

# Calculate the percentage contribution of missing values by each column
column_missing_percentages = calculate_missing_value_percentages(file_path)
print("Percentage of missing values contributed by each column:")
print(column_missing_percentages)


# In[12]:


# Calculate % of missing values per column

import pandas as pd

def calculate_missing_value_percentages(file_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Calculate the number of missing values for each column
    missing_values_per_column = df.isnull().sum()
    
    # Calculate the total number of entries per column
    total_entries_per_column = len(df)
    
    # Calculate the percentage of missing values for each column
    missing_percentage_per_column = (missing_values_per_column / total_entries_per_column) * 100
    
    return missing_percentage_per_column

# Specify the path to your CSV file
file_path = 'all_road_bikes_with_ratios_2020-2024.csv'

# Calculate the percentage of missing values by each column
column_missing_percentages = calculate_missing_value_percentages(file_path)
print("Percentage of missing values per column:")
print(column_missing_percentages)


# In[3]:


# This function is used to confirm the combining road bike folder works properly. 

import pandas as pd

def calculate_missing_pair_percentage(file_path, column1, column2):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Identify rows where both specified columns are missing values
    missing_both = df[df[column1].isnull() & df[column2].isnull()]
    
    # Calculate the total number of rows in the DataFrame
    total_rows = len(df)
    
    # Calculate the percentage of rows where both columns are missing
    percentage_missing_both = (len(missing_both) / total_rows) * 100
    
    return percentage_missing_both

# Specify the path to your CSV file
file_path = 'all_road_bikes_2020-2024.csv'

# Specify the columns to check for missing values
column1 = 'geometry.source.trailMM'
column2 = 'geometry.computed.trailMM'

# Calculate the percentage of rows where both specified columns are missing
missing_percentage = calculate_missing_pair_percentage(file_path, column1, column2)
print(f"Percentage of rows missing values in both {column1} and {column2}: {missing_percentage:.2f}%")


# In[ ]:





# In[44]:


# Calculating unique sizes 

import pandas as pd

def get_unique_values(file_path, column_name):
    # Load the dataset from a CSV file
    df = pd.read_csv(file_path)
    
    # Get unique values from the specified column
    unique_values = df[column_name].dropna().unique()
    
    return unique_values

# Specify the path to your CSV file
file_path = 'all_road_bikes_2020-2024.csv'

# Specify the column you want to search
column_name = 'name'

# Get unique values in the specified column
unique_values = get_unique_values(file_path, column_name)

# Print the unique values
print(f"Unique values in {column_name}:")
print(unique_values)


# In[ ]:





# In[ ]:





# # Determing number of Clusters (K)

# ### Elbow Method
# 
# This helps define the number of clusters (k). 
# This is required before the clustering algorithm can be run. 

# In[7]:


import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Step 1: Reading the CSV file
data = pd.read_csv('ready_for_clustering_2020-2024.csv')

# Step 2: Optionally, select relevant columns if necessary
# data = data[['feature1', 'feature2', 'feature3']]

# Step 3: Apply the Elbow Method
ssd = []
K = range(1, 11)  # Example range for k
for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(data)
    ssd.append(km.inertia_)

# Plotting the Elbow
plt.plot(K, ssd, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# ### Silhouette Method

# In[ ]:





# In[9]:


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Step 1: Read the CSV file into a pandas DataFrame
df = pd.read_csv('ready_for_clustering_2020-2024.csv')

# Step 2: Optional - Select relevant columns if your CSV contains more than the required features
# df = df[['feature1', 'feature2', '...']]

# Step 3: Calculate silhouette scores for different numbers of clusters
silhouette_scores = {}
for k in [3, 4]:
    # Notice the inclusion of n_init=10 to address the warning
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(df)
    score = silhouette_score(df, km.labels_)
    silhouette_scores[k] = score

print(silhouette_scores)


# In[ ]:




