#!/usr/bin/env python
# coding: utf-8

# # This file is for adding the required ratios to the CSV file

# In[ ]:


# Functions of this script:

# 1. Calculate the ratios required based off the standardised values.
# 2. print out all the column names
# 3. Drop the unwanted columns now that we have the ratios included. 
# 4. Standardise the ratios. 
# 5. Calculate the mean and std deviation for the new file. 

# after completing all these steps, the csv file is now ready for clustering. 


# In[4]:


import pandas as pd
import numpy as np

# Load the data
file_path = 'modified_imputed_road_bikes_2020-2024.csv'  # Make sure to replace with the correct path
bikes_data = pd.read_csv(file_path)

# Compute the ratios and add them as new columns using vectorized operations
bikes_data['SRR'] = bikes_data['geometry.source.stackMM'] / bikes_data['geometry.source.reachMM']
bikes_data['STRR'] = bikes_data['geometry.source.headTubeAngle_radians'] / bikes_data['geometry.source.trailMM']
bikes_data['CSR'] = (bikes_data['geometry.source.wheelbaseMM'] * bikes_data['geometry.source.trailMM']) / bikes_data['geometry.source.bottomBracketDropMM']
bikes_data['AI'] = (bikes_data['geometry.source.reachMM'] * np.tan(np.radians(bikes_data['geometry.source.headTubeAngle_radians']))) / bikes_data['geometry.source.wheelbaseMM']
bikes_data['CS/WB'] = bikes_data['geometry.source.chainstayLengthMM'] / bikes_data['geometry.source.wheelbaseMM']
bikes_data['BBH/WB'] = bikes_data['geometry.computed.bottomBracketHeightMM'] / bikes_data['geometry.source.wheelbaseMM']
bikes_data['FC/WB'] = bikes_data['geometry.source.frontCenterMM'] / bikes_data['geometry.source.wheelbaseMM']

# Handle potential division by zero and missing data
bikes_data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Save the updated dataframe to a new CSV file
new_file_path = 'all_road_bikes_with_ratios_2020-2024.csv'  # Replace with desired new file path
bikes_data.to_csv(new_file_path, index=False)

print(f"Updated CSV with ratios saved to {new_file_path}")


# In[5]:


# Use this script to Print out all the columns from your CSV file. 
# Change the path to file 

import pandas as pd

# Load the data from a CSV file
bikes_data = pd.read_csv('all_road_bikes_with_ratios_2020-2024.csv')

# Print the column names
print(bikes_data.columns.tolist())


# In[6]:


import pandas as pd

def process_csv(input_file, output_file, columns_to_include):
    # Load the CSV file into a DataFrame with only the specified columns
    df = pd.read_csv(input_file, usecols=columns_to_include)
    
    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_file, index=False)

# Replace with the names of columns you want to include
columns_to_include = [
    'SRR',
    'STRR',
    'CSR',
    'AI', 
    'CS/WB',
    'BBH/WB',
    'FC/WB'
]

# Enter Data:
input_file = 'all_road_bikes_with_ratios_2020-2024.csv'
output_file = 'ready_for_clustering_2020-2024.csv'

process_csv(input_file, output_file, columns_to_include)

print('Filtered Dataset saved to new csv file: ready_for_clustering_2020-2024.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




