#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import random

def sample_rows(input_file, output_file, num_samples=20):
    # Load the dataset
    df = pd.read_csv(input_file)
    
    # Sample num_samples rows from the dataframe at random
    sampled_df = df.sample(n=num_samples, random_state=random.seed(42))
    
    # Save the sampled rows to a new CSV file
    sampled_df.to_csv(output_file, index=False)
    
    # Print confirmation that the file has been saved
    print(f"Saved {num_samples} randomly sampled rows to '{output_file}'.")


# Enter Data:
if __name__ == "__main__":
    input_file = 'encoded_road_bikes_2020-2024.csv'
    output_file = 'sample_road_bikes_2020-2024.csv'
    sample_rows(input_file, output_file)


# In[1]:


# Fetching only bikes from 2020 which are not included in the training data

import requests
import os
import json

# Initialize the API details
base_url = 'https://api.99spokes.com/v1/bikes'
auth_token = os.environ.get('MY_API_KEY')  # Retrieve API key from environment variable
headers = {
    'x-api-key': auth_token  # Use x-api-key header for authorization
}

# Initialize the common query parameters
params = {
    'category': 'road',
    'include': 'sizes',
    'limit': 500,  # Adjust limit to 500
    'year': 2020,  # Set to fetch only bikes from the year 2020
    'cursor': 'start'  # Initial cursor value
}

# List to hold all bikes
all_bikes = []

# Use a while loop to page through the results
while True:
    response = requests.get(base_url, headers=headers, params=params)
    response.raise_for_status()

    data = response.json()
    bikes = data.get('items', [])
    all_bikes.extend(bikes)

    # Update the cursor for the next page
    cursor = data.get('nextCursor')
    if cursor:
        params['cursor'] = cursor
    else:
        print("No more pages to fetch for the year 2020.")
        break  # Exit the loop if there is no next cursor

# Save the fetched data to a JSON file
category = params['category']
file_name = f'all_{category}_bikes_2020.json'
with open(file_name, 'w') as file:
    json.dump(all_bikes, file)

total_count = len(all_bikes)
print(f"Total bikes available: {total_count}")
print(f"Total {category} bikes from 2020 fetched: {len(all_bikes)}")
print(f"All {category} bikes from 2020 saved to {file_name}")


# In[5]:


# ensuring order of 2020 is the same as training data (2021-2024)

import pandas as pd
import numpy as np

def align_csv_columns(source_csv_path, target_csv_path, output_csv_path):
    # Load the source CSV file to determine the column order
    source_df = pd.read_csv(source_csv_path)
    source_columns = source_df.columns.tolist()
    
    # Load the target CSV file whose columns need to be reordered
    target_df = pd.read_csv(target_csv_path)
    
    # Reorder the target DataFrame columns according to the source DataFrame
    # Fill missing columns with NaN
    for column in source_columns:
        if column not in target_df.columns:
            target_df[column] = np.nan
    
    # Reorder target DataFrame columns to match the source DataFrame
    reordered_df = target_df[source_columns]
    
    # Save the reordered DataFrame to a new CSV file
    reordered_df.to_csv(output_csv_path, index=False)
    print(f"Reordered CSV saved to {output_csv_path}")

# Example usage
if __name__ == "__main__":
    source_csv_path = 'encoded_road_bikes_2020-2024.csv'  # Path to the first CSV file
    target_csv_path = 'encoded_road_bikes_2020.csv'  # Path to the second CSV file
    output_csv_path = 'new_encoded_road_bikes_2020.csv'  # Path where the reordered CSV will be saved
    align_csv_columns(source_csv_path, target_csv_path, output_csv_path)


# In[ ]:




