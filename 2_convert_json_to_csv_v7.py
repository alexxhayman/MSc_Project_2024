#!/usr/bin/env python
# coding: utf-8

# # This file covers converting a JSON file format to a CSV 
# JSON is the format in which the dataset from 99 spokes is received. 
# 
# ### CSV is a preferred method because of the following reasons:
# - the data is in a simpler, tabular format which is straightforward for data analysis tools to process
# - Each line in a CSV file usually represents a row in the table
# - Each comma separated value represents the column in the table
# - JSON supports nested, hierarchical data structure. While this is powerful for representing complex data   relationships, it can make data analysis more challenging, as you often need to flatten the structure before analysis.
# - CSV files are more widely supported by data analysis tools, visualisation software, as well as spreadsheet applications. 
# - CSV files can be more efficient in terms of memory and processing time for large datasets, especially if the data is primarily tabular without nested structures. CSV parsing is typically faster and less memory-intensive than JSON parsing.
# - JSON files can be more verbose and larger in size because they include field names and structural brackets in the data, which can lead to increased storage requirements and slower processing for large datasets.
# 
# 
# 

# In[1]:


# This script reads a json file and then determines whether it is nested or not

# To determine if you have a nested JSON file, you can inspect its structure.
# Hierarchical Structure: If the JSON data contains values that are objects, objects within objects or arrays
# within objects, it is nested.
# For example, if a key maps to another object or an array of objects rather than a simple value like a string,
# number, or boolean, the JSON is nested.


# Keys with Complex Values: In a simple, flat JSON structure, each key directly maps to a simple value.
# In a nested structure, keys may map to arrays or objects.


import json

def is_nested(json_obj):
    if isinstance(json_obj, dict):
        for value in json_obj.values():
            if isinstance(value, (dict, list)):
                return True  # Nested structure found
    elif isinstance(json_obj, list):
        for item in json_obj:
            if isinstance(item, (dict, list)):
                return True  # Nested structure found
    return False

def main():
    # Prompt user to enter the file path
    file_path = input("Please enter the path to the JSON file: ")
    
    try:
        # Load the JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Check if the JSON is nested
        print(f"The JSON is {'nested' if is_nested(data) else 'not nested'}")
    except FileNotFoundError:
        print("Error: The file was not found.")
    except json.JSONDecodeError:
        print("Error: The file is not a valid JSON.")

if __name__ == "__main__":
    main()


# ### Final Version
# This version is going to be downloaded as a .py file so that it can be used as a m

# In[1]:


# This is a modified version that dynamically changes the file name according to the limit set

import pandas as pd
import json

def json_to_csv_limit(limit):
    try:
        with open(f'{limit}_road_bikes_2020-2024.json', 'r') as file:
            json_data = json.load(file)

        # Preprocess the data to be normalized
        normalized_data = []
        for item in json_data:
            if 'sizes' in item:
                for size in item['sizes']:
                    # Create a flat dictionary for each size, including item-level attributes
                    size_data = {**size}
                    for key, value in item.items():
                        if key != 'sizes':
                            # Handle list and non-list values appropriately
                            if isinstance(value, list):
                                for i, val in enumerate(value):
                                    size_data[f"{key}_{i}"] = val
                            else:
                                size_data[key] = value
                    normalized_data.append(size_data)

        # Normalize the collected data in a single operation
        if normalized_data:
            full_normalized_data = pd.json_normalize(normalized_data)

            file_name = f'{limit}_road_bikes_2020-2024.csv'
            full_normalized_data.to_csv(file_name, index=False)
            print(f"JSON file converted to CSV and saved as \"{file_name}\"")
        else:
            print("No data to write to CSV.")

    except json.JSONDecodeError as e:
        print(f"An error occurred while parsing JSON: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

json_to_csv_limit('all')


# In[10]:


# This script is used to visualise the data from the csv file
# This is a quick and easy way to check if all columns are displaying
# You can also check to see the values are displaying correctly by comparing them to the source data



# Load the latest fully normalized CSV file to check its contents and structure, focusing on 'geometry'
df_latest = pd.read_csv('road_bikes_2024.csv')

# Display the first few rows of the DataFrame to inspect the 'geometry' attribute and overall structure
df_latest.head()


# In[ ]:





# In[ ]:





# In[ ]:




