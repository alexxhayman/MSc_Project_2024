#!/usr/bin/env python
# coding: utf-8

# Where we stand:
# 
# 1. JSON file converted to CSV
# 2. Ratios added to CSV file
# 
# Next Steps: 
# 
# 1. Remove unwanted columns
# 2. Treat missing values using median imputation
# 3. Insert ratios
# 4. Standerdise dataset

# # Filter out the columns we do not need any data from

# In[4]:


# Primary Filter

# Filtering out the columns (features) we don't need for analysis. 
# This is done by only including those columns required instead of dropping columns - 
# we don't need as they may change. 

import pandas as pd

def process_csv(input_file, output_file, columns_to_include):
    # Load the CSV file into a DataFrame with only the specified columns
    df = pd.read_csv(input_file, usecols=columns_to_include)
    
    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_file, index=False)

# Replace with the names of columns you want to include
columns_to_include = [
    'name',
    'id',
    'url',
    'subcategory',
    'geometry.source.stackMM',
    'geometry.source.reachMM',
    'geometry.source.seatTubeLengthMM',
    'geometry.source.seatTubeAngle',
    'geometry.source.headTubeLengthMM',
    'geometry.source.topTubeLengthMM',
    'geometry.source.headTubeAngle',
    'geometry.source.chainstayLengthMM',
    'geometry.source.bottomBracketDropMM',
    'geometry.source.wheelbaseMM',
    'geometry.source.rakeMM',
    'geometry.source.trailMM',
    'geometry.computed.stackReachRatio',
    'geometry.computed.bottomBracketHeightMM',
    'geometry.computed.frontCenterMM',
    'geometry.computed.rakeMM',
    'geometry.computed.trailMM',
    'geometry.source.bottomBracketHeightMM',
    'geometry.source.frontCenterMM',
    'geometry.computed.wheelbaseMM',
    'geometry.computed.bottomBracketDropMM',

    
    

    
]

# Enter Data:
input_file = 'all_road_bikes_2020-2024.csv'
output_file = 'primary_filtered_road_bikes_2020-2024.csv'

process_csv(input_file, output_file, columns_to_include)

print('Filtered Dataset saved to new csv file: primary_filtered_road_bikes_2020-2024.csv')


# ## Combine the primary and secondary columns into a single column

# In[5]:


# Combining Trail, Rake, BB Height, BB Drop, Front Center and Wheelbase Columns. 
# There are 2 columns for each of these features. 
# There is a priority for each column if both contain values, otherwise defaults to one with value. 

import pandas as pd

def combine_columns(input_file, output_file):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_file)
    
    # Define column pairs to check and fill missing values
    column_pairs = {
        'geometry.source.trailMM': 'geometry.computed.trailMM',
        'geometry.source.rakeMM': 'geometry.computed.rakeMM',
        'geometry.source.frontCenterMM': 'geometry.computed.frontCenterMM',
        'geometry.source.bottomBracketDropMM': 'geometry.computed.bottomBracketDropMM',
        'geometry.source.wheelbaseMM': 'geometry.computed.wheelbaseMM',
        'geometry.computed.bottomBracketHeightMM': 'geometry.source.bottomBracketHeightMM'  # Handling as per your script logic
    }

    # Fill primary column with secondary column values if primary is empty
    for primary, secondary in column_pairs.items():
        if primary in df.columns and secondary in df.columns:
            df[primary] = df[primary].fillna(df[secondary])
        else:
            print(f"Warning: Missing one of the columns: {primary} or {secondary}")

    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Combined dataset saved to new csv file: {output_file}")

# Enter Data:
input_file = 'primary_filtered_road_bikes_2020-2024.csv'
output_file = 'combined_road_bikes_2020-2024.csv'

combine_columns(input_file, output_file)


# ## Filter out secondary column

# In[6]:


# Secondary Filter

# Now that the 6 columns have been consolidated we can filter out the extra


import pandas as pd

def process_csv(input_file, output_file, columns_to_include):
    # Load the CSV file into a DataFrame with only the specified columns
    df = pd.read_csv(input_file, usecols=columns_to_include)
    
    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_file, index=False)

# Replace with the names of columns you want to include
columns_to_include = [  
    'subcategory', 
    'geometry.source.stackMM',
    'geometry.source.reachMM',
    'geometry.source.seatTubeLengthMM',
    'geometry.source.topTubeLengthMM',
    'geometry.source.seatTubeAngle',
    'geometry.source.headTubeAngle',
    'geometry.source.chainstayLengthMM',
    'geometry.source.bottomBracketDropMM',
    'geometry.source.wheelbaseMM',
    'geometry.source.rakeMM',
    'geometry.source.trailMM',
    'geometry.computed.stackReachRatio',
    'geometry.computed.bottomBracketHeightMM',
    'geometry.source.frontCenterMM',
]
    


# Enter Data:
input_file = 'combined_road_bikes_2020-2024.csv'
output_file = 'filtered_combined_road_bikes_2020-2024.csv'

process_csv(input_file, output_file, columns_to_include)

print('Filtered Dataset saved to new csv file: filtered_combined_road_bikes_2020-2024.csv')


# # Impute Missing Values

# In[8]:


# Remove rows that contain more than 30% empty fields
# This is the threshold at which imputation will begin to introduce heavy bias. 

import pandas as pd

def remove_rows_with_missing_data(csv_file_path, threshold=0.30):
    # Load the dataset from a CSV file
    df = pd.read_csv(csv_file_path)
    
    # Calculate the number of columns
    total_columns = df.shape[1]
    
    # Calculate the maximum allowed missing values per row
    max_allowed_missing = total_columns * threshold
    
    # Remove rows with missing values exceeding the threshold
    # This keeps rows with at least (total_columns - max_allowed_missing) non-missing values
    df_cleaned = df.dropna(thresh=total_columns - max_allowed_missing)
    
    # Save the cleaned dataset to a new CSV file
    new_file_path = 'imputed_road_bikes_2020-2024.csv'
    df_cleaned.to_csv(new_file_path, index=False)
    
    print(f"Data cleaned. Rows with more than {threshold*100}% missing values removed.")
    print(f"Cleaned data saved to {new_file_path}")

    return df_cleaned

# Enter Data:
if __name__ == "__main__":
    # Path to your CSV file
    csv_file_path = 'filtered_combined_road_bikes_2020-2024.csv'
    cleaned_df = remove_rows_with_missing_data(csv_file_path)
    print("Cleaned DataFrame:")
    print(cleaned_df.head())


# In[9]:


# Median imputation. 
# Now impute the rows that contained more than 30% empty values. 

import pandas as pd

def impute_missing_values_median(csv_file_path):
    # Load the dataset from a CSV file
    df = pd.read_csv(csv_file_path)
    
    # List of columns to exclude from imputation (hard-coded)
    exclude_columns = ['subcategory']  # Add more column names as needed
    
    # Separate the excluded columns and the rest of the data
    id_and_other_excluded_columns = df[exclude_columns]
    data_to_impute = df.drop(columns=exclude_columns)
    
    # Impute missing values using the median for each column
    for column in data_to_impute.columns:
        median_value = data_to_impute[column].median()
        data_to_impute[column].fillna(median_value, inplace=True)
    
    # Reattach the excluded columns to the dataframe
    for col in exclude_columns:
        data_to_impute[col] = id_and_other_excluded_columns[col]
    
    # Ensure excluded columns are at the beginning of the dataframe
    cols = exclude_columns + [col for col in data_to_impute.columns if col not in exclude_columns]
    imputed_df = data_to_impute[cols]
    
    # Save the dataset with imputed values to a new CSV file
    new_file_path = 'median_imputed_road_bikes_2020-2024.csv'
    imputed_df.to_csv(new_file_path, index=False)
    
    print(f"Data with median-imputed values has been saved to {new_file_path}")
    
    return imputed_df

# Enter Data:
if __name__ == "__main__":
    # Path to your CSV file
    csv_file_path = 'imputed_road_bikes_2020-2024.csv'
    df_imputed = impute_missing_values_median(csv_file_path)
    print("DataFrame with median-imputed values:")
    print(df_imputed.head())


# # Prepare for Ratios

# In[10]:


# Convert Angles to Radians for Formulas

import pandas as pd
import numpy as np

def convert_angles_to_radians(csv_file_path, angle_columns, new_file_name):
    """
    Reads a DataFrame from a CSV file, converts specified angle columns from degrees to radians,
    renames these columns to indicate the conversion, and saves the updated DataFrame to a new CSV file.

    Parameters:
        csv_file_path (str): The path to the CSV file to process.
        angle_columns (list): A list of column names to convert from degrees to radians.
        new_file_name (str): The name of the new CSV file to save.

    Returns:
        pd.DataFrame: The DataFrame with updated columns where angles are now in radians.
    """
    # Load the dataset from a CSV file
    df = pd.read_csv(csv_file_path)
    
    # Track columns not found
    columns_not_found = []

    # Iterate over each column in the list of angle columns
    for col in angle_columns:
        if col in df.columns:
            # Convert the column from degrees to radians and create a new column for it
            df[col + '_radians'] = np.radians(df[col])
        else:
            columns_not_found.append(col)

    # Drop the original angle columns after all conversions are done
    df.drop(columns=angle_columns, inplace=True, errors='ignore')

    # Provide feedback about missing columns
    if columns_not_found:
        print(f"Warning: Columns not found in the DataFrame: {', '.join(columns_not_found)}")

    # Save the modified DataFrame to a CSV file
    df.to_csv(new_file_name, index=False)
    print(f"Data with angles converted to radians has been saved to {new_file_name}")
    
    return df

# Enter Data:
if __name__ == "__main__":
    # Define the path to your CSV file
    csv_file_path = 'median_imputed_road_bikes_2020-2024.csv'  # Replace with the actual path to your CSV file
    
    # Define the columns whose angles you want to convert
    angle_columns = ['geometry.source.headTubeAngle', 'geometry.source.seatTubeAngle']  # Adjust column names as necessary

    # Specify the name for the new file 
    new_file_name = 'radians_median_imputed_road_bikes_2020-2024.csv'
    
    # Perform the conversion and get the updated DataFrame
    df_updated = convert_angles_to_radians(csv_file_path, angle_columns, new_file_name)
    
    # Print the updated DataFrame to verify changes
    print("Updated DataFrame:")
    print(df_updated.head())


# # Calculate Ratios

# In[11]:


import pandas as pd
import numpy as np

# Load the data
file_path = 'radians_median_imputed_road_bikes_2020-2024.csv'  # Make sure to replace with the correct path
bikes_data = pd.read_csv(file_path)


# List of columns to normalize (add or remove columns as needed)
columns_to_normalize = ['geometry.source.chainstayLengthMM',
                        'geometry.computed.bottomBracketHeightMM',
                        'geometry.source.bottomBracketDropMM', 
                        'geometry.source.stackMM', 
                        'geometry.source.trailMM',
                        'geometry.source.reachMM',
                        'geometry.source.rakeMM',
                        'geometry.source.wheelbaseMM']

# Seat tube length as the reference column
reference_column = 'geometry.source.seatTubeLengthMM'

# Normalize the specified columns
for column in columns_to_normalize:
    normalized_column_name = f'normalized_{column}'  # Naming the new column
    bikes_data[normalized_column_name] = bikes_data[column] / bikes_data[reference_column]


    
# Ratios
bikes_data['SRR'] = bikes_data['geometry.source.stackMM'] / bikes_data['geometry.source.reachMM']
bikes_data['AI'] = (bikes_data['geometry.source.reachMM'] * np.tan(np.radians(bikes_data['geometry.source.headTubeAngle_radians']))) / bikes_data['geometry.source.wheelbaseMM']
bikes_data['CS/BBD'] = bikes_data['geometry.source.chainstayLengthMM'] / bikes_data['geometry.source.bottomBracketDropMM']
bikes_data['ETT/S'] = bikes_data['geometry.source.topTubeLengthMM'] / bikes_data['geometry.source.stackMM']



# New Ratios
bikes_data['T/WB'] = bikes_data['geometry.source.trailMM'] / bikes_data['geometry.source.wheelbaseMM']
bikes_data['HTA/Trail'] = bikes_data['geometry.source.headTubeAngle_radians'] / bikes_data['geometry.source.trailMM']
bikes_data['CSL/STA'] = bikes_data['geometry.source.chainstayLengthMM'] / bikes_data['geometry.source.seatTubeAngle_radians']





# Interactions
bikes_data['BBH_CS_Interaction'] = bikes_data['geometry.computed.bottomBracketHeightMM'] * bikes_data['geometry.source.chainstayLengthMM']
bikes_data['BBH_WB_Interaction'] = bikes_data['geometry.computed.bottomBracketHeightMM'] * bikes_data['geometry.source.wheelbaseMM']



# Composite Indices

bikes_data['Stability_Index'] = (
    0.5 * bikes_data['geometry.computed.bottomBracketHeightMM'] +
    0.3 * bikes_data['geometry.source.chainstayLengthMM'] +
    0.2 * bikes_data['geometry.source.wheelbaseMM']
)

bikes_data['Handling_Index'] = (
    0.5 * bikes_data['geometry.source.headTubeAngle_radians'] +
    0.25 * bikes_data['geometry.source.trailMM'] +
    0.25 * bikes_data['geometry.source.rakeMM']
)

bikes_data['Comfort_Index'] = (
    0.5 * bikes_data['BBH_CS_Interaction'] +
    0.3 * bikes_data['CS/BBD'] +
    0.2 * bikes_data['ETT/S']
)

# Save the DataFrame if needed
# bikes_data.to_csv('path_to_your_updated_data.csv', index=False)  # Uncomment this line and replace with your path

# Use this updated dataset for further modeling and analysis


# Handle potential division by zero and missing data
bikes_data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Save the updated dataframe to a new CSV file
new_file_path = 'all_road_bikes_with_ratios_2020-2024.csv'  # Replace with desired new file path
bikes_data.to_csv(new_file_path, index=False)

print(f"Updated CSV with ratios and indices saved to {new_file_path}")


# # Standardising the Dataset

# In[12]:


# This script handles missing values and then standardises the dataset

# Standardise the dataset using z-normalisation.

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('all_road_bikes_with_ratios_2020-2024.csv')

# Identify numerical columns (excluding any potential ID columns or categorical columns)
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Initialize the StandardScaler
scaler = StandardScaler()

# Z-normalize the numerical columns
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Save the updated dataframe to a new CSV file
df.to_csv('std_all_road_bikes_with_ratios_2020-2024.csv', index=False)

# Print confirmation that the file has been saved
print('File has been saved to std_all_road_bikes_with_ratios_2020-2024.csv')


# In[13]:


# Secondary Filter

# Now that the 6 columns have been consolidated we can filter out the extra


import pandas as pd

def process_csv(input_file, output_file, columns_to_include):
    # Load the CSV file into a DataFrame with only the specified columns
    df = pd.read_csv(input_file, usecols=columns_to_include)
    
    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_file, index=False)

# Replace with the names of columns you want to include
columns_to_include = [
    
    'subcategory',
    'geometry.source.headTubeAngle_radians',
    'geometry.source.seatTubeAngle_radians',
    'normalized_geometry.source.chainstayLengthMM',
    'normalized_geometry.computed.bottomBracketHeightMM',
    'normalized_geometry.source.bottomBracketDropMM',
    'normalized_geometry.source.stackMM', 
    'normalized_geometry.source.trailMM', 
    'normalized_geometry.source.reachMM',
    'normalized_geometry.source.rakeMM', 
    'normalized_geometry.source.wheelbaseMM',
    'SRR', 
    'AI',
    'CS/BBD',
    'ETT/S', 
    'T/WB', 
    'HTA/Trail',
    'CSL/STA', 
    'BBH_CS_Interaction',
    'BBH_WB_Interaction',
    'Stability_Index', 
    'Handling_Index', 
    'Comfort_Index'
]
    
    


# Enter Data:
input_file = 'std_all_road_bikes_with_ratios_2020-2024.csv'
output_file = 'std1_all_road_bikes_with_ratios_2020-2024.csv'

process_csv(input_file, output_file, columns_to_include)

print('Filtered Dataset saved to new csv file: std1_all_road_bikes_2020-2024.csv')


# # Convert Subcateogry to number

# In[14]:


import pandas as pd

def encode_subcategory(csv_file_path, output_file_path):
    """
    Reads a DataFrame from a CSV file, encodes the 'subcategory' column into numeric labels using a predefined mapping,
    and saves the updated DataFrame to a new CSV file.

    Parameters:
        csv_file_path (str): The path to the CSV file to process.
        output_file_path (str): The path where the modified CSV file should be saved.

    Returns:
        pd.DataFrame: The DataFrame with the 'subcategory' column encoded as integers.
    """
    # Load the dataset from a CSV file
    df = pd.read_csv(csv_file_path)
    
    # Check if the 'subcategory' column exists in the DataFrame
    if 'subcategory' in df.columns:
        # Predefined mapping for subcategories
        subcategory_mapping = {
            'gravel': 0,
            'race': 1,
            'endurance': 2,
            'aero': 3,
            'triathlon': 4,
            'cyclocross': 5,
            'touring': 6,
            'general-road': 7,
            'track': 8
        }
        
        # Encode the 'subcategory' column using the mapping
        df['subcategory'] = df['subcategory'].map(subcategory_mapping)
        
        # Check for any subcategories that weren't in the mapping
        if df['subcategory'].isna().any():
            print("Warning: Some subcategories were not in the predefined mapping and have been encoded as NaN.")
        
        # Save the modified DataFrame to the specified new CSV file
        df.to_csv(output_file_path, index=False)
        
        print(f"Data with 'subcategory' encoded saved to {output_file_path}")
        print("Subcategory encoding assignment:")
        for k, v in subcategory_mapping.items():
            print(f"{k}: {v}")
    else:
        print("Error: 'subcategory' column not found in the DataFrame.")

    return df

# Enter Data:
if __name__ == "__main__":
    csv_file_path = 'std1_all_road_bikes_with_ratios_2020-2024.csv'  # Replace with the actual path to your CSV file
    output_file_path = 'encoded_road_bikes_2020-2024.csv'  # Specify the output file name
    df_encoded = encode_subcategory(csv_file_path, output_file_path)
    
    # Optionally print the head of the updated DataFrame to verify the changes
    print("Encoded DataFrame:")
    print(df_encoded.head())


# In[ ]:





# In[ ]:





# In[ ]:





# # End of normalisation

# In[ ]:





# In[ ]:





# In[ ]:





# In[37]:


# Use this script to Print out all the columns from your CSV file. 
# Change the path to file 

import pandas as pd

# Load the data from a CSV file
bikes_data = pd.read_csv('all_road_bikes_with_ratios_2020-2024.csv')

# Print the column names
print(bikes_data.columns.tolist())


# # Print DataFrame Statement Explained 
# 
# The `print(bikes_data.columns.tolist())` statement is used to display the column names of a pandas DataFrame. 
# 
# 1. **`bikes_data`**: This is a variable that typically represents a pandas DataFrame. In your case, it would be the DataFrame that contains your bike data, which you would have loaded from a CSV file using pandas' `read_csv` function.
# 
# 2. **`.columns`**: This is an attribute of the DataFrame that holds an Index object containing the column labels of the DataFrame.
# 
# 3. **`.tolist()`**: This is a method called on the DataFrame's `.columns` attribute. The `Index` object (which `.columns` returns) has a method called `tolist()` that converts the index into a standard Python list containing the column names.
# 
# 4. **`print()`**: This is the standard Python function that outputs information to the console. When you wrap `bikes_data.columns.tolist()` inside a `print()` function, it prints the list of column names to the console.
# 
# Here's how this all works in practice:
# - First, you load your data from a CSV file into the `bikes_data` DataFrame. This operation parses the CSV file and creates a DataFrame structure where the column headers in the CSV become the column labels in the DataFrame.
# - Then, using `bikes_data.columns.tolist()`, you retrieve these labels and convert them into a list.
# - Finally, by passing this list to the `print()` function, you can visually inspect the column names directly in your console or output window.
# 
# This process is helpful for debugging and ensuring that you reference the correct column names in your code, especially when setting up data processing tasks like filtering, scaling, or any operations specific to certain columns. If there's a typo in the column name or if you're unsure about the exact naming, this output lets you quickly verify and correct the names.

# In[ ]:





# In[ ]:




