#!/usr/bin/env python
# coding: utf-8

# # This file is for making API requests to the 99 Spokes server
# 
# The API Key is stored in the terminal on the local Macbook machine
# This is to avoid sharing the API key when pushing the code or sharing with others
# 
# API Variable = MY_API_KEY

# ### Breakdown on how to use the terminal on Mac OS (14)
# 
# 1. Set an environment Variable (theAPI Key) to your shell configuration.
# 2. Open terminal, check is shell is using "zsh" as default shell, to do this type:
# 
# "echo $SHELL" 
# 
# If the output ends with "/zsh", then your default shell is zsh.
# 
# 3. Edit the shell configuration file, type:
# "nano ~/.zshrc"
# 
# 4. In the file, move the cursor to below the last line, outside the block of code managed by conda init to avoid interfering with Conda's initialization script. 
# 5. Add the following line:
# 
# "export MY_API_KEY="your_actual_api_key_here""
# 
# 6. "Ctrl O" 
# will write (save) the changes to the file. 
# 
# 7. Press "Enter" 
# to confirm these changes
# 
# 8. Restart the terminal:
# 
# "source ~/.zshrc"
# 
# 9. Confirm the environment variable is set, you can echo it in the Terminal:
# 
# "echo $MY_API_KEY"
# 
# 
# 
# 

# # Testing API Key is set

# In[1]:


# This script is for testing whether the environment variable has been set and can be accessed by jupyter notebooks

import os

def apikey_test():

    # Retrieve the API key from the environment variable
    api_key = os.environ.get('MY_API_KEY')

    # Check if the API key was successfully retrieved
    if api_key is not None:
        print("API key is set.")
    else:
        print("API key is not set.")
        
apikey_test()


# In[ ]:





# # Making an API Call
# 
# Now that we know the API key is set and is reachable, we can make our first API request

# ### Final Version
# This version is going to be saved as a module and called in a new notebook using a function 

# In[1]:


# Version Final
# This version now handles all elements required for the final API Request:
# 1. It handles the authorisation using the x-api-key header (which was adjusted from Bearer at first). 
# 2. The API Key is stored in the local environment in the terminal as variable "MY_API_KEY". 
# 3. The cursor parameter is set as 'start' to begin with as it is the first request. 
# 4. A while loop is used to constantly request for the next 500 items (the limit), until the 'nextCursor' is none.
# 5. All the items are then saved in a JSON file (which is dynamically named according to the params

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
}

# List to hold all bikes
all_bikes = []

# Loop through the years 2020 to 2024
for year in range(2020, 2025):
    params['year'] = year
    params['cursor'] = 'start'  # Reset cursor for each year

    # Use a while loop to page through the results for each year
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
            print(f"No more pages to fetch for year {year}.")
            break  # Exit the loop if there is no next cursor

# Save the fetched data to a JSON file
category = params['category']
file_name = f'all_{category}_bikes_2020-2024.json'
with open(file_name, 'w') as file:
    json.dump(all_bikes, file)

total_count = len(all_bikes)
print(f"Total bikes available: {total_count}")
print(f"Total {category} bikes from 2020 to 2024 fetched: {len(all_bikes)}")
print(f"All {category} bikes from 2020 to 2024 saved to {file_name}")


# Let's break down how this code works step by step:
# 
# 1. **Import Necessary Libraries**
#    - `requests` for making HTTP requests.
#    - `os` to access environment variables.
#    - `json` for JSON parsing and file operations.
# 
# 2. **Initialize API Details**
#    - `base_url` holds the endpoint URL of the 99 Spokes API for bike information.
#    - `auth_token` is fetched from an environment variable `MY_API_KEY`, ensuring that the API key is not hardcoded in the script. This is a security best practice.
#    - `headers` dictionary contains the API key under `x-api-key`, used for authenticating requests to the API.
# 
# 3. **Set Up Query Parameters**
#    - `params` dictionary contains parameters for the API request: year of the bikes, category (`road`), limit per request (`500`), and initial cursor value (`start`). These parameters help in filtering and paginating the API response.
# 
# 4. **Fetch and Aggregate Data**
#    - An empty list `all_bikes` is initialized to store the fetched bike data.
#    - A `while` loop starts, making repeated requests to the API endpoint as long as there is data to fetch.
#    - Inside the loop, `requests.get()` is called with the API `base_url`, `headers`, and `params` to fetch the data. `.raise_for_status()` ensures that an exception is thrown for HTTP error responses, which helps in debugging failed requests.
#    - The response is parsed to JSON, and the `items` key is accessed to get the list of bikes, which is then extended into the `all_bikes` list.
#    - The `cursor` is checked in the response. If there's a `nextCursor`, the `params['cursor']` is updated to fetch the next set of data; otherwise, the loop breaks.
# 
# 5. **Save Data to JSON File**
#    - After exiting the loop, variables `year` and `category` are fetched from the `params` for use in naming the output file.
#    - The file name is dynamically created using `year` and `category`, ensuring that the saved file clearly indicates the data it contains.
#    - `json.dump()` writes the aggregated bike data into the named file in JSON format.
# 
# 6. **Print Summary**
#    - The total number of bikes fetched is printed, alongside a confirmation message indicating the completion of data retrieval and the file where the data is saved.
# 
# By following these steps, the script robustly handles API communication, data fetching, and saving while maintaining good practices such as dynamic naming, pagination handling, and error checking.

# In[ ]:





# # Fetching Only 10 Bikes

# In[8]:


# This script requests up to 10 bikes using API pagination.
# It's used to test the ratios calculations, read the file, calculate ratios, and insert them into the file.

import requests
import os
import json

# Initialize the API details
base_url = 'https://api.99spokes.com/v1/bikes'
auth_token = os.environ.get('MY_API_KEY')  # Retrieve API key from environment variable
headers = {
    'x-api-key': auth_token  # Use x-api-key header for authorization
}

# Initialize the query parameters
params = {
    'year': 2024,
    'category': 'road',
    'include': 'sizes',
    'limit': 10  # Set limit to fetch up to 10 bikes per API call
}

# List to hold all bikes
all_bikes = []

# Use a while loop to page through the results
try:
    while len(all_bikes) < 10:
        response = requests.get(base_url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()
        bikes = data.get('items', [])
        all_bikes.extend(bikes[:max(0, 10 - len(all_bikes))])  # Adjust to not exceed 10 bikes

        # Update the cursor for the next page
        cursor = data.get('nextCursor')
        if cursor and len(all_bikes) < 10:
            params['cursor'] = cursor
        else:
            print("No more pages to fetch or reached the limit of 10 bikes.")
            break  # Exit the loop if there is no next cursor or 10 bikes are collected
except requests.RequestException as e:
    print(f"An error occurred: {e}")

year = params['year']
limit = params['limit']
category = params['category']
file_name = f'{limit}_{category}_bikes_{year}.json'

# Save the fetched data to a JSON file
with open(file_name, 'w') as file:
    json.dump(all_bikes, file)

total_count = len(all_bikes)
print(f"Total bikes available: {total_count}")
print(f"Total {category} bikes from {year} fetched: {len(all_bikes)}")
print(f"{limit} {category} bikes from {year} saved to {file_name}")


# In[ ]:




