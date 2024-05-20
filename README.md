# MSc_Project_2024
Optimising Bicycle Selection: Machine Learning Approaches to Frame Geometry Analysis

Very Important: 
1. The scripts contained in this repository are .pynb file as the project was developed using Jupyter Notebooks.
2. I am not sure if the visualisations will run in another another IDE like VS Code but I have still chosen to include them. 

Instructions on how to use python scripts:

1. The primary file required is 'all_road_bikes_2020-2024.json'
2. This file is needed as the script 'api_request_v8' will not work as the API is stored on my terminal show of my personal device for security reasons.
3. Save this file to you personal device if you wish to interact with the remaining scripts.

Order of Operations:
1. Open the 'convert_json_to_csv_v7' file.
2. Run all the cells in this file. The final output is a csv file 'all_road_bikes_2020-2024.csv' (this file is also included if you are unable to run this properly).
3. This new csv file should be saved to your device. 
4. Open the 'Normalisation_v6' script. Run all the cells in this script.
5. The output for this script will be a new csv file called 'encoded_road_bikes_2020-2024.csv.
6. Open the 'sampling_v1' script and run the first cell to save a new csv file with only sample data that will be used in the 'random_forest_v2' script
7. Open the 'random_forest_v2' script and run all the cells in the script. (DO NOT RUN THE GRID SEARCH CV FILE AS THIS WILL RUN AN OPTIMISATION FUNCTION THAT IS COMPUTATIONALLY INTENSIVE)
8. The results of the random forest should now display.
9. Open the 'k_means_clustering_alg_v2' script and run all the cells in the script.
10. This will show you how the bikes are clustered.
11. Now you have seen the results for both the unsupervised and supervised models.

Other files to be aware of:
1. The 'Data_Exploration_v1' script is used to explore key attributes about the dataset like the distribution, missing values and elbow method for determining optimal clusters for the k-means clustering algorrithm.
2. The 'visualisation_v1' script can be used to evaluate some additional distribution statistics about the data like mean, median, min, max and standard deviation values.
3. The 'ratios_v3' script is for viewing just the ratios used in the 'Normalisation_v6' script.

Closing Remarks:

If anyone has any adivce on how to improve these models or would like to discuss any of the scripts and findings with me, please reach out as I would love to discuss and chat to people also interested in this space as I am still new to programming and looking to learn. 

- Alex
