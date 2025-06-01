import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd

# create empty excel file
results_excel = pd.ExcelWriter("predicts_final_results.xlsx", engine='openpyxl')

# define the path of the results
predicts_results_folder_path = os.path.join("outputs", "predicts")

final_results = []

# get all folders in the predicts_results_folder_path
for folder_data_name in os.listdir(predicts_results_folder_path):
    # get the path of the folder
    folder_data_name_path = os.path.join(predicts_results_folder_path, folder_data_name)
    
    rows_results = []
    
    # get all folder_model_name in the folder_data_name
    for folder_model_name in os.listdir(folder_data_name_path):
        # get the path of the file
        folder_model_name_path = os.path.join(folder_data_name_path, folder_model_name)

        for file_name in os.listdir(folder_model_name_path):
            # check if the file is metrics.csv
            if "metrics" not in file_name:
                continue
            # get the path of the file
            file_name_path = os.path.join(folder_model_name_path, file_name)
            
            # read the file
            result_model_df = pd.read_csv(file_name_path)
            
            model_results = {
                "model": folder_model_name,
                "data": folder_data_name,
                "Similarity": result_model_df["Similarity"].values[0],
                "NMAE": result_model_df["NMAE"].values[0],
                "RMSE": result_model_df["RMSE"].values[0],
                "R2": result_model_df["R2"].values[0],
                "FSD": result_model_df["FSD"].values[0],
                "FB": result_model_df["FB"].values[0],
                "FA2": result_model_df["FA2"].values[0]
            }
            rows_results.append(model_results)
    
    # create new sheet in the excel file
    df = pd.DataFrame(rows_results)
    df.to_excel(results_excel, sheet_name=folder_data_name, index=False)

    # append the results to the final_results
    final_results.extend(rows_results)
    # add empty row
    final_results.append({})

# create new sheet in the excel file for all the results in the first sheet
df = pd.DataFrame(final_results)
df.to_excel(results_excel, sheet_name="All", index=False)

# Use close() instead of save()
results_excel.close()