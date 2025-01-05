# %%
import numpy as np
from collections import defaultdict
import re
import os
import sys
import heapq

sys.path.append("/home/dekorvyb/Documents/criticality_pipeline/")
import tools


spike_directory = '/cns/share/Vincent/McHugh_lab/CA2Cre_DREADD/extracted_spike_trains/'
result_directory = '/cns/share/Vincent/McHugh_lab/CA2Cre_DREADD/mr_estimator_results/'

def group_files_by_pattern():
    '''
    creates a dictionary where the keys are the filenames, but with the tetrode numbers removed.
    So all tetrodes, that recorded the same area in the same trial can be summed.
    Also converts all subareas of CA3 to just CA3.
    '''
    grouped_files = defaultdict(list)
    pattern = re.compile(r"^(.*?_tt)\d+(_.*)$")  # Regex to match prefix and suffix excluding `tt`
    
    for filename in os.listdir(spike_directory):
        
        if filename.split("_")[4] in ["CA3a", "CA3b", "CA3c", "CA13b"]: # CA13b is a typo fix.
            key_name = filename.replace(filename.split("_")[4], "CA3")
        
        else:
            key_name = filename
        
        match = pattern.match(key_name)
        if match:
            prefix, suffix = match.groups()
            group_key = f"{prefix[:-3]}{suffix}"
            grouped_files[group_key].append(filename)
    return grouped_files


def merge_activity(filename_list):
    spike_trains = []
    
    for file in filename_list:
        path = f'{spike_directory}{file}'
        spike_trains.append(np.load(path))
    
    return np.array(list(heapq.merge(*spike_trains)))



tetrode_dict = group_files_by_pattern()

for trial_area in tetrode_dict.keys():
    try:
        timestamps = merge_activity(tetrode_dict[trial_area])
        binned = tools.bin_timestamps(timestamps, unit = "us") # data unit according to https://neuralynx.com/_software/NeuralynxDataFileFormats.pdf
        tools.mr_estimator_for_numpy(binned, window_size = 90, bin_size = 5, filename = f'{result_directory}/{trial_area[:-4]}', fit_func = "f_complex")
        
    except Exception as e:
        with open('/cns/share/Vincent/McHugh_lab/CA2Cre_DREADD/failed_analysis.txt', 'a') as f:
            f.write(f"{trial_area}: {e}\n")
        continue
    

# %%
# %%
'''
# %%
####################################
# Create linear Model
####################################

import os
import pandas as pd
import statsmodels.formula.api as smf

filepath = "/cns/share/Vincent/McHugh_lab/analysis_results/test/"
results = [f for f in os.listdir(filepath)]

big_df = []

for result in results:
    try:
        # Create a new DataFrame for each result file
        df = pd.DataFrame()

        # Extract the area (e.g., "CA1") from the filename
        area = result.split("_")[3] 
        state = result.split("_")[2][:-1]
        cno = result.split("_")[-1][:-11]

        # Read the "beta" column from the parquet file
        tau = pd.read_parquet(f"{filepath}{result}")["tau"].values[0]
        
        file_data = {
                    "tau": tau,
                    "area": str(area),
                    "state": str(state),
                    "cno": str(cno)
                }
        big_df.append(file_data)
        
    except:
        continue
    
    
    df["tau"] = tau
    
    # Add the 'area' as a new column
    df["area"] = str(area)

    print(df)
    # Append this DataFrame to the list
    big_df.append(df)
    
    
combined_df = pd.DataFrame(big_df)
    
combined_df = combined_df.dropna()
print(combined_df)

formula = "tau ~ area * state * cno"

# Fit the linear model using Ordinary Least Squares (OLS)
model = smf.ols(formula=formula, data=combined_df).fit()

# Print the summary of the model
print(model.summary())


'''
# %%













