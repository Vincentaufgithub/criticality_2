# %%

import pandas as pd
import os
from statsmodels.regression.mixed_linear_model import MixedLM


criticality_dict = '/cns/share/Vincent/McHugh_lab/CA2Cre_DREADD/mr_estimator_results/'
files = [f for f in os.listdir(criticality_dict) if f.endswith(".parquet")]

all_data = []


for filename in files:
    filename_split = filename.split('_') # with filename pattern being:  "animal name _ experimental day _ task _ tetrode nr. _ brain area _ pre or post treatment _ timechunk .parquet"
    
    df = pd.read_parquet(f'{criticality_dict}/{filename}')
    df['tau'] = df['tau'][0]
    df['animal'] = filename_split[0]
    df['day'] = filename_split [1]
    df ['task'] = filename_split[2]
    df['area'] = filename_split[3]
    df['treatment'] = filename_split[4][:-4]
    df['timechunk'] = int(filename_split[5].split('.')[0])
    all_data.append(df)


# Combine all data
df = pd.concat(all_data, ignore_index=True)
print('combined all data')


# Convert categorical variables
df['area'] = df['area'].astype('category')
df['treatment'] = df['treatment'].astype('category')
df['animal'] = df['animal'].astype('category')
df['task'] = df['task'].astype('category')

df.to_parquet('/cns/share/Vincent/McHugh_lab/CA2Cre_DREADD/models_and_figures/big_df.parquet')

# %%

df = pd.read_parquet("/cns/share/Vincent/McHugh_lab/CA2Cre_DREADD/result_df")

# Rename subareas to broader categories
df['area'] = df['area'].replace({
    'CA3a': 'CA3',
    'CA3b': 'CA3',
    'CA3c': 'CA3',
    'CA13b': 'CA3',
    
    })


print("fitting model")
# 3. Fit the LME Model
# Formula: tau ~ treatment * area + (1|animal)
model = MixedLM.from_formula("tau ~ treatment * area", groups="animal", data=df)
result = model.fit()

# 4. Summarize Resultsd
print(result.summary())

# %%

#######################################
#######################################
# These are the animal-day combinations which perfectly fulfill the protocol (sleep - LT1 - CNO - LT1 - sleep; or so)
# 161: d12
# 368: d16
# 550: d17 & d21
# 556: d13 & 19
# 572: 17, 18
# 1223: d12, d13, d14 & d15
# 1232: 13, 14, 15
# 1216: 10, 12


# %%
# create violin plot

import seaborn as sns
import matplotlib.pyplot as plt

# Set plot style
sns.set(style="whitegrid")

# Filter data to include only tau values <= 3000
filtered_df = df[df['tau'] <= 3000]

# Create the violin plot with reduced smoothing
plt.figure(figsize=(10, 6))
sns.violinplot(
    data=filtered_df,
    x='area',
    y='tau',
    hue='treatment',
    bw=0.2,  # Reduce smoothing
    cut=0,   # Restrict violins to data range
    split=True,
    palette='muted'
)

# Add labels and title
plt.title('Tau Distributions by Brain Area and Treatment (Filtered at 3000)', fontsize=16)
plt.xlabel('Brain Area', fontsize=14)
plt.ylabel('Tau', fontsize=14)
plt.legend(title='Treatment')
plt.tight_layout()

# Show plot
plt.show()











# %%
import pandas as pd

# filtering the dataframe
# ===========================================================================

df_full = pd.read_parquet("/cns/share/Vincent/McHugh_lab/CA2Cre_DREADD/result_df")

# Rename subareas to broader categories
df_full['area'] = df_full['area'].replace({
    'CA3a': 'CA3',
    'CA3b': 'CA3',
    'CA3c': 'CA3',
    'CA13b': 'CA3',
    
    })

# ensure are categorical
df_full['area'] = df_full['area'].astype('category')
df_full['treatment'] = df_full['treatment'].astype('category')
df_full['animal'] = df_full['animal'].astype('category')


combinations = {
    "model161": ["d12"],
    "model368": ["d16"],
    "model550": ["d17", "d21"],
    "model556": ["d13", "d19"],
    "model572": ["d17", "d18"],
    "model1223": ["d12", "d13", "d14", "d15"],
    "model1232": ["d13", "d14", "d15"],
    "model1216": ["d10", "d12"]
}

# Filter rows based on the combinations
filtered_df = df_full[df_full.apply(lambda row: row['day'] in combinations.get(row['animal'], []), axis=1)]

# Further filter rows where 'task' starts with "lt"
df = filtered_df[filtered_df['task'].str.startswith("lt")]

# remove animals with no observations
df = df[df['animal'].map(df['animal'].value_counts()) > 1]

# remove outliers
df = df[df['tau'] <= 3000]

# List of areas to remove
areas_to_remove = ['DEEP', 'CA2-CA1', 'CA3-CA2']

# Filter the DataFrame to exclude rows with these areas
df = df[~df['area'].isin(areas_to_remove)]
df['area'] = df['area'].cat.remove_unused_categories()


# Display the updated DataFrame
print(df['area'].value_counts())
# Display the resulting DataFrame
print(df)
















# %%
import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLM

# filtering the dataframe

df_full = pd.read_parquet("/cns/share/Vincent/McHugh_lab/CA2Cre_DREADD/result_df")

# Rename subareas to broader categories
df_full['area'] = df_full['area'].replace({
    'CA3a': 'CA3',
    'CA3b': 'CA3',
    'CA3c': 'CA3',
    'CA13b': 'CA3',
    
    })

# ensure are categorical
df_full['area'] = df_full['area'].astype('category')
df_full['treatment'] = df_full['treatment'].astype('category')
df_full['animal'] = df_full['animal'].astype('category')

df_full['task'] = df_full['task'].str.replace(r'\d$', '', regex=True)
df_full['task'] = df_full['task'].str.replace(r'\d$', '', regex=True)
df_full['task'] = df_full['task'].replace("cno?", "cno")

df_full = df_full[df_full['task'] != "hab"]

df_full['task'] = df_full['task'].replace("cno?", "cno")
df_full['task'] = df_full['task'].replace("boxC", "box")
df_full['task'] = df_full['task'].replace("boxA", "box")
df_full['task'] = df_full['task'].replace("boxB", "box")

df_full['task'] = df_full['task'].astype('category')
df_full['task'] = df_full['task'].astype('category')

areas_to_remove = ['CA2-CA1', 'CA3-CA2']

# Filter the DataFrame to exclude rows with these areas
df_full = df_full[~df_full['area'].isin(areas_to_remove)]
df['area'] = df['area'].cat.remove_unused_categories()

model = MixedLM.from_formula("tau ~ treatment * area * task", groups="animal", data=df_full)
result = model.fit()
print(result.summary())

# %%




















# %%



# Investigate correlation between activity of brain area and criticality in the same area
# =====================================================================

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

# Directory containing the result files
output_directory = "/cns/share/Vincent/McHugh_lab/CA2Cre_DREADD/models_and_figures/"

# Initialize a list to store all dataframes for the combined plot
df = pd.read_parquet("/cns/share/Vincent/McHugh_lab/CA2Cre_DREADD/models_and_figures/big_df.parquet")

df['area'] = df['area'].replace({
    'CA3a': 'CA3',
    'CA3b': 'CA3',
    'CA3c': 'CA3',
    'CA13b': 'CA3',
    
    })


areas_to_remove = ['CA2-CA1', 'CA3-CA2']

# Filter the DataFrame to exclude rows with these areas
df = df[~df['area'].isin(areas_to_remove)]
df['area'] = df['area'].cat.remove_unused_categories()

# remove outliers
df = df[df['tau'] <= 3000]


unique_areas = df['area'].unique()


# Create scatter plots for each area with regression lines
for area in unique_areas:
    plt.figure(figsize=(8, 6))
    
    # Filter data for the specific brain area
    area_data = df[df['area'] == area]
    
    # Fit a regression line
    x = area_data['activity']
    y = area_data['tau']
    if len(x) > 1:  # Ensure enough data points for regression
        x_with_const = sm.add_constant(x)
        model = sm.OLS(y, x_with_const).fit()
        slope = model.params['activity']
        intercept = model.params['const']
        label = f"y = {intercept:.2f} + {slope:.2f}x\nn = {len(x)}"
    else:
        slope, intercept = np.nan, np.nan
        label = f"Insufficient data (n={len(x)})"
    
    # Scatter plot
    sns.scatterplot(data=area_data, x='activity', y='tau', alpha=0.7, label="Data points")
    
    # Add regression line if valid
    if not np.isnan(slope):
        plt.plot(x, intercept + slope * x, color='red', label=f"Regression: {label}")
    
    # Add titles and labels
    plt.title(f"Scatter Plot with Regression for {area}", fontsize=16)
    plt.xlabel("Activity", fontsize=14)
    plt.ylabel("Tau", fontsize=14)
    plt.legend(fontsize=12)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{output_directory}scatter_with_regression_{area}.png")
    plt.close()

# Create combined scatter plot with regression lines for all areas
plt.figure(figsize=(10, 8))

# Scatter plot
sns.scatterplot(data=df, x='activity', y='tau', hue='area', alpha=0.7, palette='muted', legend="full")

# Titles and labels
plt.title("Combined Scatter Plot for All Brain Areas", fontsize=16)
plt.xlabel("Activity", fontsize=14)
plt.ylabel("Tau", fontsize=14)

# Add annotations for each area
for area in unique_areas:
    area_data = df[df['area'] == area]
    x = area_data['activity']
    y = area_data['tau']
    if len(x) > 1:
        x_with_const = sm.add_constant(x)
        model = sm.OLS(y, x_with_const).fit()
        slope = model.params['activity']
        intercept = model.params['const']
        plt.plot(x, intercept + slope * x, label=f"{area}: y = {intercept:.2f} + {slope:.2f}x\nn = {len(x)}")
    else:
        plt.annotate(f"{area} (n={len(x)})", xy=(x.mean(), y.mean()), fontsize=10)

# Adjust legend
plt.legend(title="Brain Area", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

# Save the combined plot
plt.tight_layout()
plt.savefig(f"{output_directory}scatter_combined_with_regression.png")
plt.show()















# %%

# correlation between tau and activity in CA2 (which may be turned on or off)
# ==============================================================================
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
import numpy as np

df = pd.read_parquet("/cns/share/Vincent/McHugh_lab/CA2Cre_DREADD/models_and_figures/big_df.parquet")

df['area'] = df['area'].replace({
    'CA3a': 'CA3',
    'CA3b': 'CA3',
    'CA3c': 'CA3',
    'CA13b': 'CA3',
    
    })


areas_to_remove = ['CA2-CA1', 'CA3-CA2']

# Filter the DataFrame to exclude rows with these areas
df = df[~df['area'].isin(areas_to_remove)]
df['area'] = df['area'].cat.remove_unused_categories()

# remove outliers
df = df[df['tau'] <= 3000]


# Filter CA2 data and rename 'activity' for clarity
ca2_data = df[df['area'] == "CA2"].rename(columns={'activity': 'activity_CA2'})

# Merge CA2 activity with the original dataframe
merged_df = pd.merge(
    df,
    ca2_data[['animal', 'day', 'task', 'timechunk', 'activity_CA2']],
    on=['animal', 'day', 'task', 'timechunk'],
    how='inner'
)

# Exclude rows where area is CA2 (since we're comparing other areas to CA2)
comparison_df = merged_df[merged_df['area'] != "CA2"]

# Directory to save plots
output_directory = "/cns/share/Vincent/McHugh_lab/CA2Cre_DREADD/models_and_figures/"

# Get unique areas to create scatter plots
unique_areas = comparison_df['area'].unique()

# Create scatter plots for each area
for area in unique_areas:
    plt.figure(figsize=(8, 6))
    
    # Filter data for the specific brain area
    area_data = comparison_df[comparison_df['area'] == area]
    
    # Fit a regression line
    x = area_data['activity_CA2']
    y = area_data['tau']
    if len(x) > 1:  # Ensure enough data points for regression
        x_with_const = sm.add_constant(x)
        model = sm.OLS(y, x_with_const).fit()
        slope = model.params['activity_CA2']
        intercept = model.params['const']
        label = f"y = {intercept:.2f} + {slope:.2f}x\nn = {len(x)}"
    else:
        slope, intercept = np.nan, np.nan
        label = f"Insufficient data (n={len(x)})"
    
    # Scatter plot
    sns.scatterplot(data=area_data, x='activity_CA2', y='tau', alpha=0.7, label="Data points")
    
    # Add regression line if valid
    if not np.isnan(slope):
        plt.plot(x, intercept + slope * x, color='red', label=f"Regression: {label}")
    
    # Add titles and labels
    plt.title(f"Scatter Plot Comparing {area} with CA2", fontsize=16)
    plt.xlabel("Activity in CA2", fontsize=14)
    plt.ylabel(f"Tau in {area}", fontsize=14)
    plt.legend(fontsize=12)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{output_directory}scatter_comparison_CA2_{area}.png")
    plt.close()


# %%