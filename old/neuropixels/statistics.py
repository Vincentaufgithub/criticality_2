# %%

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.mixed_linear_model import MixedLM


#---------------------------------------------------------------------------------#

def read_mr_estimator_results(result_directory, output_directory = None):
    """made output_directory optional. But highly recommended since integration of all data may take some time

    Args:
        result_directory (str): _description_
        output_directory (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    files = [f for f in os.listdir(result_directory) if f.endswith(".parquet")]

    all_data = []
    for filename in files:
        filename_split = filename.split('_') # with filename pattern being:  "animal name _ experimental day _ task _ tetrode nr. _ brain area _ pre or post treatment _ timechunk .parquet"
        
        df = pd.read_parquet(f'{result_directory}/{filename}')
        df['tau'] = df['tau'][0]
        df['session'] = filename_split[0]
        df['area'] = filename_split [1]
        df ['chunk'] = filename_split[2].split(' ')[0]
        all_data.append(df)

    # Combine all data
    df = pd.concat(all_data, ignore_index=True)

    # Convert categorical variables
    df['session'] = df['session'].astype('category')
    
    if output_directory:
        df.to_parquet(f'{output_directory}big_df_neuropixels.parquet')
    
    return df

def tau_for_outer_vs_inner_violin(df, output_directory: str = None, add_filename: str = None, log_scale: bool = False):

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
        hue='arm',
        bw=0.2,  # Reduce smoothing
        cut=0,   # Restrict violins to data range
        split=True,
        palette='muted'
    )

    # Add labels and title
    plt.title('Tau Distributions by Brain Area and arm in W-maze(Filtered at 3000)', fontsize=16)
    plt.xlabel('Brain Area', fontsize=14)
    plt.ylabel('Tau', fontsize=14)
    plt.legend(title='arm')
    plt.tight_layout()

    if log_scale:
        plt.yscale('log')
    
    if output_directory:
        plt.savefig(f"{output_directory}violin_out-inbound_areas_{add_filename}.png")
        
    # Show plot
    plt.show()


def tau_for_outer_vs_inner_violin_2(df, output_directory: str = None, add_filename: str = None, log_scale: bool = False):

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
        hue='arm',
        cut=0,   # Restrict violins to data range
        dodge=True,
        palette='muted',
        inner = 'quart'
    )

    # Add labels and title
    plt.title('Tau Distributions by Brain Area and arm in W-maze(Filtered at 3000)', fontsize=16)
    plt.xlabel('Brain Area', fontsize=14)
    plt.ylabel('Tau', fontsize=14)
    plt.legend(title='arm')
    plt.tight_layout()

    if log_scale:
        plt.yscale('log')
    
    if output_directory:
        plt.savefig(f"{output_directory}violin_out-inbound_areas_{add_filename}.png")
        
    # Show plot
    plt.show()


def tau_interaction_plot(df, output_directory: str = None, add_filename: str = None):
    # Set plot style
    sns.set(style="whitegrid")
    
    df = df[df['tau'] <= 3000]

    area_order = ["CA1", "CA2", "CA3", "DG", "SUB", "ProS"]
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    ax = sns.pointplot(
        data=df,
        x='area',
        y='tau',
        capsize=0.2,  # Size of error bars
        dodge=True,  # Offset points slightly to avoid overlap
        errwidth=1,  # Width of error bars
        linestyle="-",
        markers="o",
        order=area_order
    )

    # Add labels and title
    plt.title("Tau Plot", fontsize=14)
    plt.xlabel("Brain Area", fontsize=12)
    plt.ylabel("Tau", fontsize=12)
    
    ymin, ymax = ax.get_ylim()  # Get current limits
    plt.ylim(ymin * 0.9, ymax * 1.1)

    plt.tight_layout()

    # Save if output directory is provided
    if output_directory:
        filename = f"{output_directory}interaction_tau_{add_filename}.png"
        plt.savefig(filename)

    # Show plot
    plt.show()
    
#----------------------------------------------------------------------------#

#df = read_mr_estimator_results('/groups/cns/share/Vincent/loren_frank/neuro_pixels_output_f_expo/', '/groups/cns/dekorvyb/')
df = pd.read_parquet('/groups/cns/dekorvyb/big_df_neuropixels.parquet')

df = df[df['tau'] <= 3000]


#model = MixedLM.from_formula("tau ~ arm * area", groups="animal", data=df)

#result = model.fit()

# Summarize Results
# print(result.summary())


# tau_for_outer_vs_inner_violin_2(df, '/groups/cns/dekorvyb/', '8_animals_quart', log_scale=True)
tau_interaction_plot(df, '/groups/cns/dekorvyb/', 'neuropixels_tau_allareas')


# %%