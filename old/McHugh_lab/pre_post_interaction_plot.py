# %%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLM



def tau_interaction_plot(df, output_directory: str = None, add_filename: str = None):
    # Set plot style
    sns.set(style="whitegrid")
    
    df = df[df['tau'] <= 3000]

    # Create the plot
    plt.figure(figsize=(8, 6))
    ax = sns.pointplot(
        data=df,
        x='area',
        y='tau',
        hue='treatment',
        capsize=0.2,  # Size of error bars
        dodge=True,  # Offset points slightly to avoid overlap
        markers=["o", "s"],  # Different markers for wake/sleep
        linestyles=["-", "--"],  # Solid and dashed lines for differentiation
        errwidth=1,  # Width of error bars
        palette="muted"  # Use a distinguishable color scheme
    )

    # Add labels and title
    plt.title("Tau Interaction Plot: Area vs. treatment", fontsize=14)
    plt.xlabel("Brain Area", fontsize=12)
    plt.ylabel("Tau", fontsize=12)
    plt.legend(title="arm", fontsize=10)
    
    ymin, ymax = ax.get_ylim()  # Get current limits
    plt.ylim(ymin * 0.9, ymax * 1.1)

    plt.tight_layout()

    # Save if output directory is provided
    if output_directory:
        filename = f"{output_directory}interaction_tau_{add_filename}.png"
        plt.savefig(filename)

    # Show plot
    plt.show()
    
    
df = pd.read_parquet("/groups/cns/share/Vincent/McHugh_lab/CA2Cre_DREADD/models_and_figures/big_df.parquet")

df = df[df['tau'] <= 3000]

selected_areas = ["CA1", "CA2", "CA3"]  # Replace with your desired values
df = df[df["area"].isin(selected_areas)]

df['area'] = df['area'].cat.remove_unused_categories()

#df = df[df['task'].str.startswith("lt")]
#df = df[df['task'].str.endswith("1" or "2")]

#tau_interaction_plot(df, '/groups/cns/dekorvyb/', 'mchugh_prepost_interaction_lt')

model = MixedLM.from_formula("tau ~ treatment*area", 
                             groups="animal", 
                             data=df)

result = model.fit()
# %%