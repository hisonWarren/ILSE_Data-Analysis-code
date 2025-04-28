import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
import statsmodels.stats.multitest as multitest
import os
import matplotlib
import statsmodels

# Create output directory
output_dir = "new_result/correlation_analysis"
os.makedirs(output_dir, exist_ok=True)
print(f"Created output directory: {output_dir}")

# Add software version information
print(f"pandas version: {pd.__version__}")
print(f"numpy version: {np.__version__}")
print(f"seaborn version: {sns.__version__}")
print(f"matplotlib version: {matplotlib.__version__}")
print(f"pingouin version: {pg.__version__}")
print(f"statsmodels version: {statsmodels.__version__}")

# Variable name mapping dictionary
var_mapping = {
    'T3TMA': 'TMT-A',             # Trail Making Test A
    'T3TMB': 'TMT-B',             # Trail Making Test B
    'T3EGU12logicalmemory': 'LM I',    # Logical Memory I
    'T3LGV12logicalmemory': 'LM II',   # Logical Memory II
    'T4TMA': 'TMT-A (T4)',        # Trail Making Test A at T4
    'T4TMB': 'TMT-B (T4)',        # Trail Making Test B at T4
    'T4EGU12logicalmemory': 'LM I (T4)',   # Logical Memory I at T4
    'T4LGV12logicalmemory': 'LM II (T4)',  # Logical Memory II at T4
    'T3ZNVORdigitspanforward': 'DSF',  # Digit Span Forward
    'T3ZNNACHdigitspanbackward': 'DSB', # Digit Span Backward
    'T4ZNVORdigitspanforward': 'DSF (T4)', # Digit Span Forward at T4
    'T4ZNNACHdigitspanbackward': 'DSB (T4)', # Digit Span Backward at T4
    'T4MMSE': 'MMSE',             # Mini-Mental State Examination at T4
    'T3Component1': 'IC1',        # Independent Component 1
    'T3Component2': 'IC2',        # Independent Component 2
    'T3Component3': 'IC3',        # Independent Component 3
    'T3Component4': 'IC4',        # Independent Component 4
    'T3Component5': 'IC5',        # Independent Component 5
    'T3Component6': 'IC6',        # Independent Component 6
    'T3Component7': 'IC7',        # Independent Component 7
    'T3Component8': 'IC8',        # Independent Component 8
    'T3Component9': 'IC9',        # Independent Component 9
    'T3Component10': 'IC10',      # Independent Component 10
    'T3Component11': 'IC11',      # Independent Component 11
    'T3Component12': 'IC12',      # Independent Component 12
    'APOE': 'APOE ε4 Status',     # APOE ε4 carrier status
    'SEX': 'Sex',                 # Sex
    'T3Age': 'Age',               # Age at T3
    'EDUC': 'Years of Education', # Years of education
    'TIV': 'Total Intracranial Volume', # Total intracranial volume
    'Group': 'APOE ε4 Status'     # Group is also APOE status
}

# Function to get display name
def get_display_name(variable):
    """Get the display name for a variable"""
    return var_mapping.get(variable, variable)

# Read local Excel file
file_path = r'D:\ILSE-data-20241126-审稿中\Data-Apoe\data_pre.xlsx'  # Update with your file path
sheet_name = 'Young'  # Replace with your worksheet name
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Add outlier detection
def detect_outliers(data, columns, threshold=3):
    """
    Detect outliers using Z-scores
    
    Parameters:
    data (DataFrame): DataFrame containing the data
    columns (list): List of column names to check
    threshold (float): Z-score threshold, default is 3
    
    Returns:
    DataFrame: DataFrame containing outlier information
    """
    outliers_info = []
    for col in columns:
        z_scores = np.abs((data[col] - np.mean(data[col])) / np.std(data[col]))
        outliers = data[z_scores > threshold].index.tolist()
        if outliers:
            outliers_info.append({
                'column': col,
                'display_name': get_display_name(col),
                'outlier_indices': outliers,
                'outlier_values': data.loc[outliers, col].tolist(),
                'z_scores': z_scores[outliers].tolist()
            })
    return pd.DataFrame(outliers_info)

# Select needed variable columns, excluding covariates
variables = [
    'T3TMA', 'T3TMB', 'T3EGU12logicalmemory', 'T3LGV12logicalmemory','T3ZNVORdigitspanforward','T3ZNNACHdigitspanbackward',
    'T3Component1', 'T3Component2', 'T3Component3', 'T3Component4',
    'T3Component5', 'T3Component6', 'T3Component7', 'T3Component8',
    'T3Component9', 'T3Component10', 'T3Component11', 'T3Component12'
]

# Get display variable names
display_variables = [get_display_name(var) for var in variables]

# Detect outliers
outliers_report = detect_outliers(df, variables)
outliers_report.to_csv(os.path.join(output_dir, 'outliers_report.csv'), index=False)
print(f"Saved outlier report to {output_dir}/outliers_report.csv")

# Define covariates
covariates = ['SEX', 'T3Age', 'EDUC', 'Group', 'TIV']
display_covariates = [get_display_name(cov) for cov in covariates]

# Initialize correlation matrix and p-value matrix
corr_matrix = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=display_variables, index=display_variables)
p_matrix = pd.DataFrame(np.ones((len(variables), len(variables))), columns=display_variables, index=display_variables)

# Calculate partial correlation coefficients and p-values
results = []
for i in range(len(variables)):
    for j in range(i, len(variables)):
        if i != j:  # Skip when comparing the same variable
            # Use pingouin for partial correlation analysis with correct parameters
            pcorr = pg.partial_corr(
                data=df, 
                x=variables[i], 
                y=variables[j], 
                covar=covariates, 
                method='spearman'  # Use Spearman correlation coefficient
            )
            corr = pcorr['r'].values[0]
            p_value = pcorr['p-val'].values[0]
            corr_matrix.iloc[i, j] = corr
            corr_matrix.iloc[j, i] = corr
            p_matrix.iloc[i, j] = p_value
            p_matrix.iloc[j, i] = p_value
            
            # Collect more detailed results
            results.append({
                "Variable": variables[i],
                "Component": variables[j],
                "Display_Variable": get_display_name(variables[i]),
                "Display_Component": get_display_name(variables[j]),
                "Method": "Partial Spearman",
                "Correlation": corr,
                "p-value": p_value,
                "adjusted p-value": None,  # Leave empty for now, fill later
                "sample_size": len(df),    # Add sample size information
                "covariates": ", ".join(display_covariates)  # Add covariate information
            })

# Set diagonal correlation coefficients to 1
np.fill_diagonal(corr_matrix.values, 1)

# Perform multiple corrections, explicitly specify FDR method
print("Performing Benjamini-Hochberg multiple corrections...")
p_values_flat = p_matrix.values.flatten()
p_values_flat = p_values_flat[~np.isnan(p_values_flat)]  # Remove NaN values
_, p_values_corrected, _, _ = multitest.multipletests(
    p_values_flat, 
    method='fdr_bh',  # Use Benjamini-Hochberg method
    alpha=0.05,       # Significance level
    returnsorted=False # Maintain original order
)

# Reshape corrected p-values back to matrix
p_matrix_corrected = pd.DataFrame(p_values_corrected.reshape(p_matrix.shape), columns=display_variables, index=display_variables)

# Update multiple-correction p-values to results
for result in results:
    var1_idx = variables.index(result["Variable"])
    var2_idx = variables.index(result["Component"])
    result["adjusted p-value"] = p_matrix_corrected.iloc[var1_idx, var2_idx]

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results as CSV file
results_df.to_csv(os.path.join(output_dir, 'partial_correlation_results_long_format.csv'), index=False)
print(f"Saved complete correlation analysis results to {output_dir}/partial_correlation_results_long_format.csv")

# Initialize annotated_corr_matrix with string type
annotated_corr_matrix = pd.DataFrame(index=display_variables, columns=display_variables, dtype=str)

# Mark significance and format correlation matrix
for i in range(len(variables)):
    for j in range(len(variables)):
        if i == j:
            annotated_corr_matrix.iloc[i, j] = "1.00"  # Diagonal
        elif j > i:
            annotated_corr_matrix.iloc[i, j] = ""  # Keep upper triangle empty
        else:  # j < i, lower triangle
            if p_matrix.iloc[i, j] < 0.05:
                significance = '*'
                if p_matrix.iloc[i, j] < 0.01:
                    significance = '**'
                if p_matrix.iloc[i, j] < 0.001:
                    significance = '***'
                annotated_corr_matrix.iloc[i, j] = f"{corr_matrix.iloc[i, j]:.2f}{significance}"
                if p_matrix_corrected.iloc[i, j] < 0.05:
                    annotated_corr_matrix.iloc[i, j] += '+'
            else:
                annotated_corr_matrix.iloc[i, j] = f"{corr_matrix.iloc[i, j]:.2f}"

# Save the annotated matrix to a CSV file
annotated_corr_matrix.to_csv(os.path.join(output_dir, 'annotated_partial_corr_matrix.csv'))
print(f"Saved annotated correlation matrix to {output_dir}/annotated_partial_corr_matrix.csv")

# Save correlation analysis methodology parameters
methods_info = {
    "Analysis Method": ["Partial Spearman Correlation Analysis"],
    "Sample Size": [len(df)],
    "Covariates": [", ".join(display_covariates)],
    "Significance Level": [0.05],
    "Multiple Comparison Correction Method": ["Benjamini-Hochberg (FDR)"],
    "Normality Test": ["Not required, Spearman is a non-parametric method"],
    "Outlier Threshold": [3]
}
pd.DataFrame(methods_info).to_csv(os.path.join(output_dir, 'correlation_methods_parameters.csv'), index=False)
print(f"Saved methodology parameters to {output_dir}/correlation_methods_parameters.csv")

# Set chart style
sns.set(style="white")
plt.figure(figsize=(14, 12))

# Use display names to create heatmap mask (show only lower triangle)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Use better color scheme (coolwarm)
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Increase font size for better readability
plt.rcParams.update({'font.size': 14})

# Create heatmap
heatmap = sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, annot=True, fmt='.2f',
            cbar_kws={"shrink": .5})

# Increase annotation font size
for t in heatmap.texts:
    t.set_fontsize(12)

# Set title and axis labels with larger font
plt.title('Partial Spearman Correlation Matrix of Variables\nControling for Sex, Age, Years of Education, APOE ε4 Status, and Total Intracranial Volume', fontsize=18)

# Save image with high resolution
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'partial_spearman_correlation_matrix.png'), dpi=600, bbox_inches='tight')
print(f"Saved correlation matrix heatmap to {output_dir}/partial_spearman_correlation_matrix.png")

# Close figure
plt.close()

# For better visualization, create a dedicated significance heatmap
plt.figure(figsize=(14, 12))
plt.rcParams.update({'font.size': 14})

# Create significance matrix (1: p<0.05, 2: p<0.01, 3: p<0.001, 4: FDR corrected significant)
sig_matrix = np.zeros_like(corr_matrix.values)
for i in range(len(variables)):
    for j in range(len(variables)):
        if i != j:  # Skip diagonal
            if p_matrix.iloc[i, j] < 0.05:
                sig_matrix[i, j] = 1
            if p_matrix.iloc[i, j] < 0.01:
                sig_matrix[i, j] = 2
            if p_matrix.iloc[i, j] < 0.001:
                sig_matrix[i, j] = 3
            if p_matrix_corrected.iloc[i, j] < 0.05:
                sig_matrix[i, j] = 4

# Convert to DataFrame and use mapped variable names
sig_df = pd.DataFrame(sig_matrix, index=display_variables, columns=display_variables)

# Use custom cmap to show different significance levels
cmap = sns.color_palette(["white", "lightblue", "royalblue", "darkblue", "purple"])

# Plot significance heatmap
sig_heatmap = sns.heatmap(sig_df, cmap=cmap, cbar_kws={"shrink": .5, "ticks": [0, 1, 2, 3, 4]})
plt.title('Significance Levels of Partial Correlations', fontsize=18)
cbar = plt.gcf().axes[-1] 
cbar.set_yticklabels(['Not significant', 'p < 0.05', 'p < 0.01', 'p < 0.001', 'FDR corrected'], fontsize=12)

# Increase axis label font sizes
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Save image
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation_significance_matrix.png'), dpi=600, bbox_inches='tight')
print(f"Saved correlation significance matrix image to {output_dir}/correlation_significance_matrix.png")

# Create README.md file to explain output files
readme_content = """# Correlation Analysis Results

This folder contains all output result files from the correlation analysis.

## Data Processing and Outliers

- `outliers_report.csv`: Report of outliers detected using Z-score method, including outlier locations, values, and corresponding Z-scores

## Correlation Analysis Results

- `partial_correlation_results_long_format.csv`: Detailed partial correlation analysis results, including correlation coefficients, p-values, and corrected p-values for each variable pair
- `annotated_partial_corr_matrix.csv`: Correlation matrix with significance markings, providing an intuitive view of relationships between variables
- `correlation_methods_parameters.csv`: Record of methodological parameters used in the analysis

## Visualization Files

- `partial_spearman_correlation_matrix.png`: Correlation coefficient heatmap, visually displaying correlation strength using color intensity
- `correlation_significance_matrix.png`: Correlation significance matrix, using different colors to represent different significance levels

## File Interpretation Guide

1. **Correlation Coefficient Matrix**:
   - Deeper colors indicate stronger correlations
   - Red indicates positive correlation, blue indicates negative correlation
   
2. **Significance Markings**:
   - * indicates p < 0.05
   - ** indicates p < 0.01
   - *** indicates p < 0.001
   - + indicates still significant after FDR correction

3. **Partial Correlation Analysis**: Shows correlation relationships after controlling for covariates including Sex, Age, Years of Education, APOE ε4 Status, and Total Intracranial Volume, reflecting the true relationships between variables after excluding the influence of these factors
"""

with open(os.path.join(output_dir, 'README.md'), 'w', encoding='utf-8') as f:
    f.write(readme_content)
print(f"Created result description file {output_dir}/README.md") 