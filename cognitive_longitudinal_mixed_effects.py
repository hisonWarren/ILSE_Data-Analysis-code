import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import stats
import os
import shutil

# Create output directory
output_dir = "results/mixed_effects_analysis"
os.makedirs(output_dir, exist_ok=True)
print(f"Created output directory: {output_dir}")

# Output software version information
print(f"pandas version: {pd.__version__}")
print(f"numpy version: {np.__version__}")
print(f"statsmodels version: {sm.__version__}")
print(f"scikit-learn version: {StandardScaler().__module__.split('.')[0]}")
print(f"matplotlib version: {matplotlib.__version__}")
print(f"seaborn version: {sns.__version__}")
print(f"scipy version: {scipy.__version__}")

# Define test name mapping dictionary
test_labels = {
    'TMA': 'TMT-A',        # Trail Making Test A 
    'TMB': 'TMT-B',        # Trail Making Test B
    'EGU12logicalmemory': 'LM I',   # Logical Memory I
    'LGV12logicalmemory': 'LM II',  # Logical Memory II 
    'ZNVORdigitspanforward': 'DSF',  # Digit Span Forward
    'ZNNACHdigitspanbackward': 'DSB' # Digit Span Backward
}

# Helper function to get display name of test
def get_display_name(test):
    """Get the display name for a test"""
    return test_labels.get(test, test)

# Read data
file_path = 'Data-Apoe/reshaped_cognitive_test_data.csv'
data_long = pd.read_csv(file_path)

# Output basic data information
print(f"Data shape: {data_long.shape}")
print(f"Unique subject IDs: {data_long['ID'].nunique()}")
print(f"Cognitive test types: {data_long['Test'].unique()}")

# Save descriptive statistics
data_description = data_long.groupby('Test').describe()
data_description.to_csv(os.path.join(output_dir, 'cognitive_tests_descriptive_stats.csv'))
print(f"Saved descriptive statistics to {output_dir}/cognitive_tests_descriptive_stats.csv")

# Outlier detection function
def detect_outliers(data, column, groupby_cols=None, threshold=3):
    """
    Detect outliers
    
    Parameters:
    data (DataFrame): Data
    column (str): Column to check
    groupby_cols (list): Grouping columns, if needed for groupwise detection
    threshold (float): Z-score threshold
    
    Returns:
    Series: Boolean index of outliers
    """
    if groupby_cols:
        return data.groupby(groupby_cols)[column].transform(
            lambda x: np.abs(x - x.mean()) > threshold * x.std()
        )
    else:
        z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
        return z_scores > threshold

# Detect and record outliers
outliers = detect_outliers(data_long, 'Score', ['Test', 'Time'], threshold=3)
outliers_data = data_long[outliers]
if not outliers_data.empty:
    outliers_data.to_csv(os.path.join(output_dir, 'outliers_in_cognitive_scores.csv'), index=False)
    print(f"Detected {len(outliers_data)} outliers, saved to {output_dir}/outliers_in_cognitive_scores.csv")
else:
    print("No outliers detected")

# Standardize variables (excluding SEX, as it's already coded as 1 and 2)
scaler = StandardScaler()
data_long[['Age', 'EDUC']] = scaler.fit_transform(data_long[['Age', 'EDUC']])

# Save standardization parameters
scaling_params = {
    'Age_mean': data_long['Age'].mean(),
    'Age_std': data_long['Age'].std(),
    'EDUC_mean': data_long['EDUC'].mean(),
    'EDUC_std': data_long['EDUC'].std()
}
pd.DataFrame([scaling_params]).to_csv(os.path.join(output_dir, 'scaling_parameters.csv'), index=False)
print(f"Saved standardization parameters to {output_dir}/scaling_parameters.csv")

# Build mixed-effects models for each cognitive test
results_with_covariates = {}
results_simplified = {}
model_diagnostics = {}  # Store model diagnostic results
failed_tests_with_covariates = []
failed_tests_simplified = []

# Use AIC and BIC to evaluate model goodness of fit
model_fit_metrics = []

# Add interaction effects to model
formula_with_interaction = 'Score ~ Time + APOE + Age + SEX + EDUC + Time:APOE + Time:SEX'
formula_simplified = 'Score ~ Time'

for test in data_long['Test'].unique():
    # Select data for specific cognitive test
    test_data = data_long[data_long['Test'] == test]
    
    # Get display name of the test
    display_test = get_display_name(test)
    
    # Record basic test information
    test_info = {
        'Test': test,
        'Display_Test': display_test,
        'Sample_Size': len(test_data),
        'Unique_Subjects': test_data['ID'].nunique(),
        'Time_Points': test_data['Time'].nunique()
    }
    
    # Model fitting attempt - full model
    try:
        # Fit model with covariates and interaction effects
        md = mixedlm(formula_with_interaction, test_data, groups=test_data['ID'], re_formula="~Time")
        mdf = md.fit(method='lbfgs', maxiter=1000)  # Explicitly specify maximum iterations
        results_with_covariates[test] = mdf
        
        # Record model fit metrics
        test_info.update({
            'Full_Model_AIC': mdf.aic,
            'Full_Model_BIC': mdf.bic,
            'Full_Model_LogLikelihood': mdf.llf
        })
        
        # Model diagnostics - residual analysis
        residuals = mdf.resid
        
        # Save residual descriptive statistics
        resid_stats = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals)
        }
        
        # Shapiro-Wilk normality test
        shapiro_test = stats.shapiro(residuals)
        
        # Residuals vs. fitted values plot
        plt.figure(figsize=(10, 6))
        plt.rcParams.update({'font.size': 14})  # Increase font size
        fitted_values = mdf.predict(test_data)
        plt.scatter(fitted_values, residuals)
        plt.axhline(y=0, color='red', linestyle='-')
        plt.xlabel('Fitted Values', fontsize=16)
        plt.ylabel('Residuals', fontsize=16)
        plt.title(f'Residuals vs Fitted Values for {display_test} - Full Model', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(os.path.join(output_dir, f'{test}_full_model_residual_plot.png'), dpi=300)
        plt.close()
        
        # Q-Q plot to check residual normality
        plt.figure(figsize=(10, 6))
        plt.rcParams.update({'font.size': 14})  # Increase font size
        fig = stats.probplot(residuals, dist="norm", plot=plt)
        plt.title(f'Q-Q Plot for {display_test} - Full Model', fontsize=16)
        plt.xlabel('Theoretical Quantiles', fontsize=16)
        plt.ylabel('Sample Quantiles', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.savefig(os.path.join(output_dir, f'{test}_full_model_qq_plot.png'), dpi=300)
        plt.close()
        
        # Save diagnostic results
        model_diagnostics[f"{test}_full"] = {
            'residual_stats': resid_stats,
            'shapiro_test': {
                'statistic': shapiro_test[0],
                'p_value': shapiro_test[1]
            }
        }
        
    except Exception as e:
        print(f"Failed to fit model with covariates: {display_test}, error: {e}")
        failed_tests_with_covariates.append(test)
        test_info['Full_Model_Fit_Failed'] = True
        test_info['Full_Model_Error'] = str(e)
    
    # Model fitting attempt - simplified model
    try:
        # Fit simplified model
        md_simplified = mixedlm(formula_simplified, test_data, groups=test_data['ID'], re_formula="~Time")
        mdf_simplified = md_simplified.fit(method='lbfgs', maxiter=1000)
        results_simplified[test] = mdf_simplified
        
        # Record model fit metrics
        test_info.update({
            'Simple_Model_AIC': mdf_simplified.aic,
            'Simple_Model_BIC': mdf_simplified.bic,
            'Simple_Model_LogLikelihood': mdf_simplified.llf
        })
        
        # If both models fitted successfully, compare models
        if test in results_with_covariates:
            # Likelihood ratio test
            lr_stat = -2 * (mdf_simplified.llf - mdf.llf)
            lr_df = len(mdf.params) - len(mdf_simplified.params)
            lr_pval = stats.chi2.sf(lr_stat, lr_df)
            
            test_info.update({
                'LR_Test_Statistic': lr_stat,
                'LR_Test_DF': lr_df,
                'LR_Test_PValue': lr_pval,
                'Preferred_Model': 'Full' if lr_pval < 0.05 else 'Simple'
            })
        
    except Exception as e:
        print(f"Simplified model also failed to fit: {display_test}, error: {e}")
        failed_tests_simplified.append(test)
        test_info['Simple_Model_Fit_Failed'] = True
        test_info['Simple_Model_Error'] = str(e)
    
    # Add test information to list
    model_fit_metrics.append(test_info)

# Save model fit metrics
pd.DataFrame(model_fit_metrics).to_csv(os.path.join(output_dir, 'model_fit_metrics.csv'), index=False)
print(f"Saved model fit metrics to {output_dir}/model_fit_metrics.csv")

# Save model diagnostic results
with open(os.path.join(output_dir, 'model_diagnostics.txt'), 'w') as f:
    for model_name, diagnostics in model_diagnostics.items():
        test_code = model_name.split('_')[0]
        display_name = get_display_name(test_code)
        f.write(f"Model: {display_name} ({model_name})\n")
        f.write("Residual Statistics:\n")
        for stat_name, value in diagnostics['residual_stats'].items():
            f.write(f"  {stat_name}: {value}\n")
        f.write("Shapiro-Wilk Test for Normality:\n")
        f.write(f"  Statistic: {diagnostics['shapiro_test']['statistic']}\n")
        f.write(f"  p-value: {diagnostics['shapiro_test']['p_value']}\n")
        f.write("\n")
print(f"Saved model diagnostic results to {output_dir}/model_diagnostics.txt")

# Get all p-values from all models
p_values_with_covariates = []
for model in results_with_covariates.values():
    p_values_with_covariates.extend(model.pvalues.dropna().values)

p_values_simplified = []
for model in results_simplified.values():
    p_values_simplified.extend(model.pvalues.dropna().values)

# Print original p-value counts
print(f"Full model original p-value count: {len(p_values_with_covariates)}")
print(f"Simplified model original p-value count: {len(p_values_simplified)}")

# Check if there are enough p-values for multiple comparison correction
if len(p_values_with_covariates) > 0:
    # Explicitly specify FDR correction parameters
    _, corrected_p_values_with_covariates, _, _ = multipletests(
        p_values_with_covariates, 
        alpha=0.05, 
        method='fdr_bh',
        returnsorted=False
    )
    print(f"Full model FDR-corrected p-value count: {len(corrected_p_values_with_covariates)}")
else:
    corrected_p_values_with_covariates = np.array([])
    print("Full model doesn't have enough p-values for correction")

if len(p_values_simplified) > 0:
    _, corrected_p_values_simplified, _, _ = multipletests(
        p_values_simplified, 
        alpha=0.05, 
        method='fdr_bh',
        returnsorted=False
    )
    print(f"Simplified model FDR-corrected p-value count: {len(corrected_p_values_simplified)}")
else:
    corrected_p_values_simplified = np.array([])
    print("Simplified model doesn't have enough p-values for correction")

# Reassign corrected p-values back to each model
def corrected_pvalues_dict(models, corrected_p_values):
    corrected_p_dict = {}
    index = 0
    for test, model in models.items():
        num_params = len(model.pvalues.dropna())
        if index + num_params <= len(corrected_p_values):
            corrected_p_dict[test] = corrected_p_values[index:index+num_params]
            index += num_params
        else:
            print(f"Warning: Corrected p-values for {get_display_name(test)} may not be assigned correctly")
            corrected_p_dict[test] = np.array([])
    return corrected_p_dict

if len(corrected_p_values_with_covariates) > 0:
    corrected_p_dict_with_covariates = corrected_pvalues_dict(results_with_covariates, corrected_p_values_with_covariates)
else:
    corrected_p_dict_with_covariates = {test: np.array([]) for test in results_with_covariates.keys()}

if len(corrected_p_values_simplified) > 0:
    corrected_p_dict_simplified = corrected_pvalues_dict(results_simplified, corrected_p_values_simplified)
else:
    corrected_p_dict_simplified = {test: np.array([]) for test in results_simplified.keys()}

# Ensure all columns have equal length
def ensure_equal_length(data):
    max_len = max(len(col) for col in data.values())
    for key in data:
        if len(data[key]) < max_len:
            data[key] = np.append(data[key], [np.nan] * (max_len - len(data[key])))
    return data

# Function to save model results to CSV file
def save_model_results(test, model, filename_suffix, corrected_p=None):
    display_test = get_display_name(test)
    description = model.params.index.tolist()
    pvalues_len = len(model.pvalues.dropna())
    if corrected_p is not None and len(corrected_p) > 0:
        corrected_p = corrected_p[:pvalues_len]  # Ensure consistent length
    else:
        corrected_p = np.full(pvalues_len, np.nan)
    
    data = {
        "Description": description,
        "Coefficient": model.params.round(3).values,
        "Std Error": model.bse.round(3).values,
        "z-value": model.tvalues.round(3),
        "P>|z|": model.pvalues.round(3),
        "Corrected P>|z|": corrected_p.round(3) if not np.all(np.isnan(corrected_p)) else model.pvalues.round(3),
        "[0.025": model.conf_int().iloc[:, 0].round(3),
        "0.975]": model.conf_int().iloc[:, 1].round(3)
    }

    # Ensure all columns have equal length
    data = ensure_equal_length(data)
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(data)
    file_path = os.path.join(output_dir, f"{test}_{filename_suffix}.csv")  # Save to output directory
    results_df.to_csv(file_path, index=False)
    print(f"Saved {display_test} {filename_suffix} results to {file_path}")
    return file_path

# Save methodology parameter information
methods_info = {
    "Analysis Method": ["Linear Mixed Effects Model"],
    "Package": ["statsmodels.formula.api.mixedlm"],
    "Full Model Formula": [formula_with_interaction],
    "Simplified Model Formula": [formula_simplified],
    "Random Effects": ["~Time (grouped by subject ID)"],
    "Optimization Method": ["LBFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno)"],
    "Maximum Iterations": [1000],
    "Standardized Variables": ["Age, EDUC (Z-score standardization)"],
    "Multiple Comparison Correction": ["Benjamini-Hochberg (FDR), alpha=0.05"],
    "Model Comparison Method": ["Likelihood Ratio Test"],
    "Model Evaluation Metrics": ["AIC, BIC, Log Likelihood"],
    "Diagnostic Tests": ["Residual analysis, Shapiro-Wilk normality test, Q-Q plot, Residual scatter plot"]
}
pd.DataFrame(methods_info).to_csv(os.path.join(output_dir, 'mixed_effects_methods_parameters.csv'), index=False)
print(f"Saved methodology parameter information to {output_dir}/mixed_effects_methods_parameters.csv")

# Save model results
saved_files = []

# Save models with covariates results
for test, model in results_with_covariates.items():
    try:
        file_path = save_model_results(test, model, "model_with_covariates_results", 
                                      corrected_p_dict_with_covariates.get(test, np.array([])))
        if file_path:
            saved_files.append(file_path)
    except Exception as e:
        print(f"Failed to save {get_display_name(test)} results, error: {e}")

# Save simplified model results
for test, model in results_simplified.items():
    try:
        file_path = save_model_results(test, model, "simplified_model_results",
                                      corrected_p_dict_simplified.get(test, np.array([])))
        if file_path:
            saved_files.append(file_path)
    except Exception as e:
        print(f"Failed to save {get_display_name(test)} simplified model results, error: {e}")

# Extract forest plot data
def extract_forest_data(models):
    forest_data = []
    for test, model in models.items():
        try:
            conf = model.conf_int()
            # Check if 'Time[T.T4]' is in the model parameters
            if 'Time[T.T4]' in model.params:
                forest_data.append({
                    "Test": test,
                    "Display_Test": get_display_name(test),
                    "Coefficient": model.params['Time[T.T4]'],
                    "Lower CI": conf.loc['Time[T.T4]', 0],
                    "Upper CI": conf.loc['Time[T.T4]', 1],
                    "P-value": model.pvalues['Time[T.T4]'],
                    "Color": 'red' if model.pvalues['Time[T.T4]'] < 0.05 else 'blue'
                })
            else:
                print(f"Warning: Test {get_display_name(test)} model does not have 'Time[T.T4]' parameter")
        except Exception as e:
            print(f"Error extracting forest plot data for {get_display_name(test)}: {e}")
    return forest_data

# Forest plot drawing function
def plot_forest_plot(forest_data, title, file_name):
    if not forest_data:
        print(f"Warning: No data available to draw forest plot '{title}'")
        return
    
    forest_df = pd.DataFrame(forest_data)
    # Use display test names
    display_tests = forest_df['Display_Test']
    
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.size': 16})  # Increase font size
    
    for i in range(len(forest_df)):
        row = forest_df.iloc[i]
        plt.errorbar(row['Coefficient'], i, 
                    xerr=[[row['Coefficient'] - row['Lower CI']], [row['Upper CI'] - row['Coefficient']]], 
                    fmt='o', color=row['Color'], ecolor=row['Color'], capsize=5, markersize=10)
    
    plt.yticks(range(len(forest_df)), display_tests, fontsize=18)
    plt.xticks(fontsize=16)
    plt.axvline(x=0, linestyle='--', color='black')
    plt.xlabel('Coefficient', fontsize=18)
    plt.title(title, fontsize=20)
    plt.gca().invert_yaxis()  # Invert Y-axis to put first test at top
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, file_name), dpi=600)  # Save as 600dpi resolution file
    plt.close()
    print(f"Saved forest plot to {output_dir}/{file_name}")

# Generate forest plot data and draw charts
try:
    forest_data_with_covariates = extract_forest_data(results_with_covariates)
    plot_forest_plot(forest_data_with_covariates, 'Cognitive Tests Forest Plot (Model with Covariates)', 'forest_plot_with_covariates.png')
except Exception as e:
    print(f"Error drawing forest plot with covariates: {e}")

try:
    forest_data_simplified = extract_forest_data(results_simplified)
    plot_forest_plot(forest_data_simplified, 'Cognitive Tests Forest Plot (Simplified Model)', 'forest_plot_simplified.png')
except Exception as e:
    print(f"Error drawing simplified model forest plot: {e}")

# Function to create combined forest plots
def plot_combined_forest_plots(simplified_data, covariates_data, file_name="combined_forest_plot.png"):
    """
    Plot two forest plots in a single chart:
    Simplified model on the left, full model on the right
    """
    if not simplified_data or not covariates_data:
        print(f"Warning: Unable to draw combined forest plot, at least one dataset is empty")
        return
    
    # Convert to DataFrame
    simplified_df = pd.DataFrame(simplified_data)
    covariates_df = pd.DataFrame(covariates_data)
    
    # Ensure consistent test order between the two datasets
    test_display_order = dict(zip(simplified_df['Test'], simplified_df['Display_Test']))
    simplified_df = simplified_df.set_index('Test')
    covariates_df = covariates_df.set_index('Test')
    
    # Get common tests
    common_tests = sorted(set(simplified_df.index) & set(covariates_df.index), 
                        key=lambda x: list(test_display_order.keys()).index(x) if x in test_display_order else 999)
    
    if not common_tests:
        print("Warning: No common tests between simplified and full models")
        return
    
    # Filter to common tests and order consistently
    simplified_df = simplified_df.loc[common_tests]
    covariates_df = covariates_df.loc[common_tests]
    
    # Set up chart size and layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
    plt.rcParams.update({'font.size': 14})
    
    # Get display test names for y-axis
    display_tests = [test_display_order[test] for test in common_tests]
    
    # Left side: Simplified model
    for i, test in enumerate(common_tests):
        row = simplified_df.loc[test]
        ax1.errorbar(row['Coefficient'], i, 
                    xerr=[[row['Coefficient'] - row['Lower CI']], [row['Upper CI'] - row['Coefficient']]], 
                    fmt='o', color=row['Color'], ecolor=row['Color'], capsize=5, markersize=10)
    
    ax1.set_yticks(range(len(common_tests)))
    ax1.set_yticklabels(display_tests, fontsize=14)
    ax1.axvline(x=0, linestyle='--', color='black')
    ax1.set_xlabel('Coefficient', fontsize=14)
    ax1.set_title('Cognitive Tests (Simplified Model)', fontsize=16, pad=15)
    ax1.grid(alpha=0.3)
    
    # Right side: Full model
    for i, test in enumerate(common_tests):
        row = covariates_df.loc[test]
        ax2.errorbar(row['Coefficient'], i, 
                    xerr=[[row['Coefficient'] - row['Lower CI']], [row['Upper CI'] - row['Coefficient']]], 
                    fmt='o', color=row['Color'], ecolor=row['Color'], capsize=5, markersize=10)
    
    ax2.axvline(x=0, linestyle='--', color='black')
    ax2.set_xlabel('Coefficient', fontsize=14)
    ax2.set_title('Cognitive Tests (Full Model with Covariates)', fontsize=16, pad=15)
    ax2.grid(alpha=0.3)
    
    # Invert Y-axis to put first test at top
    ax1.invert_yaxis()
    
    # Add legend
    fig.subplots_adjust(bottom=0.15) # Make space for legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='b', markersize=10, linestyle='None', label='Non-significant Effect'),
        plt.Line2D([0], [0], marker='o', color='r', markersize=10, linestyle='None', label='Significant Effect (p < 0.05)'),
        plt.Line2D([0], [0], linestyle='--', color='black', label='No Effect (Zero)')
    ]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3, frameon=False, fontsize=12)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Make space for bottom legend
    plt.savefig(os.path.join(output_dir, file_name), dpi=600)
    plt.close()
    print(f"Saved combined forest plot to {output_dir}/{file_name}")
    return os.path.join(output_dir, file_name)

# Create combined forest plot
try:
    plot_combined_forest_plots(
        simplified_data=forest_data_simplified,
        covariates_data=forest_data_with_covariates,
        file_name="combined_forest_plots.png"
    )
except Exception as e:
    print(f"Error drawing combined forest plots: {e}")

# Output summary information
print("Tests that failed to fit (with covariates):", [get_display_name(test) for test in failed_tests_with_covariates])
print("Tests that failed to fit (simplified model):", [get_display_name(test) for test in failed_tests_simplified])
print(f"Number of successfully saved result files: {len(saved_files)}")
print("Analysis complete!") 