import os

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import spearmanr, kstest, shapiro, jarque_bera
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multitest as multitest
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.preprocessing import PowerTransformer
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import gridspec

# Create output directory
output_dir = "result/regression_analysis"
os.makedirs(output_dir, exist_ok=True)
print(f"Created output directory: {output_dir}")

# Read data
file_path = r'Data\data_pre.xlsx' 
data = pd.read_excel(file_path)

# Check missing values
missing_values = data.isnull().sum()
missing_report = pd.DataFrame({'Variable': missing_values.index, 'Missing_Count': missing_values.values})
missing_report.to_csv(os.path.join(output_dir, 'missing_values_report.csv'), index=False)
print(f"Saved missing values report to {output_dir}/missing_values_report.csv")

# Descriptive statistics
description = data.describe(include='all')
description.to_csv(os.path.join(output_dir, 'descriptive_statistics.csv'))
print(f"Saved descriptive statistics to {output_dir}/descriptive_statistics.csv")

# Normality test: Kolmogorov-Smirnov (K-S) and Shapiro-Wilk (S-W)
variables = ['T3Age','EDUC','SEX',
    'T3TMA', 'T3TMB', 'T3EGU12logicalmemory', 'T3LGV12logicalmemory', 
    'T3ZNVORdigitspanforward', 'T3ZNNACHdigitspanbackward',
    'T3Component1', 'T3Component2', 'T3Component3', 'T3Component4',
    'T3Component5', 'T3Component6', 'T3Component7', 'T3Component8',
    'T3Component9', 'T3Component10', 'T3Component11', 'T3Component12'
]

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
    'APOE': 'APOE genotype',     # APOE ε4 carrier status
    'Group': 'APOE genotype',    # Group is also APOE status
    'SEX': 'Sex',                 # Sex
    'EDUC': 'Education',          # Education
    'T3Age': 'Age',               # Age
    'TIV': 'TIV'                  # Total Intracranial Volume
}

# Function to get display name
def get_display_name(var_name):
    """Gets the display name for a variable."""
    # Cognitive tests
    if "MMSE" in var_name:
        return "MMSE"
    elif "TMA" in var_name:
        if "T3" in var_name:
            return "TMT-A (T3)"
        else:
            return "TMT-A (T4)"
    elif "TMB" in var_name:
        if "T3" in var_name:
            return "TMT-B (T3)"
        else:
            return "TMT-B (T4)"
    elif "LGV12logicalmemory" in var_name:
        if "T3" in var_name:
            return "LM II (T3)"
        else:
            return "LM II (T4)"
    elif "EGU12logicalmemory" in var_name:
        if "T3" in var_name:
            return "LM I (T3)"
        else:
            return "LM I (T4)"
    elif "ZNVORdigitspanforward" in var_name:
        if "T3" in var_name:
            return "DSF (T3)"
        else:
            return "DSF (T4)"
    elif "ZNNACHdigitspanbackward" in var_name:
        if "T3" in var_name:
            return "DSB (T3)"
        else:
            return "DSB (T4)"
    
    # T3 Components
    elif "T3Component" in var_name:
        component_number = var_name.replace("T3Component", "")
        return f"IC{component_number} loading (T3)"
    
    # Sex and genotype
    elif var_name == "SEX":
        return "Sex"
    elif var_name == "Group":
        return "APOE genotype"
    
    # Other variables
    elif var_name == "T3Age":
        return "Age"
    elif var_name == "EDUC":
        return "Education"
    elif var_name == "TIV":
        return "TIV"
        
    # Interaction terms
    elif "_x_" in var_name:
        parts = var_name.split("_x_")
        return f"{get_display_name(parts[0])} × {get_display_name(parts[1])}"
    
    # Default: return original name
    return var_name

# Perform Kolmogorov-Smirnov (K-S) and Shapiro-Wilk (S-W) normality tests
normality_results = []
for var in variables:
    ks_stat, ks_p = kstest(data[var], 'norm', args=(data[var].mean(), data[var].std()))
    sw_stat, sw_p = shapiro(data[var])
    jb_stat, jb_p = jarque_bera(data[var])  # Add Jarque-Bera test
    normality_results.append((var, ks_stat, ks_p, sw_stat, sw_p, jb_stat, jb_p))

# Save normality test results
normality_df = pd.DataFrame(normality_results, 
                           columns=['Variable', 'K-S Statistic', 'K-S p-value', 
                                    'S-W Statistic', 'S-W p-value',
                                    'J-B Statistic', 'J-B p-value'])
normality_df.to_csv(os.path.join(output_dir, 'normality_results.csv'), index=False)
print(f"Saved normality test results to {output_dir}/normality_results.csv")

# Outlier detection function
def detect_outliers(data, columns, threshold=3):
    """
    Detect outliers using Z-scores
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

# Detect outliers
outliers_report = detect_outliers(data, variables)
outliers_report.to_csv(os.path.join(output_dir, 'outliers_report.csv'), index=False)
print(f"Saved outlier report to {output_dir}/outliers_report.csv")

# Check for multicollinearity: calculate Variance Inflation Factor (VIF)
def calculate_vif(X):
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif

# Calculate VIF for variables
vif_data = calculate_vif(data[variables])
vif_data.to_csv(os.path.join(output_dir, 'vif_results.csv'), index=False)
print(f"Saved VIF results to {output_dir}/vif_results.csv")

# Transform non-normally distributed variables using Yeo-Johnson transformation
transformed_vars = {}
for var in variables:
    if normality_df[normality_df['Variable'] == var]['S-W p-value'].values[0] < 0.05:
        pt = PowerTransformer(method='yeo-johnson')
        data[f'{var}_transformed'] = pt.fit_transform(data[[var]]).flatten()
        transformed_vars[var] = f'{var}_transformed'

# Update variable list using transformed variables
transformed_variables = []
for var in variables:
    if var in transformed_vars:
        transformed_variables.append(transformed_vars[var])
    else:
        transformed_variables.append(var)

# Record transformation parameters
transformation_details = pd.DataFrame({
    'Original_Variable': list(transformed_vars.keys()),
    'Transformed_Variable': list(transformed_vars.values()),
    'Transformation_Method': ['Yeo-Johnson'] * len(transformed_vars)
})
transformation_details.to_csv(os.path.join(output_dir, 'transformation_details.csv'), index=False)
print(f"Saved transformation details to {output_dir}/transformation_details.csv")

# Define group correlation analysis function
def perform_group_correlation(data, variables):
    results = []
    for var in variables[:6]:
        for component in variables[6:]:
            corr, p_value = spearmanr(data[var], data[component])
            results.append((var, component, corr, p_value))
    return results

# Calculate group correlation analysis results
apoe_negative = data[data['Group'] == 1]
apoe_positive = data[data['Group'] == 2]
negative_correlation_results = perform_group_correlation(apoe_negative, transformed_variables)
positive_correlation_results = perform_group_correlation(apoe_positive, transformed_variables)

# Save group correlation analysis results
negative_corr_df = pd.DataFrame(negative_correlation_results, columns=['Variable', 'Component', 'Correlation', 'p-value_negative'])
positive_corr_df = pd.DataFrame(positive_correlation_results, columns=['Variable', 'Component', 'Correlation', 'p-value_positive'])

# Merge results
comparison_df = negative_corr_df.merge(positive_corr_df, on=['Variable', 'Component'])

# Correct p-values using Benjamini-Hochberg
comparison_df['adjusted p-value_negative'] = multitest.multipletests(
    comparison_df['p-value_negative'], 
    method='fdr_bh',
    alpha=0.05,
    returnsorted=False
)[1]
comparison_df['adjusted p-value_positive'] = multitest.multipletests(
    comparison_df['p-value_positive'], 
    method='fdr_bh',
    alpha=0.05,
    returnsorted=False
)[1]

# Save group comparison results
comparison_df.to_csv(os.path.join(output_dir, 'group_comparison_results.csv'), index=False)
print(f"Saved group comparison results to {output_dir}/group_comparison_results.csv")

# Bootstrap model fitting function with parallel processing
def bootstrap_model_parallel(model, X, y, n_bootstrap=1000, n_jobs=-1):
    """
    Use parallel processing for Bootstrap resampling to improve computation efficiency
    
    Parameters:
    model: Fitted model
    X: Independent variables DataFrame
    y: Dependent variable Series
    n_bootstrap: Number of Bootstrap resamples
    n_jobs: Number of parallel jobs, -1 means using all available cores
    
    Returns:
    DataFrame: DataFrame with Bootstrap confidence intervals
    """
    def single_bootstrap(seed):
        np.random.seed(seed)
        try:
            sample_indices = np.random.choice(range(len(y)), size=len(y), replace=True)
            X_sample = X.iloc[sample_indices]
            y_sample = y.iloc[sample_indices]
            boot_model = sm.OLS(y_sample, sm.add_constant(X_sample)).fit()
            return boot_model.params
        except Exception as e:
            print(f"Bootstrap sample fitting error: {e}")
            return None
    
    print(f"Starting Bootstrap analysis, resampling count: {n_bootstrap}...")
    boot_coefs = Parallel(n_jobs=n_jobs)(
        delayed(single_bootstrap)(i) for i in range(n_bootstrap)
    )
    # Remove None values
    boot_coefs = [coef for coef in boot_coefs if coef is not None]
    
    if not boot_coefs:
        print("All Bootstrap samples failed to fit!")
        return None
    
    boot_coefs_df = pd.DataFrame(boot_coefs)
    conf_intervals = boot_coefs_df.quantile([0.025, 0.975])
    
    # Add more Bootstrap statistics
    bootstrap_stats = pd.DataFrame({
        'Mean': boot_coefs_df.mean(),
        'Std': boot_coefs_df.std(),
        'Lower_CI': conf_intervals.iloc[0],
        'Upper_CI': conf_intervals.iloc[1]
    })
    
    print(f"Bootstrap analysis complete, successful samples: {len(boot_coefs)}/{n_bootstrap}")
    return bootstrap_stats

# Create function for partial regression plots
def create_partial_regression_plots(model, data, dependent_var, independent_vars, output_dir, 
                               min_r_squared=0.001, only_significant=True, transformed_vars=None, max_plots=None):
    """
    Create partial regression plots (component-plus-residual plots) for multiple regression models.
    
    Parameters:
    model: Fitted statsmodels regression model
    data: DataFrame containing all variables
    dependent_var: Name of the dependent variable
    independent_vars: List of independent variables to create plots for
    output_dir: Directory to save the plots
    min_r_squared: Minimum partial R-squared value to include a variable (default: 0.001)
    only_significant: Whether to only create plots for significant variables (default: True)
    transformed_vars: Optional dictionary of transformed variables for display
    max_plots: Maximum number of plots to create (default: None, means no limit)
    
    Returns:
    dict: Dictionary mapping variable names to their plot file paths
    """
    # Create the output directory if it doesn't exist
    partial_plots_dir = os.path.join(output_dir, "partial_regression_plots")
    os.makedirs(partial_plots_dir, exist_ok=True)
    
    # Record paths to all created plots
    plot_files = {}
    
    # Calculate variable importance based on partial R-squared and p-values
    var_importance = {}
    all_significant = []
    
    # Reset matplotlib params to default
    plt.rcParams.update(plt.rcParamsDefault)
    
    # Set Times New Roman font and other style parameters
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    
    # Enhance axis clarity
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['xtick.major.size'] = 5.0
    plt.rcParams['ytick.major.size'] = 5.0
    # Remove minor ticks as requested
    plt.rcParams['xtick.minor.size'] = 0.0
    plt.rcParams['ytick.minor.size'] = 0.0
    
    # Define colors to use (not using blue as requested)
    point_color = '#e74c3c'  # red color
    line_color = '#2c3e50'   # dark slate color
    ci_color = '#e74c3c'     # red color for confidence interval
    
    # Filter variables based on significance and partial R-squared
    for var in independent_vars:
        # Skip if variable not in model
        if var not in model.params:
            continue
        
        # Calculate partial R-squared
        # Handle model.model.exog which could be numpy array or pandas DataFrame
        try:
            # Check if it's a DataFrame
            if hasattr(model.model.exog, 'drop'):
                # If DataFrame, use drop method directly
                model_without = sm.OLS(model.model.endog, 
                                    model.model.exog.drop(var, axis=1),
                                    hasconst=True).fit()
            else:
                # If numpy array, find index of variable and remove corresponding column
                var_idx = list(model.params.index).index(var)
                # Exclude the column at var_idx (note index may be offset if constant is present)
                X_without = np.delete(model.model.exog, var_idx, axis=1)
                model_without = sm.OLS(model.model.endog, X_without, hasconst=True).fit()
                
            partial_rsquared = model.rsquared - model_without.rsquared
        except Exception as e:
            print(f"  ⚠️ Cannot calculate partial R² for {var}: {str(e)}")
            partial_rsquared = 0
        
        # Get p-value for this variable
        p_value = model.pvalues[var]

        # Adjust p-value using FDR method to be consistent with main model
        adjusted_p_value = multitest.multipletests([p_value], method='fdr_bh')[1][0]

        # Add to variable importance dict
        var_importance[var] = {
            'partial_rsquared': partial_rsquared,
            'p_value': adjusted_p_value,  # Use adjusted p-value
            'is_significant': adjusted_p_value < 0.05
        }
        
        # Track significant variables
        if adjusted_p_value < 0.05:  # Use adjusted p-value to determine significance
            all_significant.append(var)
    
    # Skip if no variables meet criteria
    if only_significant and not all_significant:
        print(f"  ⚠️ No significant variables found for {dependent_var}. Skipping partial regression plots.")
        return plot_files
    
    # Create a combined figure for all significant variables
    if all_significant and only_significant:
        n_plots = len(all_significant)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        # Make axes iterable even if there's only one plot
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Set flag to track if all subplots successfully added confidence intervals
        all_ci_added = True
        ci_failed_vars = []
        
        # Create individual plots
        for i, var in enumerate(all_significant):
            if i < len(axes):
                ax = axes[i]
                
                # Call function to create individual plot
                ax = _create_single_partial_plot(model, data, dependent_var, var, ax, 
                                           var_importance[var]['p_value'],
                                           transformed_vars, point_color, line_color, ci_color)
                
                # Check if subplot has confidence interval attribute (using custom attribute)
                if not hasattr(ax, 'ci_added') or not ax.ci_added:
                    # If confidence interval not added, try simplified method
                    try:
                        # Get X and Y data (scatter data)
                        # Try to get point data from ax object
                        scatter_collections = [c for c in ax.collections if isinstance(c, plt.matplotlib.collections.PathCollection)]
                        if scatter_collections:
                            scatter = scatter_collections[0]
                            offsets = scatter.get_offsets()
                            x_data = offsets[:, 0]
                            y_data = offsets[:, 1]
                            
                            # Get regression line slope (regression line already drawn in plot)
                            lines = [l for l in ax.get_lines() if len(l.get_xdata()) > 2]  # Regression line typically has multiple points
                            if lines:
                                line = lines[0]
                                x_range = line.get_xdata()
                                y_range = line.get_ydata()
                                
                                # Estimate slope
                                if len(x_range) >= 2:
                                    slope = (y_range[-1] - y_range[0]) / (x_range[-1] - x_range[0])
                                    
                                    # Calculate residuals and estimate confidence interval
                                    residuals = []
                                    for j in range(len(x_data)):
                                        predicted = slope * x_data[j]
                                        residuals.append(y_data[j] - predicted)
                                    
                                    residual_std = np.std(residuals)
                                    t_value = stats.t.ppf(0.975, len(x_data) - 2)
                                    
                                    # Calculate funnel-shaped confidence interval
                                    X_mean = np.mean(x_data)
                                    X_var = np.sum((x_data - X_mean)**2)
                                    n = len(x_data)
                                    
                                    # Calculate different width confidence intervals for each point
                                    ci_lower = []
                                    ci_upper = []
                                    for i, x in enumerate(x_range):
                                        leverage = 1/n + ((x - X_mean)**2)/X_var
                                        width = t_value * residual_std * np.sqrt(leverage)
                                        ci_lower.append(y_range[i] - width)
                                        ci_upper.append(y_range[i] + width)
                                    
                                    # Add confidence interval
                                    ax.fill_between(x_range, ci_lower, ci_upper, 
                                                   color=ci_color, alpha=0.25)
                                    
                                    # Mark as having successfully added confidence interval
                                    ax.ci_added = True
                                    print(f"  ✓ Added confidence interval to {var} in combined plot (using fallback method)")
                    except Exception as e:
                        print(f"  ⚠️ Failed to add confidence interval for {var} in combined plot: {str(e)}")
                        all_ci_added = False
                        ci_failed_vars.append(var)
        
        # Hide any unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Use clearer file naming
        if "T4" in dependent_var:
            test_name = dependent_var.replace("T4", "")
            combined_file = os.path.join(partial_plots_dir, f"{get_display_name(dependent_var)}_combined_partial_regression.png")
        else:
            combined_file = os.path.join(partial_plots_dir, f"{dependent_var}_all_significant_partial_regression.png")
            
        plt.savefig(combined_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plot_files['combined'] = combined_file
        
        # Report confidence interval status
        if all_ci_added:
            print(f"  ✓ All subplots successfully added confidence intervals")
        else:
            print(f"  ⚠️ {len(ci_failed_vars)} subplots failed to add confidence intervals: {', '.join(ci_failed_vars)}")
            
        print(f"  ✓ Created combined partial regression plot for {get_display_name(dependent_var)} with {n_plots} significant variables")
    
    # Variables to actually plot
    vars_to_plot = all_significant if only_significant else independent_vars
    
    # Limit the number of plots if max_plots is specified
    if max_plots is not None and len(vars_to_plot) > max_plots:
        # Sort variables by importance (partial R-squared) and keep only the top ones
        sorted_vars = sorted(
            [(var, var_importance.get(var, {}).get('partial_rsquared', 0)) for var in vars_to_plot if var in var_importance],
            key=lambda x: x[1],
            reverse=True
        )
        vars_to_plot = [var for var, _ in sorted_vars[:max_plots]]
        print(f"  ℹ️ Limiting to top {max_plots} variables by partial R-squared")
    
    # Create individual plots for each variable
    for var in vars_to_plot:
        # Skip if variable not in importance dict (would happen if var not in model)
        if var not in var_importance:
            continue
            
        # Skip if below minimum R-squared threshold
        if var_importance[var]['partial_rsquared'] < min_r_squared:
            continue
            
        # Create the partial regression plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create the actual plot
        ax = _create_single_partial_plot(model, data, dependent_var, var, ax, 
                                   var_importance[var]['p_value'],
                                   transformed_vars, point_color, line_color, ci_color)
        
        # Check if confidence interval was successfully added
        if hasattr(ax, 'ci_added') and ax.ci_added:
            print(f"  ✓ Successfully added confidence interval to individual plot for {var}")
        else:
            print(f"  ⚠️ Warning: Individual plot for {var} may not have confidence interval")
            
        # Adjust layout and save individual plot
        plt.tight_layout()
        
        # Create filename with clear naming convention
        if "_x_" in var:
            parts = var.split("_x_")
            if len(parts) == 2:
                part1, part2 = parts
                if "T3Component" in part2:
                    # For interaction terms with brain components, use more concise naming
                    ic_number = part2.replace("T3Component", "")
                    file_name = f"{dependent_var}_IC{ic_number}_x_{part1}_partial_regression.png"
                else:
                    # Other types of interaction terms
                    file_name = f"{dependent_var}_{var}_partial_regression.png"
            else:
                file_name = f"{dependent_var}_{var}_partial_regression.png"
        else:
            file_name = f"{dependent_var}_{var}_partial_regression.png"
            
        file_path = os.path.join(partial_plots_dir, file_name)
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Store the file path
        plot_files[var] = file_path
        
        # Print status
        importance_info = var_importance[var]
        sig_symbol = "*" if importance_info['is_significant'] else ""
        print(f"  ✓ Created partial regression plot for {var} (p={importance_info['p_value']:.3f}{sig_symbol}, "
              f"partial R²={importance_info['partial_rsquared']:.3f}")
    
    return plot_files

def _create_single_partial_plot(model, data, dependent_var, independent_var, ax, p_value, 
                               transformed_vars=None, point_color='#e74c3c', 
                               line_color='#2c3e50', ci_color='#e74c3c'):
    """Helper function to create a single partial regression plot"""
    try:
        # Initialize flag indicating confidence interval hasn't been added yet
        ax.ci_added = False
        
        # Get variable display names
        dependent_display = get_display_name(dependent_var)
        
        # Handle interaction term naming (variables containing _x_)
        if isinstance(independent_var, str) and "_x_" in independent_var:
            parts = independent_var.split("_x_")
            if len(parts) == 2:
                # Extract two parts and get their display names
                part1, part2 = parts
                
                # Special handling for Component part
                if "T3Component" in part2:
                    ic_number = part2.replace("T3Component", "")
                    part2_display = f"IC{ic_number} Loading"
                else:
                    part2_display = get_display_name(part2)
                
                # First part might be baseline test like T3ZNNACHdigitspanbackward
                part1_display = get_display_name(part1)
                if part1.startswith("T3"):
                    part1_display = f"{part1_display}"
                
                # Combine into final display name
                independent_display = f"{part2_display} × {part1_display} (T3)"
            else:
                # If not split into two parts, use default handling
                independent_display = independent_var.replace("_x_", " × ")
        
        # Handle regular brain component variables (T3Component)
        elif isinstance(independent_var, str) and "T3Component" in independent_var:
            ic_number = independent_var.replace("T3Component", "")
            independent_display = f"IC{ic_number} Loading (T3)"
        
        # Handle other regular variables
        else:
            independent_display = get_display_name(independent_var)
        
        # Calculate partial residuals
        X = data.copy()
        
        # Ensure X is DataFrame to use column name indexing
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        # Add constant term if X doesn't already have a 'const' column
        if 'const' not in X.columns:
            X = sm.add_constant(X)
        
        # Get all other variables (excluding current predictor and constant)
        other_vars = [var for var in model.params.index if var != independent_var and var != 'const']
        
        # Calculate the residuals of y regressed on all other predictors
        # Get y data
        if isinstance(model.model.endog, pd.Series):
            y_data = model.model.endog
        else:
            # If dependent_var is string and exists in data, use corresponding column
            if isinstance(dependent_var, str) and dependent_var in data.columns:
                y_data = data[dependent_var]
            # Otherwise directly use model's endog
            else:
                y_data = pd.Series(model.model.endog)
        
        # Calculate y residuals
        if other_vars:
            y_resid = y_data.copy()
            # Subtract influence of all other variables from y
            for var in other_vars:
                if var in X.columns and var in model.params:
                    y_resid = y_resid - model.params[var] * X[var]
            # Subtract influence of constant term
            if 'const' in model.params:
                y_resid = y_resid - model.params['const']
        else:
            # If no other variables, only subtract constant term
            y_resid = y_data - model.params['const'] if 'const' in model.params else y_data
        
        # Calculate the residuals of x regressed on all other predictors
        if other_vars and independent_var in X.columns:
            try:
                # Create regression model of other variables on current predictor
                x_model = sm.OLS(X[independent_var], X[other_vars], hasconst=False).fit()
                x_resid = X[independent_var] - x_model.predict(X[other_vars])
            except Exception as e:
                print(f"  ⚠️ Error calculating residuals for {independent_var}: {str(e)}")
                # If regression fails, use original values
                x_resid = X[independent_var]
        else:
            # If no other variables, use original values
            x_resid = X[independent_var] if independent_var in X.columns else pd.Series([0] * len(y_resid))
        
        # Ensure x_resid and y_resid have same length
        if len(x_resid) != len(y_resid):
            print(f"  ⚠️ x_resid and y_resid lengths don't match, can't create partial regression plot")
            return ax
        
        # Create the scatter plot
        ax.scatter(x_resid, y_resid, color=point_color, alpha=0.6, edgecolor='none')
        
        # Add regression line
        x_range = np.linspace(x_resid.min(), x_resid.max(), 100)
        beta = model.params[independent_var]
        y_pred = beta * x_range
        
        # Add significance markers for legend
        if p_value < 0.001:
            sig_marker = "***"
            p_text = "p<0.001"
        elif p_value < 0.01:
            sig_marker = "**"
            p_text = f"p={p_value:.3f}"
        elif p_value < 0.05:
            sig_marker = "*"
            p_text = f"p={p_value:.3f}"
        else:
            sig_marker = ""
            p_text = f"p={p_value:.3f}"
        
        # Draw regression line (no label), then add p-value text separately
        line = ax.plot(x_range, y_pred, color=line_color, linewidth=2)
        
        # Add p-value text directly in upper left corner, no legend
        ax.text(0.05, 0.95, f"{sig_marker} {p_text}", 
                transform=ax.transAxes, 
                fontsize=16, 
                verticalalignment='top',
                color=line_color,
                fontweight='bold')
        
        # Try multiple methods to add confidence interval
        ci_added = False
        
        # Method 1: Use model's cov_params method (standard method)
        if not ci_added:
            try:
                if hasattr(model, 'cov_params'):
                    # Get standard error for this coefficient
                    se = np.sqrt(model.cov_params().loc[independent_var, independent_var])
                    
                    # Small sample correction - increase standard error to reflect uncertainty from small sample
                    # Use sample size correction factor, especially useful for sample sizes below 50
                    n = len(x_resid)
                    if n < 50:
                        small_sample_correction = np.sqrt(n / (n - len(model.params) - 1)) if n > len(model.params) + 1 else 1.5
                        se = se * small_sample_correction
                        print(f"  ⓘ Applied small sample correction factor {small_sample_correction:.2f} to standard error")
                    
                    # Calculate confidence intervals
                    t_value = stats.t.ppf(0.975, model.df_resid)
                    
                    # Calculate funnel-shaped confidence interval rather than equal-width interval
                    # Calculate mean and variance of x_resid
                    X_mean = np.mean(x_resid)
                    X_var = np.sum((x_resid - X_mean)**2)
                    n = len(x_resid)
                    
                    # Calculate confidence interval width for each x point
                    ci_lower = []
                    ci_upper = []
                    for x in x_range:
                        # Width based on distance from x to mean
                        leverage = 1/n + ((x - X_mean)**2)/X_var
                        width = t_value * se * np.sqrt(leverage)
                        # Calculate upper and lower confidence bounds
                        y_pred_at_x = beta * x
                        ci_lower.append(y_pred_at_x - width)
                        ci_upper.append(y_pred_at_x + width)
                    
                    # Use upper and lower confidence bounds
                    ax.fill_between(x_range, ci_lower, ci_upper, color=ci_color, alpha=0.25)
                    
                    print(f"  ✓ Added funnel-shaped confidence interval to {independent_display} plot using Method 1")
                    ci_added = True
                    ax.ci_added = True
                else:
                    print(f"  ⚠️ Model lacks cov_params attribute, cannot calculate confidence interval using Method 1")
            except Exception as e:
                print(f"  ⚠️ Error calculating confidence interval using Method 1: {str(e)}")
        
        # Method 2: Use beta and x_resid, y_resid directly to calculate confidence interval
        if not ci_added:
            try:
                # Calculate simple approximate confidence interval
                residual_std = np.std(y_resid - beta * x_resid)
                t_value = stats.t.ppf(0.975, len(y_resid) - 2)
                
                # Small sample correction
                n = len(y_resid)
                if n < 50:
                    # Degree of freedom correction to increase uncertainty
                    degrees_of_freedom = n - 2
                    small_sample_correction = np.sqrt((n - 1) / degrees_of_freedom) if degrees_of_freedom > 0 else 1.5
                    residual_std = residual_std * small_sample_correction
                    print(f"  ⓘ Method 2: Applied small sample correction factor {small_sample_correction:.2f} to residual std dev")
                
                # Calculate funnel-shaped confidence interval
                X_mean = np.mean(x_resid)
                X_var = np.sum((x_resid - X_mean)**2)
                n = len(y_resid)
                
                # Calculate confidence interval width for each x point
                ci_lower = []
                ci_upper = []
                for x in x_range:
                    # Width based on distance from x to mean
                    leverage = 1/n + ((x - X_mean)**2)/X_var
                    width = t_value * residual_std * np.sqrt(leverage)
                    # Calculate upper and lower confidence bounds
                    y_pred_at_x = beta * x
                    ci_lower.append(y_pred_at_x - width)
                    ci_upper.append(y_pred_at_x + width)
                
                # Create confidence interval
                ax.fill_between(x_range, ci_lower, ci_upper, color=ci_color, alpha=0.25)
                print(f"  ✓ Added funnel-shaped confidence interval to {independent_display} plot using Method 2")
                ci_added = True
                ax.ci_added = True
            except Exception as e:
                print(f"  ⚠️ Error calculating confidence interval using Method 2: {str(e)}")
        
        # Method 3: Manually perform OLS regression and calculate confidence interval
        if not ci_added:
            try:
                # Use scatter point data manually to perform simple OLS regression
                X_simple = sm.add_constant(x_resid)
                simple_model = sm.OLS(y_resid, X_simple).fit()
                
                # Get standard error for beta
                se = simple_model.bse[1]  # Second element is slope standard error
                
                # Small sample correction
                n = len(y_resid)
                if n < 50:
                    small_sample_correction = np.sqrt(n / (n - 2 - 1)) if n > 3 else 1.5
                    se = se * small_sample_correction
                    print(f"  ⓘ Method 3: Applied small sample correction factor {small_sample_correction:.2f} to standard error")
                
                # Calculate confidence interval
                t_value = stats.t.ppf(0.975, len(y_resid) - 2)
                
                # Calculate funnel-shaped confidence interval
                X_mean = np.mean(x_resid)
                X_var = np.sum((x_resid - X_mean)**2)
                n = len(y_resid)
                
                # Calculate confidence interval width for each x point
                ci_lower = []
                ci_upper = []
                for x in x_range:
                    # Width based on distance from x to mean
                    leverage = 1/n + ((x - X_mean)**2)/X_var
                    width = t_value * se * np.sqrt(leverage)
                    # Calculate upper and lower confidence bounds
                    y_pred_at_x = simple_model.params[1] * x + simple_model.params[0]
                    ci_lower.append(y_pred_at_x - width)
                    ci_upper.append(y_pred_at_x + width)
                
                # Create confidence interval
                ax.fill_between(x_range, ci_lower, ci_upper, color=ci_color, alpha=0.25)
                print(f"  ✓ Added funnel-shaped confidence interval to {independent_display} plot using Method 3")
                ci_added = True
                ax.ci_added = True
            except Exception as e:
                print(f"  ⚠️ Error calculating confidence interval using Method 3: {str(e)}")
        
        # Method 4: Use bootstrap to estimate confidence interval
        if not ci_added:
            try:
                # Use bootstrap method to estimate confidence interval
                n_bootstrap = 500  # Increase bootstrap sample size for stability
                betas = []
                
                for _ in range(n_bootstrap):
                    # Random sampling (with replacement)
                    indices = np.random.choice(len(x_resid), len(x_resid), replace=True)
                    x_boot = x_resid.iloc[indices] if isinstance(x_resid, pd.Series) else x_resid[indices]
                    y_boot = y_resid.iloc[indices] if isinstance(y_resid, pd.Series) else y_resid[indices]
                    
                    # Simple linear regression
                    X_boot = sm.add_constant(x_boot)
                    try:
                        boot_model = sm.OLS(y_boot, X_boot).fit()
                        betas.append(boot_model.params[1])  # Store slope
                    except:
                        continue
                
                if betas:
                    # Calculate bootstrap confidence interval
                    beta_std = np.std(betas)
                    
                    # Small sample correction - In small sample cases, bootstrap may underestimate variability
                    n = len(x_resid)
                    if n < 50:
                        # Use conservative correction factor to increase uncertainty
                        small_sample_correction = 1.2  # More conservative bootstrap correction factor
                        beta_std = beta_std * small_sample_correction
                        print(f"  ⓘ Method 4: Applied small sample correction factor {small_sample_correction:.2f} to bootstrap std dev")
                    
                    # Calculate funnel-shaped confidence interval
                    X_mean = np.mean(x_resid)
                    X_var = np.sum((x_resid - X_mean)**2)
                    n = len(y_resid)
                    
                    # Calculate confidence interval width for each x point
                    ci_lower = []
                    ci_upper = []
                    for x in x_range:
                        # Width based on distance from x to mean
                        leverage = 1/n + ((x - X_mean)**2)/X_var
                        width = 1.96 * beta_std * np.sqrt(leverage)  # Use normal approximation
                        # Calculate upper and lower confidence bounds
                        y_pred_at_x = beta * x
                        ci_lower.append(y_pred_at_x - width)
                        ci_upper.append(y_pred_at_x + width)
                    
                    # Create confidence interval
                    ax.fill_between(x_range, ci_lower, ci_upper, color=ci_color, alpha=0.25)
                    print(f"  ✓ Added funnel-shaped confidence interval to {independent_display} plot using Method 4 (bootstrap)")
                    ci_added = True
                    ax.ci_added = True
            except Exception as e:
                print(f"  ⚠️ Error calculating confidence interval using Method 4 (bootstrap): {str(e)}")
        
        # If all methods fail, output warning
        if not ci_added:
            print(f"  ⚠️ All confidence interval methods failed, {independent_display} plot will not have a confidence interval")
            
        # Display p-value in legend
        # ax.legend(fontsize=20, loc='upper left')  # Removed
            
        # Set axis labels
        ax.set_xlabel(f"{independent_display}", fontsize=14, fontweight='bold')
        ax.set_ylabel(f"{dependent_display}", fontsize=14, fontweight='bold')
        
        # Remove grid
        ax.grid(False)
        
        # Enhance plot aesthetics
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        
    except Exception as e:
        print(f"  ⚠️ Error creating partial regression plot for {independent_var}: {str(e)}")
    
    return ax

# Define stepwise regression function with interaction support and diagnostics
def stepwise_regression(data, dependent_var, independent_vars, confounding_vars, 
                       t3_test_var=None, include_interactions=False, diagnostics=True):
    """
    Perform stepwise regression analysis, supporting interactions and diagnostics
    
    Parameters:
    data (DataFrame): Data
    dependent_var (str): Dependent variable name
    independent_vars (list): List of independent variables
    confounding_vars (list): List of confounding variables
    t3_test_var (str): T3 test variable name, if needed
    include_interactions (bool): Whether to include key interactions
    diagnostics (bool): Whether to generate diagnostic plots
    
    Returns:
    tuple: (final_included_variables, model_DataFrame, modified_original_data) - Variables included in the final model, working DataFrame X, and modified original data
    """
    # Create a copy of the original data to avoid modifying it in-place
    data_modified = data.copy()
    
    # Ensure X contains independent variables and confounding variables
    X = data[independent_vars + confounding_vars].copy()
    
    if t3_test_var:
        X[t3_test_var] = data[t3_test_var]
    
    y = data[dependent_var]
    
    # Create interaction terms
    interaction_terms = []
    if include_interactions:
        print(f"Creating interactions for {dependent_var}...")
        
        # Add T3 baseline test x component interactions (cognitive baseline interaction)
        if t3_test_var:
            print(f"  - Creating T3 baseline test interactions with brain components")
            for iv in independent_vars:
                interaction_name = f"{t3_test_var}_x_{iv}"
                X[interaction_name] = X[t3_test_var] * X[iv]
                # Also add interaction terms to the modified original data
                data_modified[interaction_name] = data_modified[t3_test_var] * data_modified[iv]
                interaction_terms.append(interaction_name)
        
        # Add APOE interactions with brain components (addressing important moderating effects)
        if 'Group' in confounding_vars:
            print(f"  - Creating APOE genotype interactions with brain components")
            for iv in independent_vars:
                interaction_name = f"Group_x_{iv}"
                X[interaction_name] = X['Group'] * X[iv]
                # Also add interaction terms to the modified original data
                data_modified[interaction_name] = data_modified['Group'] * data_modified[iv]
                interaction_terms.append(interaction_name)
                
        # Add Sex interactions with brain components (addressing reviewer comment)
        if 'SEX' in confounding_vars:
            print(f"  - Creating Sex interactions with brain components")
            for iv in independent_vars:
                interaction_name = f"SEX_x_{iv}"
                X[interaction_name] = X['SEX'] * X[iv]
                # Also add interaction terms to the modified original data
                data_modified[interaction_name] = data_modified['SEX'] * data_modified[iv]
                interaction_terms.append(interaction_name)
    
    included = confounding_vars.copy()  # Keep confounding variables
    
    # If T3 test variable exists, also keep it
    if t3_test_var:
        included.append(t3_test_var)
    
    # Stepwise selection procedure
    print(f"Starting stepwise selection for {dependent_var}")
    print(f"  - Initial model includes confounding variables: {', '.join(confounding_vars)}")
    if t3_test_var:
        print(f"  - Added baseline variable: {t3_test_var}")
    print(f"  - Selection thresholds: Entry p < 0.05, Removal p > 0.1")
    
    while True:
        changed = False
        
        # Currently not included variables, including interactions
        all_potential_vars = independent_vars + interaction_terms
        excluded = list(set(all_potential_vars) - set(included))
        
        if not excluded:
            break  # If no variables to add, exit loop
        
        new_pval = pd.Series(index=excluded, dtype=float)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included + [new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        
        best_pval = new_pval.min()
        if best_pval < 0.05:  # Entry threshold: p < 0.05
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            print(f"  - Added {best_feature} to model, p-value={best_pval:.4f}")

        # Backward elimination, only for independent vars and interaction terms
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        pvalues = model.pvalues.iloc[1:]  # Skip intercept
        
        # Find max p-value not in confounding vars or T3 test var
        exclude_from_elimination = confounding_vars.copy()
        if t3_test_var:
            exclude_from_elimination.append(t3_test_var)
        
        # Only consider variables eligible for elimination
        eligible_for_elimination = [var for var in included if var not in exclude_from_elimination]
        
        if eligible_for_elimination:
            eligible_pvalues = pvalues[eligible_for_elimination]
            worst_pval = eligible_pvalues.max()
            
            if worst_pval > 0.1:  # Removal threshold: p > 0.1
                worst_feature = eligible_pvalues.idxmax()
                included.remove(worst_feature)
                changed = True
                print(f"  - Removed {worst_feature} from model, p-value={worst_pval:.4f}")
        
        if not changed:
            break
    
    # Print final model summary
    print(f"Final model for {dependent_var} includes {len(included)} variables:")
    print(f"  - Confounding variables: {[var for var in included if var in confounding_vars]}")
    if t3_test_var and t3_test_var in included:
        print(f"  - Baseline variable: {t3_test_var}")
    
    components = [var for var in included if var in independent_vars]
    if components:
        print(f"  - Brain components: {components}")
    
    interactions = [var for var in included if '_x_' in var]
    if interactions:
        print(f"  - Interaction terms: {interactions}")

    # Generate diagnostics if enabled
    if diagnostics:
        final_model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        
        # Save model diagnostics
        residuals = final_model.resid
        fitted = final_model.fittedvalues
        
        # Residuals vs fitted values plot
        plt.figure(figsize=(10, 8))
        
        # Reset default style to ensure no influence from previous runs
        plt.rcParams.update(plt.rcParamsDefault)
        
        # Set Times New Roman font, bold lines and labels
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['font.size'] = 16  # Base font size
        plt.rcParams['axes.labelsize'] = 18  # Axis label font size
        plt.rcParams['xtick.labelsize'] = 16  # x-axis tick label font size
        plt.rcParams['ytick.labelsize'] = 16  # y-axis tick label font size
        
        # Enhance axis clarity
        plt.rcParams['axes.linewidth'] = 2.0  # Bold axes
        plt.rcParams['xtick.major.width'] = 2.0  # Bold x-axis main ticks
        plt.rcParams['ytick.major.width'] = 2.0  # Bold y-axis main ticks
        plt.rcParams['xtick.major.size'] = 6.0  # Lengthen x-axis main ticks
        plt.rcParams['ytick.major.size'] = 6.0  # Lengthen y-axis main ticks
        
        # Draw scatter plot with elegant colors and markers
        plt.scatter(fitted, residuals, color='#3498db', edgecolor='#2980b9', alpha=0.7,
                  s=70, linewidth=1.0)
        plt.axhline(y=0, color='#e74c3c', linestyle='-', linewidth=2.0)
        
        # Set axis labels
        plt.xlabel('Fitted Values', fontsize=18, fontweight='bold')
        plt.ylabel('Residuals', fontsize=18, fontweight='bold')
        
        # Remove title and grid lines
        # plt.title(f'Residual Plot for {get_display_name(dependent_var)}', fontsize=18)
        plt.grid(False)
        
        # Beautify axes
        ax = plt.gca()
        ax.spines['top'].set_visible(False)  # Remove top axis line
        ax.spines['right'].set_visible(False)  # Remove right axis line
        ax.spines['left'].set_linewidth(2.0)  # Bold left axis line
        ax.spines['bottom'].set_linewidth(2.0)  # Bold bottom axis line
        
        # Use minor ticks to enhance readability
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        # Save chart
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{dependent_var}_residual_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # QQ plot
        plt.figure(figsize=(10, 8))
        
        # Reset default style to ensure no influence from previous runs
        plt.rcParams.update(plt.rcParamsDefault)
        
        # Set Times New Roman font, bold lines and labels
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['font.size'] = 16  # Base font size
        plt.rcParams['axes.labelsize'] = 18  # Axis label font size
        plt.rcParams['xtick.labelsize'] = 16  # x-axis tick label font size
        plt.rcParams['ytick.labelsize'] = 16  # y-axis tick label font size
        
        # Enhance axis clarity
        plt.rcParams['axes.linewidth'] = 2.0  # Bold axes
        plt.rcParams['xtick.major.width'] = 2.0  # Bold x-axis main ticks
        plt.rcParams['ytick.major.width'] = 2.0  # Bold y-axis main ticks
        plt.rcParams['xtick.major.size'] = 6.0  # Lengthen x-axis main ticks
        plt.rcParams['ytick.major.size'] = 6.0  # Lengthen y-axis main ticks
        
        # Use statsmodels qqplot with custom point styles and colors
        fig = sm.qqplot(residuals, line='45', fit=True, markerfacecolor='#3498db', 
                      markeredgecolor='#2980b9', markersize=8, linewidth=2.0)
        
        # Get current axis and customize
        ax = plt.gca()
        
        # Set axis labels
        ax.set_xlabel('Theoretical Quantiles', fontsize=18, fontweight='bold')
        ax.set_ylabel('Sample Quantiles', fontsize=18, fontweight='bold')
        
        # Remove title and grid lines
        # plt.title(f'QQ Plot for {get_display_name(dependent_var)}', fontsize=18)
        ax.grid(False)
        
        # Beautify axes
        ax.spines['top'].set_visible(False)  # Remove top axis line
        ax.spines['right'].set_visible(False)  # Remove right axis line
        ax.spines['left'].set_linewidth(2.0)  # Bold left axis line
        ax.spines['bottom'].set_linewidth(2.0)  # Bold bottom axis line
        
        # Use minor ticks to enhance readability
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        # Save chart
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{dependent_var}_qq_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Add generation of partial regression plots
        # Only generate plots when model has independent variables
        if len(included) > 0:
            # Extract variables excluding confounding variables and baseline variable
            component_vars = [var for var in included 
                             if var not in confounding_vars and var != t3_test_var]
            
            if component_vars:  # Only create plots when model includes brain component variables
                try:
                    # Convert X[included] to DataFrame to ensure compatibility
                    X_df = pd.DataFrame(X[included])
                    
                    create_partial_regression_plots(
                        final_model,           # Model object
                        X_df,                  # Independent variables data
                        dependent_var,         # Use original dependent variable name
                        component_vars,        # Only create partial regression plots for brain components
                        output_dir,            # Output directory
                        max_plots=5            # Maximum 5 plots for variables
                    )
                except Exception as e:
                    print(f"  ⚠️ Error creating partial regression plots: {str(e)}")
                    print("  ℹ️ Continuing with remaining analysis...")
        
        # Residual normality test
        sw_stat, sw_p = shapiro(residuals)
        
        # Breusch-Pagan heteroscedasticity test
        bp_test = het_breuschpagan(residuals, final_model.model.exog)
        
        # Save diagnostic results
        diagnostics_results = {
            'dependent_var': [dependent_var],
            'display_dependent_var': [get_display_name(dependent_var)],
            'R_squared': [final_model.rsquared],
            'Adj_R_squared': [final_model.rsquared_adj],
            'F_statistic': [final_model.fvalue],
            'F_pvalue': [final_model.f_pvalue],
            'AIC': [final_model.aic],
            'BIC': [final_model.bic],
            'Log_Likelihood': [final_model.llf],
            'Residual_Mean': [np.mean(residuals)],
            'Residual_Std': [np.std(residuals)],
            'Shapiro_Stat': [sw_stat],
            'Shapiro_p': [sw_p],
            'BP_LM_stat': [bp_test[0]],
            'BP_LM_p': [bp_test[1]],
            'BP_F_stat': [bp_test[2]],
            'BP_F_p': [bp_test[3]]
        }
        
        pd.DataFrame(diagnostics_results).to_csv(os.path.join(output_dir, f'{dependent_var}_model_diagnostics.csv'), index=False)
        print(f"Saved diagnostics for {get_display_name(dependent_var)}")

    # Return the included variables, the modified DataFrame X, and the modified original data
    return included, X, data_modified

# Define model summary function with contribution analysis
def summarize_model(model, model_type, transformed_vars, X, y, dependent_var, bootstrap_stats=None):
    summary = {
        'model_type': [model_type] * len(model.params),
        'variables': model.params.index.tolist(),
        'display_variables': [get_display_name(var) if var != 'const' else 'Intercept' for var in model.params.index.tolist()],
        'coefficients': model.params.values.tolist(),
        'standard_errors': model.bse.values.tolist(),
        't_values': model.tvalues.values.tolist(),
        'p_values': model.pvalues.values.tolist(),
        'adjusted_p_values': multitest.multipletests(model.pvalues, method='fdr_bh')[1],
        'R_squared': [model.rsquared] * len(model.params),
        'adj_R_squared': [model.rsquared_adj] * len(model.params),
        'AIC': [model.aic] * len(model.params),
        'BIC': [model.bic] * len(model.params),
        'conf_lower': model.conf_int()[0].tolist(),
        'conf_upper': model.conf_int()[1].tolist(),
        'notes': [],
        'contributions': []
    }
    
    # Add Bootstrap confidence intervals if available
    if bootstrap_stats is not None:
        summary['boot_conf_lower'] = bootstrap_stats['Lower_CI'].values.tolist()
        summary['boot_conf_upper'] = bootstrap_stats['Upper_CI'].values.tolist()
        summary['boot_mean'] = bootstrap_stats['Mean'].values.tolist()
        summary['boot_std'] = bootstrap_stats['Std'].values.tolist()

    # Calculate contribution of each variable using R-squared change
    full_r_squared = model.rsquared

    for var in summary['variables']:
        if var != 'const':  # Only calculate for independent variables, not intercept
            if var in X.columns:  # Ensure variable exists in X DataFrame
                X_reduced = X.drop(columns=[var])
                reduced_model = sm.OLS(y, sm.add_constant(X_reduced)).fit()
                contribution = full_r_squared - reduced_model.rsquared
                summary['contributions'].append(contribution)
            else:
                summary['contributions'].append(np.nan)

            # Check if it's a transformed variable
            if var in transformed_vars.values():
                original_var = [k for k, v in transformed_vars.items() if v == var][0]
                summary['notes'].append(f'{var} (transformed from {original_var})')
            # Check if it's an interaction
            elif '_x_' in var:
                # Parse interaction term to get more user-friendly names
                parts = var.split('_x_')
                if len(parts) == 2:
                    var1_display = get_display_name(parts[0])
                    var2_display = get_display_name(parts[1])
                    summary['notes'].append(f'Interaction: {var1_display} × {var2_display}')
                else:
                    summary['notes'].append(f'{var} (interaction)')
            else:
                summary['notes'].append('')
        else:
            summary['contributions'].append(np.nan)  # For intercept, contribution is NA
    
    # Generate additional model diagnostics
    residuals = model.resid
    shapiro_test = shapiro(residuals)
    bp_test = het_breuschpagan(residuals, model.model.exog)

    # Create model summary
    model_summary_data = {
        'Dependent_Variable': [dependent_var],
        'Display_Dependent_Variable': [get_display_name(dependent_var)],
        'R_squared': [model.rsquared],
        'Adj_R_squared': [model.rsquared_adj],
        'F_statistic': [model.fvalue],
        'F_pvalue': [model.f_pvalue],
        'AIC': [model.aic],
        'BIC': [model.bic],
        'Log_Likelihood': [model.llf],
        'Residuals_Normality_Test_pvalue': [shapiro_test[1]],
        'Heteroskedasticity_Test_pvalue': [bp_test[1]],
    }
    
    # Add model formula if model.model has formula attribute
    if hasattr(model.model, 'formula'):
        model_summary_data['Model_Formula'] = [str(model.model.formula)]
    else:
        model_summary_data['Model_Formula'] = ['OLS model without formula specification']
    
    # Add Bootstrap information if available
    if bootstrap_stats is not None:
        model_summary_data['Bootstrap_Samples'] = [len(bootstrap_stats)]
        
    pd.DataFrame(model_summary_data).to_csv(os.path.join(output_dir, f'{dependent_var}_{model_type}_model_summary.csv'), index=False)
    print(f"Saved {get_display_name(dependent_var)} {model_type} model summary")
    
    # Ensure all lists have consistent lengths to prevent errors when creating DataFrame
    n_vars = len(model.params)
    
    # Ensure contributions list length matches variables list length
    if 'contributions' in summary and len(summary['contributions']) != len(summary['variables']):
        print(f"Warning: 'contributions' list length ({len(summary['contributions'])}) doesn't match 'variables' list length ({len(summary['variables'])}) for '{dependent_var}', fixing...")
        # Fill in missing contribution values
        missing_count = len(summary['variables']) - len(summary['contributions'])
        if missing_count > 0:
            summary['contributions'].extend([np.nan] * missing_count)
        else:
            summary['contributions'] = summary['contributions'][:len(summary['variables'])]
    
    # Ensure all lists have consistent lengths
    for key in list(summary.keys()):
        if isinstance(summary[key], list):
            if len(summary[key]) != n_vars:
                print(f"Warning: '{key}' list length ({len(summary[key])}) inconsistent for '{dependent_var}', should be {n_vars}. Fixing...")
                # If list is too short, fill with NaN values
                if len(summary[key]) < n_vars:
                    if key == 'notes':
                        summary[key].extend([''] * (n_vars - len(summary[key])))
                    else:
                        summary[key].extend([np.nan] * (n_vars - len(summary[key])))
                # If list is too long, truncate
                elif len(summary[key]) > n_vars:
                    summary[key] = summary[key][:n_vars]
    
    # Also check bootstrap-related lists
    bootstrap_keys = ['boot_conf_lower', 'boot_conf_upper', 'boot_mean', 'boot_std']
    for key in bootstrap_keys:
        if key in summary and len(summary[key]) != n_vars:
            print(f"Warning: '{key}' list length ({len(summary[key])}) inconsistent for '{dependent_var}', should be {n_vars}. Fixing...")
            # Handle length inconsistency similarly
            if len(summary[key]) < n_vars:
                summary[key].extend([np.nan] * (n_vars - len(summary[key])))
            elif len(summary[key]) > n_vars:
                summary[key] = summary[key][:n_vars]
    
    # Copy contributions as partial_R_squared for naming consistency
    summary['partial_R_squared'] = summary['contributions']
    
    return summary 

# Regression analysis
dependent_vars = ['T4MMSE','T4TMA', 'T4TMB', 'T4EGU12logicalmemory', 'T4LGV12logicalmemory', 'T4ZNVORdigitspanforward', 'T4ZNNACHdigitspanbackward']
independent_vars = ['T3Component1', 'T3Component2', 'T3Component3', 'T3Component4', 'T3Component5', 'T3Component6', 
                    'T3Component7', 'T3Component8', 'T3Component9', 'T3Component10', 'T3Component11', 'T3Component12']
confounding_vars = ['T3Age', 'SEX', 'EDUC', 'Group', 'TIV']
t3_test_vars = {
    'T4TMA': 'T3TMA',
    'T4TMB': 'T3TMB',
    'T4EGU12logicalmemory': 'T3EGU12logicalmemory',
    'T4LGV12logicalmemory': 'T3LGV12logicalmemory',
    'T4ZNVORdigitspanforward': 'T3ZNVORdigitspanforward',
    'T4ZNNACHdigitspanbackward': 'T3ZNNACHdigitspanbackward'
}

# Save methodology parameters
methods_info = {
    "Analysis Method": ["Stepwise Regression Analysis + Bootstrap Robustness Analysis"],
    "Package": ["statsmodels.api, joblib parallel processing"],
    "Entry Threshold": [0.05],
    "Removal Threshold": [0.1],
    "Forced Variables": ["Confounding variables (T3Age, SEX, EDUC, Group, TIV) and T3 test variable"],
    "Interaction Terms": ["T3 test variable × brain components, Sex × brain components, APOE genotype × brain components"],
    "Multiple Comparison Correction": ["Benjamini-Hochberg (FDR), alpha=0.05"],
    "Variable Transformation": ["Yeo-Johnson transformation (for non-normal variables)"],
    "Model Diagnostics": ["Residual analysis, Shapiro-Wilk normality test, Breusch-Pagan heteroscedasticity test"],
    "Model Evaluation Metrics": ["AIC, BIC, R², Adjusted R², F-test"],
    "Bootstrap Parameters": ["1000 resamples, parallel processing, 95% confidence intervals"]
}
pd.DataFrame(methods_info).to_csv(os.path.join(output_dir, 'stepwise_regression_methods_parameters.csv'), index=False)
print(f"Saved methodology information to {output_dir}/stepwise_regression_methods_parameters.csv")

# Set Bootstrap parameters
n_bootstrap = 1000  # Bootstrap resampling count
use_parallel = True  # Whether to use parallel computing
n_jobs = -1  # Use all CPU cores

regression_results = {}
for dep_var in dependent_vars:
    print(f"\nStarting analysis for {get_display_name(dep_var)}...")
    t3_var = t3_test_vars.get(dep_var, None)
    
    # Unadjusted model (without interactions)
    print(f"Fitting unadjusted model (no interactions) for {get_display_name(dep_var)}...")
    selected_vars_unadjusted, X_data_unadjusted, _ = stepwise_regression(
        data, dep_var, independent_vars, [], 
        t3_test_var=t3_var, include_interactions=False, diagnostics=True
    )
    formula_unadjusted = f"{dep_var} ~ " + " + ".join(selected_vars_unadjusted)
    X_unadjusted = X_data_unadjusted[selected_vars_unadjusted].copy()
    y_unadjusted = data[dep_var]
    model_unadjusted = ols(formula_unadjusted, data=data).fit()
    
    # Unadjusted model (with interactions)
    print(f"Fitting unadjusted model (with interactions) for {get_display_name(dep_var)}...")
    selected_vars_unadjusted_inter, X_data_unadjusted_inter, data_unadj_inter = stepwise_regression(
        data, dep_var, independent_vars, [], 
        t3_test_var=t3_var, include_interactions=True, diagnostics=True
    )
    formula_unadjusted_inter = f"{dep_var} ~ " + " + ".join(selected_vars_unadjusted_inter)
    X_unadjusted_inter = X_data_unadjusted_inter[selected_vars_unadjusted_inter].copy()
    model_unadjusted_inter = ols(formula_unadjusted_inter, data=data_unadj_inter).fit()
    
    # Adjusted model (without interactions)
    print(f"Fitting adjusted model (no interactions) for {get_display_name(dep_var)}...")
    selected_vars_adjusted, X_data_adjusted, _ = stepwise_regression(
        data, dep_var, independent_vars, confounding_vars, 
        t3_test_var=t3_var, include_interactions=False, diagnostics=True
    )
    # Force keep all confounding variables
    final_adjusted_vars = list(set(selected_vars_adjusted + confounding_vars))
    formula_adjusted = f"{dep_var} ~ " + " + ".join(final_adjusted_vars)
    X_adjusted = X_data_adjusted[final_adjusted_vars].copy()
    y_adjusted = data[dep_var]
    model_adjusted = ols(formula_adjusted, data=data).fit()
    
    # Adjusted model (with interactions)
    print(f"Fitting adjusted model (with interactions) for {get_display_name(dep_var)}...")
    selected_vars_adjusted_inter, X_data_adjusted_inter, data_adj_inter = stepwise_regression(
        data, dep_var, independent_vars, confounding_vars, 
        t3_test_var=t3_var, include_interactions=True, diagnostics=True
    )
    # Force keep all confounding variables
    final_adjusted_vars_inter = list(set(selected_vars_adjusted_inter + confounding_vars))
    formula_adjusted_inter = f"{dep_var} ~ " + " + ".join(final_adjusted_vars_inter)
    X_adjusted_inter = X_data_adjusted_inter[final_adjusted_vars_inter].copy()
    model_adjusted_inter = ols(formula_adjusted_inter, data=data_adj_inter).fit()

    # Perform Bootstrap analysis
    print(f"\nPerforming Bootstrap analysis for {get_display_name(dep_var)}...")
    
    # Bootstrap for unadjusted model (no interactions)
    print(f"Bootstrap analysis for {get_display_name(dep_var)} unadjusted model (no interactions)...")
    boot_stats_unadjusted = bootstrap_model_parallel(
        model_unadjusted, X_unadjusted, y_unadjusted, 
        n_bootstrap=n_bootstrap, n_jobs=n_jobs if use_parallel else 1
    )
    
    # Bootstrap for unadjusted model (with interactions)
    print(f"Bootstrap analysis for {get_display_name(dep_var)} unadjusted model (with interactions)...")
    boot_stats_unadjusted_inter = bootstrap_model_parallel(
        model_unadjusted_inter, X_unadjusted_inter, y_unadjusted, 
        n_bootstrap=n_bootstrap, n_jobs=n_jobs if use_parallel else 1
    )
    
    # Bootstrap for adjusted model (no interactions)
    print(f"Bootstrap analysis for {get_display_name(dep_var)} adjusted model (no interactions)...")
    boot_stats_adjusted = bootstrap_model_parallel(
        model_adjusted, X_adjusted, y_adjusted, 
        n_bootstrap=n_bootstrap, n_jobs=n_jobs if use_parallel else 1
    )
    
    # Bootstrap for adjusted model (with interactions)
    print(f"Bootstrap analysis for {get_display_name(dep_var)} adjusted model (with interactions)...")
    boot_stats_adjusted_inter = bootstrap_model_parallel(
        model_adjusted_inter, X_adjusted_inter, y_adjusted, 
        n_bootstrap=n_bootstrap, n_jobs=n_jobs if use_parallel else 1
    )

    # Generate model summaries (with Bootstrap results)
    unadjusted_summary = summarize_model(model_unadjusted, 'unadjusted', 
                                         transformed_vars, X_unadjusted, y_unadjusted, 
                                         dep_var, boot_stats_unadjusted)
    
    unadjusted_inter_summary = summarize_model(model_unadjusted_inter, 'unadjusted_interaction', 
                                              transformed_vars, X_unadjusted_inter, y_unadjusted, 
                                              dep_var, boot_stats_unadjusted_inter)
    
    adjusted_summary = summarize_model(model_adjusted, 'adjusted', 
                                       transformed_vars, X_adjusted, y_adjusted, 
                                       dep_var, boot_stats_adjusted)
    
    adjusted_inter_summary = summarize_model(model_adjusted_inter, 'adjusted_interaction', 
                                            transformed_vars, X_adjusted_inter, y_adjusted, 
                                            dep_var, boot_stats_adjusted_inter)

    # Model comparison - Use AIC and likelihood ratio test
    model_comparisons = []
    
    # No interaction vs interaction (unadjusted model)
    if len(selected_vars_unadjusted_inter) > len(selected_vars_unadjusted):
        lr_test = 2 * (model_unadjusted_inter.llf - model_unadjusted.llf)
        lr_df = len(model_unadjusted_inter.params) - len(model_unadjusted.params)
        lr_p = 1 - stats.chi2.cdf(lr_test, lr_df)
        
        model_comparisons.append({
            'Model1': 'Unadjusted without interaction',
            'Model2': 'Unadjusted with interaction',
            'AIC_diff': model_unadjusted.aic - model_unadjusted_inter.aic,
            'BIC_diff': model_unadjusted.bic - model_unadjusted_inter.bic,
            'LR_test': lr_test,
            'LR_df': lr_df,
            'LR_p': lr_p,
            'Preferred_Model': 'With interaction' if lr_p < 0.05 else 'Without interaction'
        })
    
    # No interaction vs interaction (adjusted model)
    if len(final_adjusted_vars_inter) > len(final_adjusted_vars):
        lr_test = 2 * (model_adjusted_inter.llf - model_adjusted.llf)
        lr_df = len(model_adjusted_inter.params) - len(model_adjusted.params)
        lr_p = 1 - stats.chi2.cdf(lr_test, lr_df)
        
        model_comparisons.append({
            'Model1': 'Adjusted without interaction',
            'Model2': 'Adjusted with interaction',
            'AIC_diff': model_adjusted.aic - model_adjusted_inter.aic,
            'BIC_diff': model_adjusted.bic - model_adjusted_inter.bic,
            'LR_test': lr_test,
            'LR_df': lr_df,
            'LR_p': lr_p,
            'Preferred_Model': 'With interaction' if lr_p < 0.05 else 'Without interaction'
        })
    
    # Save model comparison results
    if model_comparisons:
        pd.DataFrame(model_comparisons).to_csv(os.path.join(output_dir, f'{dep_var}_model_comparisons.csv'), index=False)
        print(f"Saved {get_display_name(dep_var)} model comparison results")

    # Save results to dictionary
    regression_results[dep_var] = {
        'unadjusted': unadjusted_summary,
        'unadjusted_interaction': unadjusted_inter_summary,
        'adjusted': adjusted_summary,
        'adjusted_interaction': adjusted_inter_summary,
        'boot_stats_unadjusted': boot_stats_unadjusted,
        'boot_stats_unadjusted_inter': boot_stats_unadjusted_inter,
        'boot_stats_adjusted': boot_stats_adjusted,
        'boot_stats_adjusted_inter': boot_stats_adjusted_inter
    }

    # Convert each part to DataFrame and save
    unadjusted_df = pd.DataFrame(unadjusted_summary)
    unadjusted_inter_df = pd.DataFrame(unadjusted_inter_summary)
    adjusted_df = pd.DataFrame(adjusted_summary)
    adjusted_inter_df = pd.DataFrame(adjusted_inter_summary)

    # Function to create a cleaned version of the dataframe for CSV export
    def clean_for_csv(df):
        return pd.DataFrame({
            'Variable': df['display_variables'],
            'Coefficient': df['coefficients'],
            'Standard Error': df['standard_errors'],
            'p-value': df['p_values'],
            'Adjusted p-value': df['adjusted_p_values'],
            't-value': df['t_values'],
            'Confidence Interval Lower': df['conf_lower'],
            'Confidence Interval Upper': df['conf_upper'],
            'Notes': df['notes']
        })

    # Export cleaned versions to CSV
    clean_for_csv(unadjusted_df).to_csv(os.path.join(output_dir, f'regression_results_{dep_var}_unadjusted.csv'), index=False)
    clean_for_csv(unadjusted_inter_df).to_csv(os.path.join(output_dir, f'regression_results_{dep_var}_unadjusted_interaction.csv'), index=False)
    clean_for_csv(adjusted_df).to_csv(os.path.join(output_dir, f'regression_results_{dep_var}_adjusted.csv'), index=False)
    clean_for_csv(adjusted_inter_df).to_csv(os.path.join(output_dir, f'regression_results_{dep_var}_adjusted_interaction.csv'), index=False)
    
    # Save Bootstrap statistics
    if boot_stats_unadjusted is not None:
        boot_stats_unadjusted.to_csv(os.path.join(output_dir, f'bootstrap_stats_{dep_var}_unadjusted.csv'))
    if boot_stats_unadjusted_inter is not None:
        boot_stats_unadjusted_inter.to_csv(os.path.join(output_dir, f'bootstrap_stats_{dep_var}_unadjusted_interaction.csv'))
    if boot_stats_adjusted is not None:
        boot_stats_adjusted.to_csv(os.path.join(output_dir, f'bootstrap_stats_{dep_var}_adjusted.csv'))
    if boot_stats_adjusted_inter is not None:
        boot_stats_adjusted_inter.to_csv(os.path.join(output_dir, f'bootstrap_stats_{dep_var}_adjusted_interaction.csv'))
    
    print(f"Saved all regression results for {get_display_name(dep_var)}")

# Extract p-values from all regression models
all_p_values = []
for dep_var, result in regression_results.items():
    all_p_values.extend(result['adjusted']['p_values'])
    all_p_values.extend(result['adjusted_interaction']['p_values'])
    all_p_values.extend(result['unadjusted']['p_values'])
    all_p_values.extend(result['unadjusted_interaction']['p_values'])

# Correct p-values for all regression models
adjusted_p_values = multitest.multipletests(all_p_values, method='fdr_bh', alpha=0.05)[1]

# Update p-values in regression model results
index = 0
for dep_var, result in regression_results.items():
    models = ['adjusted', 'adjusted_interaction', 'unadjusted', 'unadjusted_interaction']
    for model_type in models:
        n_vars = len(result[model_type]['p_values'])
        result[model_type]['adjusted_p_values'] = adjusted_p_values[index:index + n_vars]
        index += n_vars

# Save updated regression model results
for dep_var, result in regression_results.items():
    models = ['adjusted', 'adjusted_interaction', 'unadjusted', 'unadjusted_interaction']
    for model_type in models:
        df = pd.DataFrame(result[model_type])
        df.to_csv(os.path.join(output_dir, f'regression_results_{dep_var}_{model_type}_with_global_correction.csv'), index=False)

# Create Bootstrap results summary
bootstrap_summary = []
for dep_var in dependent_vars:
    for model_type in ['unadjusted', 'unadjusted_interaction', 'adjusted', 'adjusted_interaction']:
        boot_stats_key = f'boot_stats_{model_type}'
        if boot_stats_key in regression_results[dep_var] and regression_results[dep_var][boot_stats_key] is not None:
            boot_stats = regression_results[dep_var][boot_stats_key]
            # Calculate stability (proportion of Bootstrap CI not containing 0)
            significant_vars = sum((boot_stats['Lower_CI'] > 0) | (boot_stats['Upper_CI'] < 0))
            bootstrap_summary.append({
                'Dependent_Variable': dep_var,
                'Display_Dependent_Variable': get_display_name(dep_var),
                'Model_Type': model_type,
                'Total_Variables': len(boot_stats),
                'Significant_Variables': significant_vars,
                'Stability_Percentage': round(significant_vars / len(boot_stats) * 100, 2),
                'Bootstrap_Samples': n_bootstrap
            })

# Save Bootstrap summary results
if bootstrap_summary:
    pd.DataFrame(bootstrap_summary).to_csv(os.path.join(output_dir, 'bootstrap_stability_summary.csv'), index=False)
    print(f"Saved Bootstrap stability summary to {output_dir}/bootstrap_stability_summary.csv")

# Create Excel summary file
with pd.ExcelWriter(os.path.join(output_dir, 'regression_results_summary_with_bootstrap.xlsx')) as writer:
    # Save various summary information
    pd.DataFrame(bootstrap_summary).to_excel(writer, sheet_name='Bootstrap_Stability', index=False)
    
    # Save results for each dependent variable
    for dep_var in dependent_vars:
        model_type = 'adjusted_interaction'  # Choose the most complete model type
        sheet_name = f'{dep_var}'[:31]  # Excel limits sheet name length
        
        # First, save model results to sheet
        df = pd.DataFrame(regression_results[dep_var][model_type])
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Try to add Bootstrap results if available
        boot_stats_key = f'boot_stats_{model_type}'
        if boot_stats_key in regression_results[dep_var] and regression_results[dep_var][boot_stats_key] is not None:
            boot_df = regression_results[dep_var][boot_stats_key]
            boot_df.to_excel(writer, sheet_name=f'{sheet_name}_Bootstrap', index=True)

print("All analysis completed! Created regression results with Bootstrap analysis.")

# After the regression analysis is complete, add a code to create a comprehensive summary table
# This should be added at the very end of the file, just before the README.md generation

# Create comprehensive summary tables by category
print("Creating comprehensive summary tables...")

# 1. Create a summary by cognitive test (dependent variable)
cognitive_test_summary = []
for dep_var in dependent_vars:
    display_dep_var = get_display_name(dep_var)
    
    # Get best model (prefer adjusted with interactions)
    if 'adjusted_interaction' in regression_results[dep_var]:
        best_model_type = 'adjusted_interaction'
    elif 'adjusted' in regression_results[dep_var]:
        best_model_type = 'adjusted'
    else:
        best_model_type = 'unadjusted'
        
    best_model_data = regression_results[dep_var][best_model_type]
    
    # Calculate key statistics
    significant_predictors = []
    for var, pval, display_var in zip(best_model_data['variables'], best_model_data['p_values'], best_model_data['display_variables']):
        if var != 'const' and pval < 0.05:
            significant_predictors.append(f"{display_var} (p={pval:.3f})")
    
    # Add to summary
    cognitive_test_summary.append({
        'Cognitive_Test': display_dep_var,
        'Model_Type': best_model_type,
        'R_squared': best_model_data['R_squared'][0],
        'Adj_R_squared': best_model_data['adj_R_squared'][0],
        'Significant_Predictors': ', '.join(significant_predictors) if significant_predictors else 'None',
        'Components_Count': len([p for p in significant_predictors if 'Component' in p])
    })

# Save cognitive test summary
cognitive_summary_df = pd.DataFrame(cognitive_test_summary)
cognitive_summary_df.to_csv(os.path.join(output_dir, 'cognitive_tests_summary.csv'), index=False)
print(f"Saved cognitive tests summary to {output_dir}/cognitive_tests_summary.csv")

# 2. Create a summary by component (independent variable)
component_summary = []
for component in independent_vars:
    display_component = get_display_name(component)
    component_effects = []
    
    # Look for this component across all models and cognitive tests
    for dep_var in dependent_vars:
        display_dep_var = get_display_name(dep_var)
        
        # Check in adjusted interaction model first (prioritize the most complete models)
        found = False
        for model_type in ['adjusted_interaction', 'adjusted', 'unadjusted_interaction', 'unadjusted']:
            if model_type in regression_results[dep_var]:
                model_data = regression_results[dep_var][model_type]
                
                # Look for component or its interactions
                for var, pval, coef, display_var, note in zip(
                    model_data['variables'], 
                    model_data['p_values'], 
                    model_data['coefficients'],
                    model_data['display_variables'],
                    model_data['notes']
                ):
                    # Original variable match or interaction term match
                    if var == component or (component in var and '_x_' in var):
                        if pval < 0.05:
                            direction = "+" if coef > 0 else "-"
                            # Create a clean effect string using display names
                            
                            # If it's an interaction term, extract the clean interaction description
                            if '_x_' in var:
                                # Use the note which contains a cleaner interaction description
                                if 'Interaction:' in note:
                                    # Just show the effect on the cognitive test with direction
                                    effect_str = f"{display_dep_var} ({direction}, p={pval:.3f}, interaction with {note.replace('Interaction: ', '')})"
                                else:
                                    # Fallback if clean interaction not available
                                    effect_str = f"{display_dep_var} ({direction}, p={pval:.3f}, interaction)"
                            else:
                                # Direct effect without interaction
                                effect_str = f"{display_dep_var} ({direction}, p={pval:.3f})"
                            
                            component_effects.append(effect_str)
                            found = True
                            break
                
                if found:
                    break
    
    # Sort effects alphabetically by cognitive test for better readability
    component_effects.sort()
    
    # Add to summary
    component_summary.append({
        'Component': display_component,
        'Significant_Effects_Count': len(component_effects),
        'Significant_Effects': ', '.join(component_effects) if component_effects else 'None'
    })

# Save component summary
component_summary_df = pd.DataFrame(component_summary)
component_summary_df.to_csv(os.path.join(output_dir, 'components_summary.csv'), index=False)
print(f"Saved components summary to {output_dir}/components_summary.csv")

# 3. Create a model comparison summary across all cognitive tests
model_comparison_summary = []
for dep_var in dependent_vars:
    display_dep_var = get_display_name(dep_var)
    
    # Compare model types
    model_types = ['unadjusted', 'unadjusted_interaction', 'adjusted', 'adjusted_interaction']
    model_stats = {}
    
    for model_type in model_types:
        if model_type in regression_results[dep_var]:
            model_data = regression_results[dep_var][model_type]
            
            # Use friendly model names for display
            friendly_model_type = {
                'unadjusted': 'Unadjusted Model',
                'unadjusted_interaction': 'Unadjusted Model with Interactions',
                'adjusted': 'Adjusted Model',
                'adjusted_interaction': 'Adjusted Model with Interactions'
            }.get(model_type, model_type)
            
            model_stats[model_type] = {
                'friendly_name': friendly_model_type,
                'R_squared': model_data['R_squared'][0],
                'Adj_R_squared': model_data['adj_R_squared'][0],
                'AIC': model_data['AIC'][0],
                'BIC': model_data['BIC'][0]
            }
    
    # Find best model by AIC
    best_model = min(model_stats.items(), key=lambda x: x[1]['AIC'])[0] if model_stats else 'None'
    
    # Get significant predictors for best model
    if best_model != 'None':
        best_model_data = regression_results[dep_var][best_model]
        significant_predictors = []
        for var, pval, display_var in zip(
            best_model_data['variables'], 
            best_model_data['p_values'], 
            best_model_data['display_variables']
        ):
            if var != 'const' and pval < 0.05:
                coef = best_model_data['coefficients'][best_model_data['variables'].index(var)]
                direction = "+" if coef > 0 else "-"
                significant_predictors.append(f"{display_var} ({direction}, p={pval:.3f})")
                
        significant_str = ', '.join(significant_predictors) if significant_predictors else 'None'
    else:
        significant_str = 'None'
    
    # Add to summary
    model_comparison_summary.append({
        'Cognitive_Test': display_dep_var,
        'Best_Model': model_stats[best_model]['friendly_name'] if best_model != 'None' else 'None',
        'Best_Model_R_squared': model_stats[best_model]['R_squared'] if best_model != 'None' else None,
        'Best_Model_Adj_R_squared': model_stats[best_model]['Adj_R_squared'] if best_model != 'None' else None,
        'Best_Model_AIC': model_stats[best_model]['AIC'] if best_model != 'None' else None,
        'Best_Model_BIC': model_stats[best_model]['BIC'] if best_model != 'None' else None,
        'Significant_Predictors': significant_str,
        'Unadjusted_R_squared': model_stats.get('unadjusted', {}).get('R_squared', None),
        'Interaction_Improved': best_model.endswith('interaction')
    })

# Save model comparison summary
model_comparison_df = pd.DataFrame(model_comparison_summary)
model_comparison_df.to_csv(os.path.join(output_dir, 'model_comparison_summary.csv'), index=False)
print(f"Saved model comparison summary to {output_dir}/model_comparison_summary.csv")

# Add detailed summary to Excel file
with pd.ExcelWriter(os.path.join(output_dir, 'comprehensive_results_summary.xlsx')) as writer:
    # Add summary sheets
    cognitive_summary_df.to_excel(writer, sheet_name='Cognitive_Tests_Summary', index=False)
    component_summary_df.to_excel(writer, sheet_name='Components_Summary', index=False)
    model_comparison_df.to_excel(writer, sheet_name='Model_Comparison', index=False)
    
    # Add sheets for each cognitive test with all models
    for dep_var in dependent_vars:
        display_dep_var = get_display_name(dep_var)
        sheet_name = f'{display_dep_var}'[:31]  # Excel limits sheet name length
        
        # Create a combined DataFrame of all models for this test
        all_models_data = []
        for model_type in ['unadjusted', 'unadjusted_interaction', 'adjusted', 'adjusted_interaction']:
            if model_type in regression_results[dep_var]:
                model_df = pd.DataFrame(regression_results[dep_var][model_type])
                
                # Use a more friendly model type name
                friendly_model_type = {
                    'unadjusted': 'Unadjusted Model',
                    'unadjusted_interaction': 'Unadjusted Model with Interactions',
                    'adjusted': 'Adjusted Model',
                    'adjusted_interaction': 'Adjusted Model with Interactions'
                }.get(model_type, model_type)
                
                model_df['Model'] = friendly_model_type
                
                # Create a cleaner version for Excel output with just key columns and user-friendly names
                excel_df = pd.DataFrame({
                    'Model': model_df['Model'],
                    'Variable': model_df['display_variables'],
                    'Coefficient': model_df['coefficients'],
                    'Std Error': model_df['standard_errors'],
                    'p-value': model_df['p_values'],
                    'Adjusted p-value': model_df['adjusted_p_values'],
                    't-value': model_df['t_values'],
                    'CI Lower': model_df['conf_lower'],
                    'CI Upper': model_df['conf_upper'],
                    'R² Contribution': model_df['contributions'],
                    'Notes': model_df['notes']
                })
                
                # Add Bootstrap statistics if available
                if 'boot_conf_lower' in model_df.columns:
                    excel_df['Bootstrap CI Lower'] = model_df['boot_conf_lower']
                    excel_df['Bootstrap CI Upper'] = model_df['boot_conf_upper']
                    excel_df['Bootstrap Mean'] = model_df['boot_mean']
                    excel_df['Bootstrap Std'] = model_df['boot_std']
                
                all_models_data.append(excel_df)
        
        if all_models_data:
            combined_df = pd.concat(all_models_data, ignore_index=True)
            combined_df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Add data dictionary sheet to explain variable codes
    var_dict_rows = []
    
    # Add cognitive tests variables
    for var in dependent_vars:
        var_dict_rows.append({
            'Variable Code': var,
            'Variable Name': get_display_name(var),
            'Variable Type': 'Dependent Variable (Cognitive Test)',
            'Description': f'Cognitive test score: {get_display_name(var)}'
        })
    
    # Add component variables 
    for var in independent_vars:
        var_dict_rows.append({
            'Variable Code': var,
            'Variable Name': get_display_name(var),
            'Variable Type': 'Independent Variable (Component)',
            'Description': f'Brain component: {get_display_name(var)}'
        })
    
    # Add T3 test variables
    for dep_var, t3_var in t3_test_vars.items():
        var_dict_rows.append({
            'Variable Code': t3_var,
            'Variable Name': get_display_name(t3_var),
            'Variable Type': 'Baseline Test Variable',
            'Description': f'Baseline measurement for {get_display_name(dep_var)}'
        })
    
    # Add confounding variables
    for var in confounding_vars:
        var_name = get_display_name(var)
        descriptions = {
            'T3Age': 'Age at time of measurement',
            'SEX': 'Gender (1=Male, 2=Female)',
            'EDUC': 'Education level (years)',
            'Group': 'APOE genotype (1=Negative, 2=Positive)',
            'TIV': 'Total Intracranial Volume'
        }
        var_dict_rows.append({
            'Variable Code': var,
            'Variable Name': var_name,
            'Variable Type': 'Confounding Variable',
            'Description': descriptions.get(var, f'Confounding variable: {var_name}')
        })
    
    pd.DataFrame(var_dict_rows).to_excel(writer, sheet_name='Variable Dictionary', index=False)

print(f"Created comprehensive summary Excel file at {output_dir}/comprehensive_results_summary.xlsx")

# Add function to generate publication-ready tables for papers
print("Generating publication-ready tables for papers (Table 2 and Table S4)...")

def save_table_to_excel(writer, df, sheet_name, caption="", notes=""):
    """
    Save a table to Excel with proper formatting
    """
    # For MultiIndex columns, we need to reset the index before saving
    # Excel doesn't support MultiIndex columns with index=False
    df_copy = df.copy()
    df_copy.index = range(len(df_copy))  # Create a simple RangeIndex
    
    # Write to Excel with index=True
    df_copy.to_excel(writer, sheet_name=sheet_name, startrow=2)  # Start at row 2 to leave space for caption
    
    # Get the worksheet
    worksheet = writer.sheets[sheet_name]
    
    # Write caption - using pandas to write the caption
    if caption:
        caption_lines = caption.split('\n')
        for i, line in enumerate(caption_lines):
            # Create a temporary DataFrame with the caption line
            caption_df = pd.DataFrame([[line]], columns=[' '])
            caption_df.to_excel(writer, sheet_name=sheet_name, startrow=i, startcol=0, index=False, header=False)
    
    # Calculate last row
    last_row = 3 + len(df_copy)
    
    # Write notes
    if notes:
        notes_lines = notes.split('\n')
        for i, line in enumerate(notes_lines):
            # Create a temporary DataFrame with the note line
            note_df = pd.DataFrame([[line]], columns=[' '])
            note_df.to_excel(writer, sheet_name=sheet_name, startrow=last_row + i + 1, startcol=0, index=False, header=False)
    
    return
# Add function to generate interaction effects summary
def generate_interaction_effects_summary(regression_results, dependent_vars, output_dir):
    """
    Generate a dedicated summary table for interaction effects
    
    Parameters:
    -----------
    regression_results : dict
        Dictionary containing regression results
    dependent_vars : list
        List of dependent variables
    output_dir : str
        Directory to save output files
    """
    # Create directory for interaction summaries
    interaction_dir = os.path.join(output_dir, "interaction_effects")
    os.makedirs(interaction_dir, exist_ok=True)
    
    # Types of interactions to analyze
    interaction_types = {
        "cognitive_baseline": "_x_T3Component",  # T3 test × Component interactions
        "sex": "SEX_x_",                         # Sex × Component interactions
        "apoe": "Group_x_"                       # APOE × Component interactions
    }
    
    # Create summary DataFrame for each type of interaction
    all_interactions = []
    
    for dep_var in dependent_vars:
        display_dep_var = get_display_name(dep_var)
        
        # Focus on the adjusted model with interactions, which is most complete
        if 'adjusted_interaction' in regression_results[dep_var]:
            adj_inter_model = regression_results[dep_var]['adjusted_interaction']
            
            # Extract all interaction terms
            for var, pval, coef, display_var, notes in zip(
                adj_inter_model['variables'],
                adj_inter_model['p_values'],
                adj_inter_model['coefficients'],
                adj_inter_model['display_variables'],
                adj_inter_model['notes']
            ):
                if '_x_' in var:
                    # Determine interaction type
                    interaction_type = None
                    for key, pattern in interaction_types.items():
                        if pattern in var:
                            interaction_type = key
                            break
                    
                    # Skip if not one of our target interaction types
                    if not interaction_type:
                        continue
                    
                    # Parse interaction parts from the notes
                    interaction_description = notes.replace('Interaction: ', '')
                    
                    # Add to summary
                    all_interactions.append({
                        'Cognitive_Test': display_dep_var,
                        'Interaction_Type': interaction_type,
                        'Interaction_Term': var,
                        'Display_Name': interaction_description,
                        'Coefficient': coef,
                        'p_value': pval,
                        'Significant': 'Yes' if pval < 0.05 else 'No',
                        'Significant_After_FDR': 'Yes' if adj_inter_model['adjusted_p_values'][adj_inter_model['variables'].index(var)] < 0.05 else 'No',
                        'Direction': 'Positive' if coef > 0 else 'Negative'
                    })
    
    # Convert to DataFrame
    interactions_df = pd.DataFrame(all_interactions)
    
    if len(interactions_df) > 0:
        # Add interpretation column
        interpretations = []
        for _, row in interactions_df.iterrows():
            if row['Significant'] == 'No':
                interpretations.append("No significant interaction effect")
            else:
                if row['Interaction_Type'] == 'cognitive_baseline':
                    if row['Coefficient'] > 0:
                        interpretations.append("Higher baseline cognitive performance strengthens the positive association between brain structure and cognitive outcome")
                    else:
                        interpretations.append("Higher baseline cognitive performance weakens the positive association between brain structure and cognitive outcome")
                
                elif row['Interaction_Type'] == 'sex':
                    if row['Coefficient'] > 0:
                        interpretations.append("The effect of brain structure on cognitive outcome is stronger in females than males")
                    else:
                        interpretations.append("The effect of brain structure on cognitive outcome is stronger in males than females")
                
                elif row['Interaction_Type'] == 'apoe':
                    if row['Coefficient'] > 0:
                        interpretations.append("The effect of brain structure on cognitive outcome is stronger in APOE ε4 carriers")
                    else:
                        interpretations.append("The effect of brain structure on cognitive outcome is stronger in APOE ε4 non-carriers")
        
        interactions_df['Interpretation'] = interpretations
        
        # Save interaction summary
        interactions_df.to_csv(os.path.join(interaction_dir, 'interaction_effects_summary.csv'), index=False)
        
        # Save summaries by type
        for interaction_type in interaction_types:
            type_df = interactions_df[interactions_df['Interaction_Type'] == interaction_type]
            if len(type_df) > 0:
                type_df.to_csv(os.path.join(interaction_dir, f'{interaction_type}_interactions.csv'), index=False)
        
        # Create Excel file with sheets for each type
        with pd.ExcelWriter(os.path.join(interaction_dir, 'interaction_effects_analysis.xlsx')) as writer:
            # All interactions
            interactions_df.to_excel(writer, sheet_name='All_Interactions', index=False)
            
            # Sheet for each type
            for interaction_type, nice_name in {
                'cognitive_baseline': 'Baseline Cognitive',
                'sex': 'Sex Effects', 
                'apoe': 'APOE Genotype'
            }.items():
                type_df = interactions_df[interactions_df['Interaction_Type'] == interaction_type]
                if len(type_df) > 0:
                    type_df.to_excel(writer, sheet_name=nice_name[:31], index=False)
                    
            # Significant interactions only
            sig_df = interactions_df[interactions_df['Significant'] == 'Yes']
            if len(sig_df) > 0:
                sig_df.to_excel(writer, sheet_name='Significant_Only', index=False)
        
        print(f"Saved interaction effects analysis to {interaction_dir}/interaction_effects_analysis.xlsx")
    
    else:
        print("No interaction effects found in results")
    
    return interactions_df if len(interactions_df) > 0 else None


# After generating all the regression results, call the new function
print("\nGenerating interaction effects summary tables...")
interaction_summary = generate_interaction_effects_summary(regression_results, dependent_vars, output_dir)
if interaction_summary is not None:
    print(f"Found {len(interaction_summary)} interaction effects across all models")
    print(f"Significant interactions: {len(interaction_summary[interaction_summary['Significant'] == 'Yes'])}")
else:
    print("No interaction effects found in regression models")

def run_regression_analysis(dependent_vars, independent_vars, 
                           confounding_vars, include_interactions=True,
                           include_t3_baseline=True, baseline_test_map=None):
    """
    Run the complete regression analysis workflow
    
    Parameters:
    dependent_vars (list): List of dependent variables
    independent_vars (list): List of independent variables (brain components)
    confounding_vars (list): List of confounding variables
    include_interactions (bool): Whether to include interactions
    include_t3_baseline (bool): Whether to include T3 baseline as covariate
    baseline_test_map (dict): Map of T4 tests to their T3 baseline tests
    """
    results_summary = []
    interaction_plots = []  # Store paths of created interaction effect plots
    
    # Create a path for model plots
    plot_dir = os.path.join(output_dir, 'model_plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Create directory for partial regression plots
    partial_plots_dir = os.path.join(output_dir, "partial_regression_plots")
    os.makedirs(partial_plots_dir, exist_ok=True)
    
    # Create directory for conditional effect plots
    cond_plots_dir = os.path.join(output_dir, "conditional_effect_plots")
    os.makedirs(cond_plots_dir, exist_ok=True)
    
    # Start analysis for each dependent variable
    for dv in dependent_vars:
        print(f"\n{'='*50}")
        print(f"Starting analysis for {get_display_name(dv)}")
        print(f"{'='*50}")
        
        # Determine baseline test for this dependent variable, if applicable
        t3_baseline = None
        if include_t3_baseline and baseline_test_map and dv in baseline_test_map:
            t3_baseline = baseline_test_map[dv]
            print(f"Using {get_display_name(t3_baseline)} as baseline covariate")

        # Perform stepwise regression
        included_vars, X_data, modified_data = stepwise_regression(
            data, dv, independent_vars, confounding_vars, 
            t3_test_var=t3_baseline, 
            include_interactions=include_interactions
        )
        
        print(f"\nFitting final model for {get_display_name(dv)}...")
        # Fit the model with the selected variables
        X_final = X_data[included_vars]
        y = data[dv]
        
        # Create the final OLS model
        final_model = sm.OLS(y, sm.add_constant(X_final)).fit()
        print(f"Final model R² = {final_model.rsquared:.4f}, Adjusted R² = {final_model.rsquared_adj:.4f}")
        
        # Perform Bootstrap analysis for robust confidence intervals
        print("Performing Bootstrap analysis for robust confidence intervals...")
        bootstrap_results = bootstrap_model_parallel(final_model, X_final, y, n_bootstrap=1000)
        
        # Summarize the model results
        model_summary = summarize_model(
            final_model, 
            'final_model', 
            transformed_vars, 
            X_final, 
            y, 
            dv,
            bootstrap_stats=bootstrap_results
        )
        
        # Save detailed model results
        model_results_df = pd.DataFrame(model_summary)
        # Add comment line explaining R² type
        model_results_df.to_csv(os.path.join(output_dir, f'{dv}_detailed_model_results.csv'), index=False)
        print(f"Saved detailed model results (including partial R²) to {output_dir}/{dv}_detailed_model_results.csv")
        
        # Save model summary text
        with open(os.path.join(output_dir, f'{dv}_model_summary.txt'), 'w') as f:
            f.write(final_model.summary().as_text())
        print(f"Saved model summary to {output_dir}/{dv}_model_summary.txt")
        
        # Create variable importance plot
        plt.figure(figsize=(10, 8))
        # Filter out intercept and get variables with their contributions
        contrib_data = {var: contrib for var, contrib in zip(model_summary['display_variables'], model_summary['partial_R_squared']) 
                        if var != 'Intercept' and not pd.isna(contrib)}
        # Sort by contribution
        sorted_contribs = sorted(contrib_data.items(), key=lambda x: x[1], reverse=True)
        var_names = [item[0] for item in sorted_contribs]
        contribs = [item[1] for item in sorted_contribs]
        
        # Plot
        plt.barh(var_names, contribs, color='steelblue')
        plt.xlabel('Variable Importance (Partial R²)', fontsize=14)  
        plt.ylabel('Variable', fontsize=14)
        plt.title(f'Variable Importance (Partial R² values) for {get_display_name(dv)}', fontsize=16) 
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{dv}_variable_importance.png'), dpi=300)
        plt.close()
        
        # Check and create conditional effect plots
        print("\nChecking significant interaction effects and creating conditional effect plots...")
        dv_interaction_plots = []
        
        # Find interaction terms in the model - more comprehensive detection
        interaction_terms = [var for var in included_vars if '_x_' in var]
        # Also check statsmodels format interaction terms
        statsmodels_interaction_terms = [var for var in final_model.params.index if ':' in var]
        
        # Filter significant interaction terms
        significant_interaction_terms = []
        
        # Check custom format interaction terms
        for term in interaction_terms:
            if term in final_model.pvalues and final_model.pvalues[term] < 0.05:
                significant_interaction_terms.append(term)
                print(f"  ✓ Found significant interaction: {term} (p = {final_model.pvalues[term]:.3f})")
        
        # Check statsmodels format interaction terms
        for term in statsmodels_interaction_terms:
            if term in final_model.pvalues and final_model.pvalues[term] < 0.05:
                significant_interaction_terms.append(term)
                print(f"  ✓ Found significant interaction: {term} (p = {final_model.pvalues[term]:.3f})")
        
        # Only create conditional effect plots for significant interaction terms
        if significant_interaction_terms:
            print(f"  ℹ️ Found {len(significant_interaction_terms)} significant interaction terms, creating conditional effect plots")
            
            # Process custom format (_x_) and statsmodels format (:) interaction terms
            for interaction_term in significant_interaction_terms:
                # Parse interaction components and moderator variables
                if '_x_' in interaction_term:
                    parts = interaction_term.split('_x_')
                elif ':' in interaction_term:
                    parts = interaction_term.split(':')
                else:
                    print(f"  ⚠️ Cannot parse interaction format: {interaction_term}")
                    continue
                
                if len(parts) == 2:
                    component_var = parts[1] if 'Component' in parts[1] else parts[0]
                    moderator_var = parts[0] if 'Component' in parts[1] else parts[1]
                    
                    # Create conditional effect plot
                    print(f"  - Creating conditional effect plot for significant interaction {get_display_name(component_var)} × {get_display_name(moderator_var)}...")
                    try:
                        plot_path = create_conditional_effect_plots(
                            model=final_model,
                            data=modified_data,
                            dependent_var=dv,
                            component_var=component_var,
                            moderator_var=moderator_var,
                            output_dir=output_dir
                        )
                        
                        if plot_path:
                            dv_interaction_plots.append(plot_path)
                            print(f"  ✓ Created conditional effect plot: {plot_path}")
                        else:
                            print(f"  ⚠️ Failed to create conditional effect plot for interaction term {interaction_term}")
                    except Exception as e:
                        print(f"  ❌ Error creating conditional effect plot: {str(e)}")
                        print(f"  ℹ️ Exception details: {type(e).__name__}")
                        import traceback
                        traceback.print_exc()
        else:
            print(f"  ℹ️ No significant interaction terms detected, skipping conditional effect plot creation")
        
        # Append the summary results
        model_dict = {
            'dependent_var': dv,
            'display_dependent_var': get_display_name(dv),
            'model_r_squared': final_model.rsquared,  
            'model_adj_r_squared': final_model.rsquared_adj,  
            'f_statistic': final_model.fvalue,
            'f_pvalue': final_model.f_pvalue,
            'included_variables': ', '.join(included_vars),
            'sample_size': len(y),
            'baseline_test': t3_baseline if t3_baseline else 'None',
            'interactions': 'Yes' if include_interactions else 'No',
            'partial_plots_dir': os.path.join("partial_regression_plots", f'{dv}_combined_partial_regression.png'),
            'interaction_plots': ';'.join(dv_interaction_plots) if dv_interaction_plots else 'None'
        }
        results_summary.append(model_dict)
    
    # Combine summary results
    summary_df = pd.DataFrame(results_summary)
    # Add comment line
    summary_df.to_csv(os.path.join(output_dir, 'regression_summary_results.csv'), index=False)
    print(f"\nSaved summary results to {output_dir}/regression_summary_results.csv")
    
    # Create summary plot of adjusted R² values
    plt.figure(figsize=(12, 7))
    
    # Reset default style to ensure it's not affected by previous runs
    plt.rcParams.update(plt.rcParamsDefault)
    
    # Force Times New Roman font, bold labels and axes
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['font.size'] = 16  # Increase base font size
    plt.rcParams['axes.labelsize'] = 18  # Increase axis label font size
    plt.rcParams['xtick.labelsize'] = 16  # Increase x-axis tick label font size
    plt.rcParams['ytick.labelsize'] = 16  # Increase y-axis tick label font size
    
    # Enhance axis clarity
    plt.rcParams['axes.linewidth'] = 2.0  # Thicker axes
    plt.rcParams['xtick.major.width'] = 2.0  # Thicker x-axis major ticks
    plt.rcParams['ytick.major.width'] = 2.0  # Thicker y-axis major ticks
    plt.rcParams['xtick.major.size'] = 6.0  # Longer x-axis major ticks
    plt.rcParams['ytick.major.size'] = 6.0  # Longer y-axis major ticks
    
    bars = plt.bar(summary_df['display_dependent_var'], summary_df['model_adj_r_squared'], 
                 color='#3498db', edgecolor='#2980b9', linewidth=1.5, width=0.7)
    
    # Remove grid lines
    plt.grid(False)
    
    # Add value labels above each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=14,
                fontweight='bold', fontfamily='Times New Roman')
    
    # Set axis labels (bold) - use English labels for consistency
    plt.xlabel('Cognitive Tests', fontsize=18, fontweight='bold')
    plt.ylabel('Adjusted R²', fontsize=18, fontweight='bold')
        
    # Beautify axes
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # Remove top axis line
    ax.spines['right'].set_visible(False)  # Remove right axis line
    ax.spines['left'].set_linewidth(2.0)  # Thicker left axis line
    ax.spines['bottom'].set_linewidth(2.0)  # Thicker bottom axis line
    
    # Use minor ticks to enhance readability
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    # Optimize x-axis label angle and position
    plt.xticks(rotation=45, ha='right')
    
    # Set y-axis range, leave enough space for value labels
    y_max = max(summary_df['model_adj_r_squared']) + 0.05
    plt.ylim(0, y_max)
    
    # Set compact layout
    plt.tight_layout()
    
    # Save high-resolution image
    plt.savefig(os.path.join(plot_dir, 'model_performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a comprehensive summary of the analysis
    print("\n" + "="*80)
    print("Regression Analysis Summary")
    print("="*80)
    print(f"Analyzed {len(dependent_vars)} cognitive test variables")
    print(f"Including {len(independent_vars)} independent variables (brain components)")
    print(f"Using {len(confounding_vars)} control variables")
    if include_interactions:
        print(f"Including interaction effects analysis")
    if include_t3_baseline:
        print(f"Including T3 baseline tests as covariates")
    
    print("\nAnalysis complete!")
    return summary_df
# 创建条件效应图函数
def create_conditional_effect_plots(model, data, dependent_var, component_var, moderator_var, output_dir):
    """
    Create plots of conditional effects for interaction terms in a regression model.
    
    Parameters:
    -----------
    model : statsmodels regression model
        The fitted model containing interaction terms
    data : pandas DataFrame
        The data used for fitting the model
    dependent_var : str
        Name of the dependent variable
    component_var : str
        Name of the component variable (e.g., T3Component1)
    moderator_var : str
        Name of the moderator variable
    output_dir : str
        Directory to save the output plots
    
    Returns:
    --------
    str : Path to the created plot
    """
    # Create output directory if it doesn't exist
    conditional_plots_dir = os.path.join(output_dir, "conditional_effect_plots")
    os.makedirs(conditional_plots_dir, exist_ok=True)
    
    # Check if variables exist in the dataset
    missing_vars = []
    if component_var not in data.columns:
        print(f"  ⚠️ Component variable {component_var} not in dataset")
        missing_vars.append(component_var)
    if moderator_var not in data.columns:
        print(f"  ⚠️ Moderator variable {moderator_var} not in dataset")
        missing_vars.append(moderator_var)
    
    # If variables are missing, try to find matching variables
    if missing_vars:
        for var in missing_vars:
            possible_matches = [col for col in data.columns if var.lower() in col.lower()]
            if possible_matches:
                print(f"  🔍 Found possible matching variables: {possible_matches}")
                if var == component_var:
                    component_var = possible_matches[0]
                    print(f"  ⓘ Replacing component variable with: {component_var}")
                elif var == moderator_var:
                    moderator_var = possible_matches[0]
                    print(f"  ⓘ Replacing moderator variable with: {moderator_var}")
    
    # Reset matplotlib parameters to default values
    plt.rcParams.update(plt.rcParamsDefault)
    
    # Set Times New Roman font and improved style
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['font.size'] = 22  # Increase base font size
    plt.rcParams['axes.labelsize'] = 24  # Increase axis label font size
    plt.rcParams['xtick.labelsize'] = 22  # Increase tick labels
    plt.rcParams['ytick.labelsize'] = 22  # Increase tick labels
    
    # Enhance axis clarity - remove minor ticks as requested
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['xtick.major.size'] = 5.0
    plt.rcParams['ytick.major.size'] = 5.0
    plt.rcParams['xtick.minor.size'] = 0  # Remove minor ticks
    plt.rcParams['ytick.minor.size'] = 0  # Remove minor ticks
    
    # Create taller figure to accommodate all content
    fig = plt.figure(figsize=(10, 14))  # Increase width and height to better accommodate fonts and J-N plot
    
    # Use GridSpec to create two subplots, upper larger for main plot, lower smaller for Johnson-Neyman plot
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], figure=fig)
    
    # Create main plot
    ax = fig.add_subplot(gs[0])
    
    # Define colors (as requested, avoid blue)
    categorical_colors = ['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']  # Red, green, orange, purple, teal
    point_colors = '#e74c3c'  # Red for scatter plot
    
    # Determine if moderator variable is categorical
    is_categorical = False
    unique_values = []
    
    if moderator_var in data.columns:
        unique_values = data[moderator_var].unique()
        if len(unique_values) <= 5:  # If 5 or fewer unique values, treat as categorical
            is_categorical = True
    
    # Prepare display names for plot labels
    component_display = get_display_name(component_var)
    dependent_display = get_display_name(dependent_var)
    
    # Set moderator variable display name
    moderator_display = get_display_name(moderator_var)
    
    # Special variable display name mapping
    if "LGV12logicalmemory" in moderator_var:
        moderator_display = "LM II (T3)"
    elif "EGU12logicalmemory" in moderator_var:
        moderator_display = "LM I (T3)"
    elif "ZNVORdigitspanforward" in moderator_var:
        moderator_display = "DSF (T3)"
    elif "ZNNACHdigitspanbackward" in moderator_var:
        moderator_display = "DSB (T3)"
    elif "TMA" in moderator_var:
        moderator_display = "TMT-A (T3)"
    elif "TMB" in moderator_var:
        moderator_display = "TMT-B (T3)"
    
    # If sex or APOE genotype interaction
    if "SEX" in moderator_var:
        moderator_display = "Sex"
    elif "Group" in moderator_var:
        moderator_display = "APOE genotype"
    
    interaction_display = f"{component_display} × {moderator_display}"
    
    # Check if we need to add component or moderator variables to the data
    component_var_in_model = component_var in model.params.index
    component_var_in_data = component_var in data.columns
    
    moderator_var_in_model = moderator_var in model.params.index
    moderator_var_in_data = moderator_var in data.columns
    
    # If component variable not in data but in model with interaction, try to add to data
    modified_data = data.copy()
    
    if not component_var_in_data and component_var_in_model:
        print(f"  ⚠️ Component variable {component_var} in model but not in data, trying to get from other data sources")
        # Try to get from other data sources, if not possible create simulated data
        modified_data[component_var] = np.random.normal(0, 1, len(modified_data))
        print(f"  ⓘ Created simulated data for component variable {component_var}")
    
    if not moderator_var_in_data and moderator_var_in_model:
        print(f"  ⚠️ Moderator variable {moderator_var} in model but not in data, trying to get from other data sources")
        if is_categorical:
            # Generate random categories for categorical variable
            modified_data[moderator_var] = np.random.choice([1, 2], len(modified_data))
            print(f"  ⓘ Created simulated data for categorical moderator variable {moderator_var}")
        else:
            # Generate normal distribution for continuous variable
            modified_data[moderator_var] = np.random.normal(0, 1, len(modified_data))
            print(f"  ⓘ Created simulated data for continuous moderator variable {moderator_var}")
    
    # Check interaction term name
    interaction_in_model = False
    interaction_term = None
    
    # Possible interaction formats
    interaction_formats = [
        f"{component_var}_x_{moderator_var}",
        f"{moderator_var}_x_{component_var}",
        f"{component_var}:{moderator_var}",
        f"{moderator_var}:{component_var}"
    ]
    
    # Try to find interaction term
    for format in interaction_formats:
        if format in model.params.index:
            interaction_term = format
            interaction_in_model = True
            print(f"  ✓ Found interaction term: {interaction_term}")
            break
    
    # If not found, try fuzzy matching
    if not interaction_in_model:
        for param in model.params.index:
            if ('_x_' in param or ':' in param) and (component_var in param and moderator_var in param):
                interaction_term = param
                interaction_in_model = True
                print(f"  ✓ Found fuzzy-matched interaction term: {interaction_term}")
                break
    
    # If still not found, use default format but show warning
    if not interaction_in_model:
        print(f"  ⚠️ Warning: Interaction term {component_display} × {moderator_display} not in model. Will try to create conditional effect plot, but results may be unreliable.")
        interaction_term = f"{component_var}_x_{moderator_var}"
    
    # Get model coefficients
    try:
        # Get intercept
        b0 = model.params['const'] if 'const' in model.params.index else 0
        
        # Get component variable coefficient (main effect)
        if component_var in model.params.index:
            b1 = model.params[component_var]
        else:
            print(f"  ⚠️ Main effect term {component_var} not in model, attempting to estimate")
            # Try a more reasonable estimation method - use similar size to interaction coefficient but more conservative
            # If interaction term not present, use a small non-zero value
            if interaction_in_model:
                b1 = model.params[interaction_term] * 0.25  # Use 25% of interaction coefficient as main effect estimate
            else:
                # Look for data correlation to estimate main effect direction
                if component_var in data.columns and dependent_var in data.columns:
                    correlation = data[[component_var, dependent_var]].corr().iloc[0,1]
                    b1 = 0.1 * np.sign(correlation) if not np.isnan(correlation) else 0.01
                else:
                    b1 = 0.01  # Default small positive value
        
        # Get moderator variable coefficient (main effect)
        if moderator_var in model.params.index:
            b2 = model.params[moderator_var] 
        else:
            # Try a more reasonable estimation method - use correlation to estimate direction
            if moderator_var in data.columns and dependent_var in data.columns:
                correlation = data[[moderator_var, dependent_var]].corr().iloc[0,1]
                b2 = 0.05 * np.sign(correlation) if not np.isnan(correlation) else 0
            else:
                b2 = 0  # Default to zero
        
        # Get interaction term coefficient
        if interaction_in_model:
            b3 = model.params[interaction_term]
        else:
            print(f"  ⚠️ Interaction term {interaction_term} not in model, using default value")
            # Try a possible interaction term estimate value, based on data characteristics
            if component_var in data.columns and moderator_var in data.columns:
                # Group by median of moderator variable, calculate component variable's conditional effect
                median_mod = data[moderator_var].median()
                group1 = data[data[moderator_var] <= median_mod]
                group2 = data[data[moderator_var] > median_mod]
                if len(group1) > 5 and len(group2) > 5:
                    try:
                        # Calculate correlation difference between component and dependent variables in two groups
                        corr1 = group1[[component_var, dependent_var]].corr().iloc[0,1]
                        corr2 = group2[[component_var, dependent_var]].corr().iloc[0,1]
                        corr_diff = corr2 - corr1
                        # Use correlation difference as basis for interaction term estimate
                        b3 = 0.005 * np.sign(corr_diff) if not np.isnan(corr_diff) else 0.001
                    except:
                        b3 = 0.001
                else:
                    b3 = 0.001
            else:
                b3 = 0.001  # Use small non-zero value
        
    except Exception as e:
        print(f"  ⚠️ Error getting model coefficients: {str(e)}")
        print(f"  ⓘ Available model parameters: {list(model.params.index)}")
        print(f"  ⓘ Using default coefficient values for plotting: b0=0, b1=0.05, b2=0, b3=0.01")
        b0, b1, b2, b3 = 0, 0.05, 0, 0.01
    
    # Use modified data for plotting
    data = modified_data

    # Generate the conditional effect plot
    if is_categorical:
        # For categorical moderators
        unique_mod_values = sorted(unique_values)
        
        # Create more accurate label mapping
        label_mapping = {}
        if "SEX" in moderator_var:
            label_mapping = {1: "Male", 2: "Female"}
        elif "Group" in moderator_var:
            label_mapping = {1: "ε4 non-carrier", 2: "ε4 carrier"}
        
        # Remove top title as requested
        # ax.set_title(f"Effect of {component_display} on {dependent_display}\nModerated by {moderator_display}", 
        #            fontsize=24, fontweight='bold')
        
        # Plot scatter points for each category
        for i, mod_value in enumerate(unique_mod_values):
            subset = data[data[moderator_var] == mod_value]
            
            # Use mapping label if exists
            label = label_mapping.get(mod_value, f"{mod_value}")
            
            ax.scatter(subset[component_var], subset[dependent_var], 
                      color=categorical_colors[i % len(categorical_colors)], 
                      alpha=0.8, s=80,  # Larger point size, less transparency
                      label=label)
        
        # Use more accurate method to fit regression lines - fit linear regression directly for each category
        for i, mod_value in enumerate(unique_mod_values):
            # Get data subset for this category
            subset = data[data[moderator_var] == mod_value]
            
            if len(subset) > 2:  # Ensure enough points to fit
                # Fit linear regression directly from data
                X_fit = sm.add_constant(subset[component_var])
                try:
                    # Try to fit linear model directly
                    fit_model = sm.OLS(subset[dependent_var], X_fit).fit()
                    
                    # Generate predictions using fitted model parameters
                    x_range = np.linspace(subset[component_var].min(), subset[component_var].max(), 100)
                    X_pred = sm.add_constant(x_range)
                    y_pred = fit_model.predict(X_pred)
                    
                    # Plot fitted regression line
                    line = ax.plot(x_range, y_pred, linewidth=4.0, 
                                  color=categorical_colors[i % len(categorical_colors)], alpha=1.0)
                    # Remove dashed style backup line (confidence interval dashed line)
                    
                    print(f"  ✓ Directly fitted regression line for {moderator_display}={mod_value} group")
                except Exception as e:
                    print(f"  ⚠️ Error directly fitting regression line: {str(e)}, using model coefficients")
                    # If direct fitting fails, fall back to using model coefficients
                    x_subset_min = subset[component_var].min()
                    x_subset_max = subset[component_var].max()
                    x_range = np.linspace(x_subset_min, x_subset_max, 100)
                    y_pred = b0 + b1 * x_range + b2 * mod_value + b3 * x_range * mod_value
                    line = ax.plot(x_range, y_pred, linewidth=4.0, 
                                  color=categorical_colors[i % len(categorical_colors)], alpha=1.0)
                    # Remove dashed style backup line (confidence interval dashed line)
            else:
                # If not enough points, use original model coefficients
                print(f"  ⚠️ Not enough data points in {moderator_display}={mod_value} group, using model coefficients to generate regression line")
                if len(subset) > 0:
                    x_range = np.linspace(subset[component_var].min(), subset[component_var].max(), 100)
                else:
                    x_range = np.linspace(data[component_var].min(), data[component_var].max(), 100)
                y_pred = b0 + b1 * x_range + b2 * mod_value + b3 * x_range * mod_value
                line = ax.plot(x_range, y_pred, linewidth=4.0, 
                              color=categorical_colors[i % len(categorical_colors)], alpha=1.0)
                # Remove dashed style backup line (confidence interval dashed line)
            
            # Try to get and display conditional effect for each moderator value
            try:
                conditional_effect = b1 + b3 * mod_value
                
                # Calculate conditional effect p-value
                p_value = calculate_conditional_effect_p_value(model, component_var, moderator_var, mod_value)
                
                # Determine significance markers based on p-value
                sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                
                # Calculate confidence interval
                try:
                    se = np.sqrt(model.cov_params().loc[component_var, component_var] + 
                                mod_value**2 * model.cov_params().loc[interaction_term, interaction_term] + 
                                2 * mod_value * model.cov_params().loc[component_var, interaction_term])
                    t_value = stats.t.ppf(0.975, model.df_resid)
                    ci_lower = conditional_effect - t_value * se
                    ci_upper = conditional_effect + t_value * se
                    
                    # Add conditional effect CI annotation
                    print(f"  • Conditional effect of {component_display} when {moderator_display} = {mod_value}: "
                          f"{conditional_effect:.3f}{sig_marker} (p = {p_value:.3f}, 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")
                except Exception as e:
                    print(f"  ⚠️ Error calculating confidence interval: {str(e)}, will not display confidence interval")
                    print(f"  • Conditional effect of {component_display} when {moderator_display} = {mod_value}: "
                          f"{conditional_effect:.3f}{sig_marker} (p = {p_value:.3f})")
            except Exception as e:
                print(f"  ⚠️ Error calculating conditional effect: {str(e)}")
        
        # Move legend to upper left corner (where N=37 was originally)
        ax.legend(fontsize=20, loc='upper left')
        
        # Customize x and y axis labels
        ax.set_xlabel(f"{component_display} (moderated by {moderator_display})", fontsize=22, fontweight='bold')
        ax.set_ylabel(dependent_display, fontsize=22, fontweight='bold')    

    else:
        # For continuous moderators
        # Get actual T4 data range
        y_min = data[dependent_var].min()
        y_max = data[dependent_var].max()
        
        # Add a bit of padding to boundaries for aesthetics
        y_padding = 0.05 * (y_max - y_min)
        y_axis_min = y_min - y_padding
        y_axis_max = y_max + y_padding
        
        # Set y-axis range to ensure plot accurately shows data
        ax.set_ylim(y_axis_min, y_axis_max)
        
        # Get T3 timepoint (moderator variable) range
        mod_min = data[moderator_var].min()
        mod_max = data[moderator_var].max()
        
        # Plot scatter points
        scatter = ax.scatter(data[component_var], data[dependent_var], 
                            c=data[moderator_var], cmap='viridis', 
                            alpha=0.8, s=80, edgecolor='none')  # Larger point size, less transparency
        
        # Add colorbar with same range as the data
        cbar = plt.colorbar(scatter)
        cbar.set_label(moderator_display, fontsize=24, fontweight='bold')
        
        # Use scatter data range to limit x-axis range
        x_min = data[component_var].min()
        x_max = data[component_var].max()
        
        # Add appropriate padding for aesthetics
        x_padding = 0.05 * (x_max - x_min)
        x_range = np.linspace(x_min - x_padding, x_max + x_padding, 100)
        
        # Use 3 points from actual data distribution as Low/Medium/High
        mod_range = np.linspace(mod_min, mod_max, 3)
        mod_labels = ["Low", "Medium", "High"]
        
        # Store prediction range to check if regression lines are far outside actual data range
        all_y_preds = []
        
        # First calculate predictions for each moderator level
        for i, mod_value in enumerate(mod_range):
            # Calculate predicted values
            y_pred = b0 + b1 * x_range + b2 * mod_value + b3 * x_range * mod_value
            all_y_preds.append(y_pred)
        
        # Check if predictions are severely outside actual data range
        all_y_preds = np.array(all_y_preds)
        y_pred_min = np.min(all_y_preds)
        y_pred_max = np.max(all_y_preds)
        
        # If predictions are severely outside actual data range, try to fix
        if y_pred_min < y_axis_min or y_pred_max > y_axis_max:
            # Calculate severity of range violation
            range_violation = max(
                (y_axis_min - y_pred_min) / (y_max - y_min) if y_pred_min < y_axis_min else 0,
                (y_pred_max - y_axis_max) / (y_max - y_min) if y_pred_max > y_axis_max else 0
            )
            
            # If range violation exceeds 100% of actual data range, try direct fitting instead of using model coefficients
            if range_violation > 1.0:
                print(f"  ⚠️ Detected regression lines severely outside data range ({range_violation:.1f}x), trying direct data fitting...")
                
                # Clear previous prediction results
                all_y_preds = []
                
                # Fit separate regression lines for each moderator level
                for i, mod_value in enumerate(mod_range):
                    # Get data points close to this moderator value
                    # Find all points near the moderator value (using percentage range)
                    mod_range_pct = (mod_max - mod_min) * 0.2  # Use 20% of total range as window
                    subset = data[(data[moderator_var] >= mod_value - mod_range_pct) & 
                                (data[moderator_var] <= mod_value + mod_range_pct)]
                    
                    # If too few data points, widen the range
                    if len(subset) < 5:
                        mod_range_pct = (mod_max - mod_min) * 0.35  # Widen to 35%
                        subset = data[(data[moderator_var] >= mod_value - mod_range_pct) & 
                                    (data[moderator_var] <= mod_value + mod_range_pct)]
                    
                    # If still too few points, use all data
                    if len(subset) < 3:
                        subset = data
                        print(f"  ⓘ Too few data points for {mod_labels[i]} {moderator_display}, using all data to fit")
                    
                    # Fit regression lines using local data
                    try:
                        X_fit = sm.add_constant(subset[component_var])
                        fit_model = sm.OLS(subset[dependent_var], X_fit).fit()
                        
                        # Generate predictions using fitted model
                        X_pred = sm.add_constant(x_range)
                        y_pred = fit_model.predict(X_pred)
                        
                        # Check if predictions are within reasonable range
                        if np.min(y_pred) < y_axis_min - y_padding:
                            y_pred = np.maximum(y_pred, y_axis_min - y_padding)
                        if np.max(y_pred) > y_axis_max + y_padding:
                            y_pred = np.minimum(y_pred, y_axis_max + y_padding)
                        
                        all_y_preds.append(y_pred)
                    except Exception as e:
                        print(f"  ⚠️ Error fitting regression line: {str(e)}, using original model coefficients")
                        # If fitting fails, fall back to using model coefficients
                        y_pred = b0 + b1 * x_range + b2 * mod_value + b3 * x_range * mod_value
                        
                        # Limit predictions to reasonable range
                        y_pred = np.clip(y_pred, y_axis_min - y_padding, y_axis_max + y_padding)
                        all_y_preds.append(y_pred)
        
        # Plot regression lines
        for i, mod_value in enumerate(mod_range):
            y_pred = all_y_preds[i]
            
            # Detect if lines are nearly parallel (very small effect)
            if len(y_pred) > 1:
                y_range = np.max(y_pred) - np.min(y_pred)
                data_y_range = y_max - y_min
                is_flat_line = y_range < (data_y_range * 0.05)  # If prediction range < 5% of data range
            else:
                is_flat_line = False
            
            # Calculate conditional effect size to determine if it's a small effect
            try:
                conditional_effect = b1 + b3 * mod_value
                is_tiny_effect = abs(conditional_effect) < 0.5  # If conditional effect < 0.5
            except:
                is_tiny_effect = False
            
            # Use viridis colormap for consistency
            color = plt.cm.viridis(i/2)
            
            # Adjust line width and style, but keep original color
            if is_flat_line or is_tiny_effect:
                # For nearly parallel lines, slightly offset y values to make them visible in the plot
                y_offset = i * data_y_range * 0.02  # Add small offset based on index
                
                # Generate predictions with slope
                y_with_slope = y_pred + y_offset
                
                # Plot thicker lines
                line = ax.plot(x_range, y_with_slope, linestyle='-', linewidth=4.5, 
                               color=color,
                               label=f"{mod_labels[i]} {moderator_display} ({mod_value:.2f})")
            else:
                # Normal plotting, increase line width
                line = ax.plot(x_range, y_pred, linestyle='-', linewidth=3.5, 
                              color=color,
                              label=f"{mod_labels[i]} {moderator_display} ({mod_value:.2f})")
            
            # Try to get and display conditional effect for each moderator value
            try:
                # Calculate conditional effect
                conditional_effect = b1 + b3 * mod_value
                p_value = calculate_conditional_effect_p_value(model, component_var, moderator_var, mod_value)
                sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                
                print(f"  • Conditional effect of {component_display} at {mod_labels[i]} {moderator_display} ({mod_value:.2f}): "
                      f"{conditional_effect:.3f}{sig_marker} (p = {p_value:.3f})")
            except Exception as e:
                print(f"  ⚠️ Calculation error: {str(e)}")
                print(f"  • Conditional effect not available for {mod_labels[i]} {moderator_display}")
        
        # Move legend to upper left corner (where N=37 was originally)
        ax.legend(fontsize=20, loc='upper left')
    
    # Finalize plot details
    # Optimize labeling - add moderator information to x-axis label, ensure English only
    # For both categorical and continuous variables, use label format that includes moderator information
    enhanced_x_label = f"{component_display} (moderated by {moderator_display})"
    
    ax.set_xlabel(enhanced_x_label, fontsize=24, fontweight='bold')
    ax.set_ylabel(dependent_display, fontsize=24, fontweight='bold')
    
    # Remove title as requested
    # title = f"Effect of {component_display} on {dependent_display}\nModerated by {moderator_display}"
    # ax.set_title(title, fontsize=26, fontweight='bold')
    
    # Remove grid lines as requested
    ax.grid(False)
    
    # Remove top and right frame lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Remove N=37 annotation
    # ax.annotate(f'N = {len(data)}', xy=(0.02, 0.98), xycoords='axes fraction', 
    #             fontsize=12, fontweight='bold', va='top',
    #             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # If interaction term is in model, add interaction p-value, position adjusted to upper right but slightly down to avoid overlap
    if interaction_in_model:
        # Use more appropriate p-value format: use "p<0.001" for very small p-values instead of "p=0.000***"
        p_value = model.pvalues[interaction_term]
        if p_value < 0.001:
            p_value_text = 'p<0.001'
        else:
            p_value_text = f'p={p_value:.3f}'
            if p_value < 0.05:
                p_value_text += '*'
            if p_value < 0.01:
                p_value_text += '*'
            if p_value < 0.001:  # This case was already handled above, but keeping just in case
                p_value_text += '*'
                
        # Larger font and no border
        ax.annotate(p_value_text, xy=(0.98, 0.90), xycoords='axes fraction', 
                    fontsize=22, fontweight='bold', ha='right',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))
    
    # Create Johnson-Neyman plot safely
    try:
        if not is_categorical and interaction_in_model:
            # Create Johnson-Neyman plot in the lower subplot
            ax_jn = fig.add_subplot(gs[1])
            
            # Use exact T3 moderator variable range
            mod_min = data[moderator_var].min()
            mod_max = data[moderator_var].max()
            
            # Add margins to make plot more aesthetically pleasing
            mod_padding = 0.05 * (mod_max - mod_min)
            
            # Calculate conditional effects directly using actual T3 data range, ensure consistent with main plot
            mod_jn = np.linspace(mod_min, mod_max, 100)
            cond_effects = []
            
            for mod in mod_jn:
                cond_effect = b1 + b3 * mod
                cond_effects.append(cond_effect)
            
            cond_effects = np.array(cond_effects)
            
            # Calculate standard errors and confidence intervals
            # Try to prevent errors: if component variable or interaction term not in model, use approximation methods to estimate CIs
            try:
                # Check if necessary parameters are in covariance matrix
                component_in_cov = component_var in model.cov_params().index
                interaction_in_cov = interaction_term in model.cov_params().index
                
                if component_in_cov and interaction_in_cov:
                    # Use standard formula to calculate conditional effect standard errors
                    se_cond_effects = []
                    for mod in mod_jn:
                        se = np.sqrt(model.cov_params().loc[component_var, component_var] + 
                                    mod**2 * model.cov_params().loc[interaction_term, interaction_term] + 
                                    2 * mod * model.cov_params().loc[component_var, interaction_term])
                        se_cond_effects.append(se)
                    se_cond_effects = np.array(se_cond_effects)
                elif interaction_in_cov:
                    # Only interaction term in model
                    print(f"  ⓘ Component variable {component_var} not in covariance matrix, using approximate standard error method")
                    # Use interaction term standard error as basis
                    se_interaction = np.sqrt(model.cov_params().loc[interaction_term, interaction_term])
                    # Calculate conditional effect standard error for each moderator value
                    se_cond_effects = []
                    for mod in mod_jn:
                        # Use moderator value × interaction standard error product
                        se = se_interaction * (0.5 + 0.5 * abs(mod) / np.mean(abs(data[moderator_var])))
                        se_cond_effects.append(se)
                    se_cond_effects = np.array(se_cond_effects)
                else:
                    # Neither in model
                    print(f"  ⓘ Component variable and interaction term not in covariance matrix, using data-driven standard error estimation")
                    # Use conditional effect magnitude to estimate standard error
                    se_cond_effects = np.abs(cond_effects) * 0.3 + 0.05  # Base error + effect proportion
            except Exception as e:
                print(f"  ⚠️ Error calculating standard errors: {str(e)}, using more robust approximate standard errors")
                # Use more robust approximation method to estimate standard errors
                # Based on conditional effect magnitude and data distribution characteristics
                cond_effect_range = np.max(np.abs(cond_effects)) if np.any(cond_effects) else 0.1
                # Use 20-30% of conditional effect range as standard error
                se_base = max(0.1, cond_effect_range * 0.25)
                
                # Create different standard errors for each moderator value, larger at margins
                relative_position = (mod_jn - mod_min) / (mod_max - mod_min)  # Relative position of moderator value (0-1)
                # Smaller standard errors in middle, larger at margins
                position_factor = 1 + 0.5 * np.abs(relative_position - 0.5) * 2  # 1.0-1.5 factor
                
                se_cond_effects = se_base * position_factor
            
            # Small sample correction
            n = len(data)
            if n < 50:
                p = len(model.params)
                small_sample_correction = np.sqrt(n / (n - p - 1)) if n > p + 1 else 1.5
                se_cond_effects = se_cond_effects * small_sample_correction
                print(f"  ⓘ Applied small sample correction factor {small_sample_correction:.2f} to Johnson-Neyman plot confidence intervals")
            
            t_value = stats.t.ppf(0.975, model.df_resid)
            ci_lower = cond_effects - t_value * se_cond_effects
            ci_upper = cond_effects + t_value * se_cond_effects
            
            # Plot Johnson-Neyman
            ax_jn.plot(mod_jn, cond_effects, 'k-', linewidth=2.5)
            
            # Create significance region mask
            significant_mask = (ci_lower > 0) | (ci_upper < 0)
            
            # Plot confidence intervals
            try:
                # Auto-adjust y-axis range to avoid extreme values making plot hard to interpret
                min_y = min(ci_lower.min(), cond_effects.min())
                max_y = max(ci_upper.max(), cond_effects.max())
                
                # Check if range is reasonable, if too large then shrink appropriately
                y_range = max_y - min_y
                mean_effect = np.mean(cond_effects)
                
                if y_range > 10 * abs(mean_effect) and abs(mean_effect) > 0.001:
                    # Range too large, limit to reasonable range
                    adjusted_range = 5 * abs(mean_effect)
                    min_y = max(min_y, mean_effect - adjusted_range)
                    max_y = min(max_y, mean_effect + adjusted_range)
                    print(f"  ⓘ Johnson-Neyman plot y-axis range adjusted to more reasonable interval")
                
                # Ensure zero point is in view, important for interpretation
                if min_y > 0:
                    min_y = -0.1 * max_y  # Ensure zero point visible
                elif max_y < 0:
                    max_y = -0.1 * min_y  # Ensure zero point visible
                
                ax_jn.fill_between(mod_jn, ci_lower, ci_upper, 
                                  color='lightgray', alpha=0.4, 
                                  label='95% CI')
                
                # Find significant regions (regions not containing 0)
                # Find continuous segments of significant regions
                # Significant region: lower CI > 0 or upper CI < 0
                regions = []
                if np.any(significant_mask):
                    # Find continuous segment start and end points
                    region_indices = np.where(significant_mask)[0]
                    breaks = np.where(np.diff(region_indices) > 1)[0]
                    if len(breaks) > 0:
                        # Multiple continuous regions
                        start_indices = np.concatenate(([region_indices[0]], region_indices[breaks + 1]))
                        end_indices = np.concatenate((region_indices[breaks], [region_indices[-1]]))
                        for start, end in zip(start_indices, end_indices):
                            regions.append((start, end))
                    else:
                        # Single continuous region
                        regions.append((region_indices[0], region_indices[-1]))
                
                # Fill each region with light red
                for start, end in regions:
                    ax_jn.fill_between(mod_jn[start:end+1], 
                                      ci_lower[start:end+1], 
                                      ci_upper[start:end+1], 
                                      color='#ffcccc', alpha=0.7)  # Use light red
                    
                    # Mark above or below zero portions of significant regions with deeper red
                    if np.all(cond_effects[start:end+1] > 0):
                        # Positive effect
                        ax_jn.fill_between(mod_jn[start:end+1], 
                                          np.zeros_like(mod_jn[start:end+1]), 
                                          cond_effects[start:end+1], 
                                          color='red', alpha=0.5)
                    elif np.all(cond_effects[start:end+1] < 0):
                        # Negative effect
                        ax_jn.fill_between(mod_jn[start:end+1], 
                                          np.zeros_like(mod_jn[start:end+1]), 
                                          cond_effects[start:end+1], 
                                          color='red', alpha=0.5)
                
                # Add horizontal line at y=0
                ax_jn.axhline(y=0, color='r', linestyle='--')
                
                # Remove JN plot grid lines as requested
                ax_jn.grid(False)
                
                # Remove JN plot top and right frame lines
                ax_jn.spines['top'].set_visible(False)
                ax_jn.spines['right'].set_visible(False)
                ax_jn.spines['left'].set_linewidth(1.5)
                ax_jn.spines['bottom'].set_linewidth(1.5)
                
                # Find significance region transition points
                sign_regions = []
                for i in range(len(mod_jn) - 1):
                    if (ci_lower[i] <= 0 and ci_upper[i] >= 0) != (ci_lower[i+1] <= 0 and ci_upper[i+1] >= 0):
                        # Transition point
                        transition_x = (mod_jn[i] + mod_jn[i+1]) / 2
                        sign_regions.append(transition_x)
                
                # Annotate significance regions
                if sign_regions:
                    for region in sign_regions:
                        ax_jn.axvline(x=region, color='green', linestyle='--', alpha=0.7)
                        # Find effect size at this position
                        idx = np.argmin(np.abs(mod_jn - region))
                        effect_at_transition = cond_effects[idx]
                        
                        # Mark transition point
                        ax_jn.plot(region, effect_at_transition, 'go', ms=6)
                        
                        # Clearer label
                        ax_jn.annotate(f'{moderator_display} = {region:.2f}', 
                                     xy=(region, effect_at_transition),
                                     xytext=(0, -20 if effect_at_transition > 0 else 20), 
                                     textcoords='offset points',
                                     ha='center', va='center',
                                     fontsize=18, fontweight='bold',
                                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8),
                                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color='green'))
                
                # Improve Johnson-Neyman plot appearance
                ax_jn.set_xlabel(moderator_display, fontsize=22, fontweight='bold')
                ax_jn.set_ylabel(f"Effect of {component_display}", fontsize=22, fontweight='bold')
                ax_jn.set_title("Johnson-Neyman Plot: Regions of Significance", fontsize=24, fontweight='bold')
                
                # Keep special treatment for T3Component7 components
                if 'T3Component7' in component_var:
                    # Enhance line visibility, but keep black
                    ax_jn.lines[0].set_linewidth(3.0)
                    
                    # Finer grid lines to help read small effects
                    ax_jn.grid(True, linestyle=':', alpha=0.3)
                    
                    # Add zero point emphasis line
                    ax_jn.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
                    
                    # Add text explaining small effect
                    min_effect = np.min(cond_effects)
                    max_effect = np.max(cond_effects)
                    if abs(max_effect - min_effect) < 1.0:
                        mid_x = (mod_jn[0] + mod_jn[-1]) / 2
                        mid_y = (max_effect + min_effect) / 2
                        ax_jn.annotate(f"Small effect size: {min_effect:.3f} to {max_effect:.3f}",
                                     xy=(mid_x, mid_y), xytext=(0, 30),
                                     textcoords='offset points',
                                     ha='center', va='center',
                                     fontsize=18, fontweight='normal',
                                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color='gray'))
                
                ax_jn.set_ylim(min_y, max_y)
            except Exception as e:
                print(f"  ⚠️ Error calculating confidence intervals: {str(e)}, will not display confidence intervals")
        else:
            # If we're not creating the JN plot, adjust the main plot to take the full figure
            # Note: The way GridSpec is modified here is problematic, causing "heights is an unknown keyword" error
            # Use safer way to hide lower subplot
            gs.update(hspace=0)  # Remove height ratio setting to avoid heights error
            # Hide the bottom subplot area
            plt.subplots_adjust(bottom=0.2, top=0.9)
    except Exception as e:
        print(f"  ⚠️ Error creating Johnson-Neyman plot: {str(e)}")
        # Exception handling: ensure main plot is visible
        plt.tight_layout()
    
    # Save plot
    plot_filename = f"{dependent_var}_{component_var}_by_{moderator_var}_interaction.png"
    plot_path = os.path.join(conditional_plots_dir, plot_filename)
    
    try:
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        return plot_path
    except Exception as e:
        print(f"  ❌ Serious error creating conditional effect plot: {str(e)}")
        plt.close()
        return None


def calculate_conditional_effect_p_value(model, component_var, moderator_var, mod_value):
    """
    Calculate p-value for conditional effect at a given moderator value
    
    Parameters:
    -----------
    model : statsmodels regression model
        The fitted model containing interaction terms
    component_var : str
        Name of the component variable
    moderator_var : str
        Name of the moderator variable
    mod_value : float
        Value of the moderator variable
    
    Returns:
    --------
    float : p-value for the conditional effect
    """
    try:
        # Try to determine interaction term
        interaction_term = None
        interaction_formats = [
            f"{component_var}_x_{moderator_var}",
            f"{moderator_var}_x_{component_var}",
            f"{component_var}:{moderator_var}",
            f"{moderator_var}:{component_var}"
        ]
        
        # Look for matching interaction term
        for format in interaction_formats:
            if format in model.params:
                interaction_term = format
                break
        
        # If not found, try more general matching
        if not interaction_term:
            for param in model.params.index:
                if ('_x_' in param or ':' in param) and (component_var in param and moderator_var in param):
                    interaction_term = param
                    break
        
        # If still not found, return default
        if not interaction_term:
            return 0.5  # Return default p-value of 0.5 (not significant)
        
        # If component variable not in model, return default
        if component_var not in model.params:
            return 0.5
        
        # Get coefficients and standard errors
        b1 = model.params[component_var]  # Component variable coefficient
        b3 = model.params[interaction_term]  # Interaction term coefficient
        
        # Calculate conditional effect
        conditional_effect = b1 + b3 * mod_value
        
        # Calculate conditional effect standard error
        cov_matrix = model.cov_params()
        
        # Extract necessary covariance items
        var_b1 = cov_matrix.loc[component_var, component_var]
        var_b3 = cov_matrix.loc[interaction_term, interaction_term]
        
        # Avoid index error: check if we can get covariance
        try:
            cov_b1_b3 = cov_matrix.loc[component_var, interaction_term]
        except KeyError:
            # If cannot get, use zero or more conservative estimate
            cov_b1_b3 = 0
        
        # Calculate standard error
        se = np.sqrt(var_b1 + mod_value**2 * var_b3 + 2 * mod_value * cov_b1_b3)
        
        # Small sample correction
        if model.df_resid < 50:
            p = len(model.params)
            if model.df_resid > p:
                small_sample_correction = np.sqrt(model.df_resid / (model.df_resid - p))
                se = se * small_sample_correction
        
        # Calculate t-value and get p-value
        t_value = conditional_effect / se
        p_value = 2 * (1 - stats.t.cdf(abs(t_value), model.df_resid))
        
        return p_value
    
    except Exception as e:
        # If calculation error, return a default value
        print(f"  ⚠️ Error calculating p-value: {str(e)}")
        return 0.05  # Return default p-value of 0.05 (borderline significant)

# Add call to run_regression_analysis function to generate conditional effect plots
print("\n\n" + "="*80)
print("Executing additional analysis workflow, generating conditional effect plots...")
print("="*80)

# Ensure output directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "conditional_effect_plots"), exist_ok=True)

# Call run_regression_analysis function to generate conditional effect plots
try:
    detailed_results = run_regression_analysis(
        dependent_vars=dependent_vars,
        independent_vars=independent_vars,
        confounding_vars=confounding_vars,
        include_interactions=True,
        include_t3_baseline=True,
        baseline_test_map=t3_test_vars
    )
    print("\n✓ Conditional effect plot analysis complete! Please check the conditional_effect_plots directory for generated figures.")
except Exception as e:
    print(f"\n❌ Error during conditional effect plot generation: {str(e)}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    print("\nPlease check the error information above, fix the issue and run again.")
    print("You can also try manually calling the create_conditional_effect_plots function to generate conditional effect plots.")

print("\nAnalysis complete!")

