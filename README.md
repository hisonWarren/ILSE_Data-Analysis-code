# Code Availability Statement

This repository contains Python scripts used to perform the statistical analyses described in our paper. The scripts implement the following key methodological approaches:

## Regression Analysis with Bootstrap

`cognitive_regression_bootstrap_analysis.py` implements our stepwise regression analysis with bootstrap confidence intervals. The script includes:

- Normality testing (Shapiro-Wilk, Kolmogorov-Smirnov)
- Outlier detection and multicollinearity assessment
- Non-normal data transformation using Yeo-Johnson method
- Stepwise regression with forward selection (p < 0.05) and backward elimination (p > 0.1)
- Bootstrap resampling (1000 iterations) for robust confidence intervals
- Interaction analysis between APOE genotype and brain components
- Variable importance estimation through drop-one analysis

## Longitudinal Mixed Effects Models 

`cognitive_longitudinal_mixed_effects.py` implements our longitudinal analysis of cognitive trajectories:

- Linear mixed-effects models with random slopes for time
- Time × APOE genotype interaction effects
- Model comparison using AIC/BIC and likelihood ratio tests
- Forest plots of estimated time effects across cognitive tests
- Fixed effects for APOE genotype, age, education and sex

## Partial Correlation Analysis

`correlation.py` implements our partial correlation analysis to examine relationships between cognitive measures and brain components:

- Partial Spearman correlation analysis to control for potential confounders
- Covariate adjustment for sex, age, education, APOE ε4 status, and total intracranial volume
- Multiple comparison correction using Benjamini-Hochberg False Discovery Rate (FDR) method
- Comprehensive outlier detection using Z-score method (threshold = 3)
- Generation of detailed correlation matrices with significance markers
- Visualization through correlation heatmaps and significance matrices
- Complete output of correlation coefficients, p-values, and adjusted p-values

Both implementations use standard Python statistical libraries (statsmodels, scipy, sklearn) and produce comprehensive outputs including diagnostic plots, variable importance metrics, and detailed statistical tables saved to CSV format.

Note that these scripts were designed for our specific dataset and analysis workflow, so they may require adaptation for use with different data structures.

## Dependencies

- Python 3.10.11
- pandas, numpy, scipy, statsmodels, sklearn
- matplotlib, seaborn
- joblib
- pingouin (for partial correlation analysis) 