import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import numpy as np
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_visualizations(train_file, test_file, output_dir):
    """
    Create comprehensive visualizations for credit risk modeling data.
    
    Args:
        train_file: Path to training data CSV
        test_file: Path to test data CSV  
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    # Combine datasets for some visualizations
    train_df['dataset'] = 'train'
    test_df['dataset'] = 'test'
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Target Variable Distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    train_df['Risk'].value_counts().plot(kind='bar', color=['#2E8B57', '#DC143C'])
    plt.title('Risk Distribution in Training Data')
    plt.xlabel('Risk Level')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    plt.subplot(1, 2, 2)
    test_df['Risk'].value_counts().plot(kind='bar', color=['#2E8B57', '#DC143C'])
    plt.title('Risk Distribution in Test Data')
    plt.xlabel('Risk Level')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'risk_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Risk distribution plots saved")
    
    # 2. Numeric Features Distribution
    numeric_cols = ['Age', 'Job', 'Credit amount', 'Duration']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, col in enumerate(numeric_cols):
        train_df[col].hist(bins=30, alpha=0.7, ax=axes[i], label='Train')
        test_df[col].hist(bins=30, alpha=0.7, ax=axes[i], label='Test')
        axes[i].set_title(f'{col} Distribution')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'numeric_features_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Numeric features distribution plots saved")
    
    # 3. Correlation Matrix (excluding non-numeric Risk column)
    plt.figure(figsize=(12, 10))
    
    # Select only numeric columns for correlation (exclude Risk and dataset)
    numeric_data = train_df.select_dtypes(include=['float64', 'int64'])
    
    # Convert boolean columns to numeric for correlation
    bool_cols = train_df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        numeric_data[col] = train_df[col].astype(int)
    
    correlation_matrix = numeric_data.corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": .5})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Correlation matrix saved")
    
    # 4. Feature Importance (based on correlation with target - encode Risk as numeric)
    plt.figure(figsize=(10, 8))
    
    # Create a copy with Risk encoded as numeric for correlation
    train_encoded = train_df.copy()
    train_encoded['Risk_numeric'] = (train_encoded['Risk'] == 'bad').astype(int)
    
    # Calculate correlation with target
    numeric_data_encoded = train_encoded.select_dtypes(include=['float64', 'int64'])
    bool_cols_encoded = train_encoded.select_dtypes(include=['bool']).columns
    for col in bool_cols_encoded:
        numeric_data_encoded[col] = train_encoded[col].astype(int)
    
    # Add the Risk_numeric column to the numeric data
    numeric_data_encoded['Risk_numeric'] = train_encoded['Risk_numeric']
    
    correlation_matrix_encoded = numeric_data_encoded.corr()
    correlations = correlation_matrix_encoded['Risk_numeric'].drop('Risk_numeric').abs().sort_values(ascending=True)
    
    correlations.plot(kind='barh', color='skyblue')
    plt.title('Feature Importance (Absolute Correlation with Risk)')
    plt.xlabel('Absolute Correlation')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Feature importance plot saved")
    
    # 5. Age vs Credit Amount by Risk
    plt.figure(figsize=(10, 6))
    
    for risk in train_df['Risk'].unique():
        subset = train_df[train_df['Risk'] == risk]
        plt.scatter(subset['Age'], subset['Credit amount'], 
                   alpha=0.6, label=f'Risk: {risk}', s=30)
    
    plt.xlabel('Age')
    plt.ylabel('Credit Amount')
    plt.title('Age vs Credit Amount by Risk Level')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'age_vs_credit_by_risk.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Age vs Credit Amount plot saved")
    
    # 6. Duration Distribution by Risk
    plt.figure(figsize=(10, 6))
    
    for risk in train_df['Risk'].unique():
        subset = train_df[train_df['Risk'] == risk]
        plt.hist(subset['Duration'], bins=20, alpha=0.7, 
                label=f'Risk: {risk}', density=True)
    
    plt.xlabel('Duration (months)')
    plt.ylabel('Density')
    plt.title('Duration Distribution by Risk Level')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'duration_by_risk.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Duration by Risk plot saved")
    
    # 7. Dataset Comparison
    plt.figure(figsize=(12, 8))
    
    # Compare key statistics between train and test
    train_stats = train_df[numeric_cols].describe()
    test_stats = test_df[numeric_cols].describe()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, col in enumerate(numeric_cols):
        train_mean = train_stats.loc['mean', col]
        test_mean = test_stats.loc['mean', col]
        train_std = train_stats.loc['std', col]
        test_std = test_stats.loc['std', col]
        
        categories = ['Mean', 'Std Dev']
        train_values = [train_mean, train_std]
        test_values = [test_mean, test_std]
        
        x = np.arange(len(categories))
        width = 0.35
        
        axes[i].bar(x - width/2, train_values, width, label='Train', alpha=0.8)
        axes[i].bar(x + width/2, test_values, width, label='Test', alpha=0.8)
        
        axes[i].set_title(f'{col} Statistics Comparison')
        axes[i].set_xlabel('Statistic')
        axes[i].set_ylabel('Value')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(categories)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Dataset comparison plots saved")
    
    logging.info(f"All visualizations saved to {output_dir}")

def main():
    """Main function to run visualizations."""
    try:
        # Define paths
        train_file = "datas/processed/train_features.csv"
        test_file = "datas/processed/test_features.csv"
        output_dir = "reports/figures"
        
        # Check if files exist
        if not os.path.exists(train_file):
            logging.error(f"Training file not found: {train_file}")
            return
        if not os.path.exists(test_file):
            logging.error(f"Test file not found: {test_file}")
            return
        
        # Create visualizations
        create_visualizations(train_file, test_file, output_dir)
        
        logging.info("Visualization process completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during visualization: {str(e)}")
        raise

if __name__ == "__main__":
    main()