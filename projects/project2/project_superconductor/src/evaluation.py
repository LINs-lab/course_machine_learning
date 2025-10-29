import torch
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os


def evaluate_model(model, X_test, y_test, model_name="TcPredictor"):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained PyTorch model
        X_test: Test features as numpy array or torch tensor
        y_test: Test targets as numpy array or torch tensor
        model_name: Name of the model for reporting
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Convert to tensors if they're numpy arrays
    if isinstance(X_test, np.ndarray):
        X_test = torch.FloatTensor(X_test)
    if isinstance(y_test, np.ndarray):
        y_test = torch.FloatTensor(y_test)
    
    # Move to the same device as the model
    device = next(model.parameters()).device
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    # Make predictions
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy()
    
    # Convert targets back to numpy for sklearn
    y_test = y_test.cpu().numpy()
    
    # Flatten in case they're 2D
    y_pred = y_pred.flatten()
    y_test = y_test.flatten()
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error) if possible
    # Avoid division by zero
    non_zero_mask = y_test != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_test[non_zero_mask] - y_pred[non_zero_mask]) / y_test[non_zero_mask])) * 100
    else:
        mape = float('inf')  # Undefined if all true values are zero
    
    # Create evaluation report
    metrics = {
        'model_name': model_name,
        'r2_score': r2,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'n_samples': len(y_test)
    }
    
    # Print metrics
    print(f"\n--- Model Evaluation: {model_name} ---")
    print(f"Number of test samples: {metrics['n_samples']}")
    print(f"R² Score: {metrics['r2_score']:.4f}")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    
    # Create visualization
    create_evaluation_plots(y_test, y_pred, model_name)
    
    return metrics


def create_evaluation_plots(y_true, y_pred, model_name="TcPredictor"):
    """
    Create evaluation plots for the model.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model for plot titles
    """
    # Create figures directory if it doesn't exist
    if not os.path.exists("./figures"):
        os.makedirs("./figures")
    
    # Calculate metrics for plot titles
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: True vs Predicted values
    axes[0].scatter(y_true, y_pred, alpha=0.6)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_xlabel('True Tc (K)')
    axes[0].set_ylabel('Predicted Tc (K)')
    axes[0].set_title(f'True vs Predicted Tc\nR² = {r2:.4f}, RMSE = {rmse:.2f} K')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('Predicted Tc (K)')
    axes[1].set_ylabel('Residuals (True - Predicted)')
    axes[1].set_title('Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plots
    filename = f"./figures/{model_name}_evaluation.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Evaluation plots saved to {filename}")
    
    plt.show()


def compare_models(models_data, model_names):
    """
    Compare multiple models based on their evaluation metrics.
    
    Args:
        models_data: List of evaluation metric dictionaries
        model_names: List of model names
        
    Returns:
        DataFrame with comparison
    """
    import pandas as pd
    
    # Create comparison DataFrame
    comparison_data = {
        'Model': model_names,
        'R² Score': [data['r2_score'] for data in models_data],
        'RMSE': [data['rmse'] for data in models_data],
        'MAE': [data['mae'] for data in models_data],
        'MAPE (%)': [data['mape'] for data in models_data]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by R² score (descending)
    comparison_df = comparison_df.sort_values(by='R² Score', ascending=False)
    
    print("\n--- Model Comparison ---")
    print(comparison_df.to_string(index=False))
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # R² comparison
    axes[0, 0].bar(comparison_df['Model'], comparison_df['R² Score'])
    axes[0, 0].set_title('R² Score Comparison')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # RMSE comparison
    axes[0, 1].bar(comparison_df['Model'], comparison_df['RMSE'])
    axes[0, 1].set_title('RMSE Comparison')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # MAE comparison
    axes[1, 0].bar(comparison_df['Model'], comparison_df['MAE'])
    axes[1, 0].set_title('MAE Comparison')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # MAPE comparison
    axes[1, 1].bar(comparison_df['Model'], comparison_df['MAPE (%)'])
    axes[1, 1].set_title('MAPE Comparison')
    axes[1, 1].set_ylabel('MAPE (%)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the comparison
    filename = "./figures/model_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Model comparison plot saved to {filename}")
    
    plt.show()
    
    return comparison_df


def save_evaluation_report(metrics, filepath="./evaluation_report.txt"):
    """
    Save evaluation metrics to a text file.
    
    Args:
        metrics: Dictionary with evaluation metrics
        filepath: Path to save the report
    """
    with open(filepath, 'w') as f:
        f.write("Superconductor Tc Prediction Model Evaluation Report\n")
        f.write("="*50 + "\n\n")
        f.write(f"Model: {metrics['model_name']}\n")
        f.write(f"Test samples: {metrics['n_samples']}\n\n")
        f.write("Metrics:\n")
        f.write(f"  R² Score:  {metrics['r2_score']:.6f}\n")
        f.write(f"  MSE:       {metrics['mse']:.6f}\n")
        f.write(f"  RMSE:      {metrics['rmse']:.6f}\n")
        f.write(f"  MAE:       {metrics['mae']:.6f}\n")
        f.write(f"  MAPE:      {metrics['mape']:.2f}%\n")
    
    print(f"Evaluation report saved to {filepath}")


if __name__ == "__main__":
    # Example usage (with dummy data)
    print("Evaluation module created. Example usage requires trained model and test data.")
    
    # This would typically be called after training:
    # model = torch.load('models/best_model.pth')
    # metrics = evaluate_model(model, X_test, y_test)
    # save_evaluation_report(metrics)