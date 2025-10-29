# Superconductor Critical Temperature Prediction

This project focuses on predicting the critical temperature (Tc) of superconductors using machine learning models. The goal is to predict Tc values based on the chemical composition and structural properties of superconducting materials.

## Project Structure

```
project_superconductor/
├── data/                 # Dataset files
├── src/                  # Source code
│   ├── data_processor.py # Data processing utilities
│   ├── model.py          # Model architecture
│   ├── train_tc_prediction.py  # Training script
│   └── inference_tc.py   # Inference script
├── models/               # Trained model files
├── figures/              # Generated plots and visualizations
├── scripts/              # Utility scripts
├── tests/                # Test files
└── README.md             # Project documentation
```

## Dataset

The dataset contains information about various superconductors with the following columns:
- `num`: Unique identifier
- `name`: Material name
- `element`: Chemical formula
- `str3`: Structure type
- `utc`: Unknown column
- `tc`: Critical temperature (target variable)
- `journal`: Source reference

## Requirements

- Python 3.7+
- PyTorch
- scikit-learn
- NumPy
- Matplotlib
- Pandas (optional, for data exploration)

## Usage

### Training

To train the model:
```bash
cd project_superconductor
python src/train_tc_prediction.py
```

### Inference

To make predictions with a trained model:
```bash
python src/inference_tc.py
```

## Model Architecture

The model uses a neural network architecture suitable for regression tasks to predict the critical temperature of superconductors. The architecture can be adjusted based on the complexity of the dataset.

## Data Processing

The data processor handles:
- Loading the superconductor dataset
- Feature engineering from chemical formulas
- Data normalization
- Train/test split (80:20 ratio)

## Evaluation Metrics

The model is evaluated using:
- R² (coefficient of determination)
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)

## Results

After training with an 80:20 train-test split, the model reports:
- Best R² score on test set
- RMSE and MAE values
- Visualizations of training progress and predictions

## Files

- `src/train_tc_prediction.py`: Main training script
- `src/inference_tc.py`: Inference/prediction script
- `src/data_processor.py`: Data loading and preprocessing utilities
- `src/model.py`: Neural network model definition
- `models/`: Directory for saved models
- `figures/`: Directory for generated plots