import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SuperconDataProcessor:
    """
    Data processor for superconductor dataset to prepare features for Tc prediction.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None

    def extract_features(self, df):
        """
        Extract features from the superconductor data.
        
        Args:
            df: DataFrame with superconductor data
            
        Returns:
            DataFrame with extracted features
        """
        pass


    def _parse_formula(self, formula):
        """
        Parse chemical formula to extract elemental composition features.

        TODO: Q - How to represent chemical formula of Superconductors with a vector/matrix?
            To do this, you should first understand the meaning of it by reading some articles and reports.

        Args:
            formula: Chemical formula string (e.g., "Ba0.2La1.8Cu1O4-Y")
            
        Returns:
            Dictionary with elemental composition features
        """
        pass

    def _encode_structure(self, structure):
        pass

    def load_and_process_data(self, data_path, test_size=0.2):
        print(f"Loading data from {data_path}")
        
        # Load the dataset
        df = pd.read_csv(data_path, sep='\t')  # Assuming TSV format based on filename
        
        # Extract features
        feature_df = self.extract_features(df)
        
        # Prepare target variable (tc - critical temperature)
        y = df['tc'].values.astype(np.float32)
        
        # Prepare feature matrix
        X = feature_df.values.astype(np.float32)
        
        # Store feature column names for reference
        self.feature_columns = feature_df.columns.tolist()
        
        # Split the data (80:20 split as requested)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale the features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        print(f"Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        print(f"Feature dimension: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test

    def process_input(self, input_data):
        """
        Process single input for inference.

        Args:
            input_data: Dictionary containing 'element' (formula) and optional 'str3' (structure)

        Returns:
            Processed feature vector
        """
        pass

    def get_feature_names(self):
        """Return the names of the features used."""
        return self.feature_columns