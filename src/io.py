import os, pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.exceptions import NotFittedError
import logging

class IOManager:
    def __init__(self, root=None, method='standard', file_path=None):
        self.root = root if root else os.getcwd()
        self.method = method
        if method == 'standard':
            self.scaler_X = StandardScaler()           
        elif self.method == 'minmax':
            self.scaler_X = MinMaxScaler()
        else:
            raise ValueError("Invalid method. Use 'standard' or 'minmax'.")
        self.scaler_y = StandardScaler()
        self.file_path = file_path if file_path is not None else f'{os.getcwd()}/model_weights'

    def save_scalers(self, data_id):     
        os.makedirs(self.file_path, exist_ok=True)
        scaler_file = os.path.join(self.file_path, f"scaler{data_id}.pkl")
        with open(scaler_file, "wb") as f:
            pickle.dump({"scaler_X": self.scaler_X, "scaler_y": self.scaler_y}, f)
        logging.info(f"scalers{data_id} saved to {self.file_path} directory.")

    def load_scalers(self, data_id):
        try:
            scaler_file = os.path.join(self.file_path, f"scaler{data_id}.pkl")
            with open(scaler_file, "rb") as f:
                scalers = pickle.load(f)
            self.scaler_X = scalers["scaler_X"]
            self.scaler_y = scalers["scaler_y"]
            logging.info(f"scalers{data_id} loaded from {self.file_path} directory.")
        except FileNotFoundError:
            raise FileNotFoundError(f"scalers{data_id} files not found in {self.file_path} directory or {self.file_path} directory not exists.")

    def read_data(self, file_name, target_props, feature_props=None, drop_columns=None, descriptor_type='magpie', handle_null=True, drop_non_numeric=True):
        
        file_path = os.path.join(self.root, f'{file_name}')
        data = pd.read_csv(file_path)
        
        if drop_columns is not None:
            drop_columns = [col for col in drop_columns if col in data.columns]
            logging.info(f"drop columns: {drop_columns}")
            data = data.drop(columns=drop_columns)

        # Check whether the requested columns exist.
        missing_cols = [col for col in target_props if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        target_props = sorted(list(set(target_props)))  # Remove duplicates if any
        print('target_properties:', target_props)

        # Check for null values
        if data.isnull().values.any():
            logging.info(f'data contains null value!')
            if not handle_null:
                raise ValueError("Data contains null values. Please handle them or set handle_null to True.")
            else:
                data, non_numeric_columns = self.handle_null_values(data, target_props, drop_non_numeric=drop_non_numeric)
        else:
            data, non_numeric_columns = self.handle_null_values(data, target_props, drop_non_numeric=drop_non_numeric)

        # If feature_props is not specified, use all columns except targets.
        if feature_props is None:
            feature_props = [col for col in data.columns if col not in target_props]
            feature_props = [col for col in feature_props if col not in non_numeric_columns]
        logging.info(f'used feature set: {feature_props}')
        
        # Check whether the requested feature columns exist.
        missing_feature_cols = [col for col in feature_props if col not in data.columns]
        if missing_feature_cols:
            raise ValueError(f"Missing feature columns: {missing_feature_cols}")

        X = data[feature_props].to_numpy()
        y = data[target_props].to_numpy()

        if y.ndim == 1:
            y = np.expand_dims(y, -1)

        return X, y

    def read_candidate_data(self, file_name, target_props, feature_props=None, drop_columns=None, descriptor_type='magpie', drop_non_numeric=True):
        file_path = os.path.join(self.root, f'{file_name}')
        data = pd.read_csv(file_path)
        
        if drop_columns is not None:
            drop_columns = [col for col in drop_columns if col in data.columns]
            logging.info(f"drop columns: {drop_columns}")
            data = data.drop(columns=drop_columns)

        # Check whether the requested columns exist.
        missing_cols = [col for col in target_props if col not in data.columns]
        if missing_cols:
            print(f"No target column {missing_cols} in candidate_file")

        target_props = sorted(list(set(target_props)))  # Remove duplicates if any

        data, non_numeric_columns = self.handle_null_values(data, target_props, drop_non_numeric=drop_non_numeric, if_train_data=False)

        # If feature_props is not specified, use all columns except targets.
        if feature_props is None:
            feature_props = [col for col in data.columns if col not in target_props]
            feature_props = [col for col in feature_props if col not in non_numeric_columns]
        logging.info(f'used feature set: {feature_props}')
        
        # Check whether the requested feature columns exist.
        missing_feature_cols = [col for col in feature_props if col not in data.columns]
        if missing_feature_cols:
            raise ValueError(f"Missing feature columns: {missing_feature_cols}")

        X = data[feature_props].to_numpy()

        return X

    def handle_null_values(self, data, target_props, drop_non_numeric, if_train_data=True):
        # Handle null target values by dropping rows that contain them.    
        if if_train_data:
            for target in target_props:
                if data[target].isnull().any():
                    data = data.dropna(subset=[target])   # Drop rows with null values.
                    logging.info(f'drop samples contains null properties: {target}')
                if drop_non_numeric and not pd.api.types.is_numeric_dtype(data[target]): 
                    unique_values = data[target].nunique() # Count the number of unique values.
                    if unique_values == 2:
                        data[target] = data[target].astype('category').cat.codes # Convert categorical data to 0/1 codes.
                    else:
                        # Raise an error for non-binary categorical data.
                        raise ValueError(f"Target column {target} contains non-numeric data that cannot be converted to binary classification.")
        non_numeric_columns = []
        # Handle null values in non-target columns by dropping rows that contain them.
        for column in data.columns:
            if column not in target_props: # For non-target columns.
                if drop_non_numeric and not pd.api.types.is_numeric_dtype(data[column]):
                    logging.info(f'drop feature which is non numeric: {column}')
                    data = data.drop(columns=[column]) # Drop non-numeric columns directly.
                    # Log non-numeric column names.
                    non_numeric_columns.append(column)  
                elif data[column].isnull().any(): # Handle null values.
                    # logging.info(f'drop feature column contains null: {column}')
                    # data = data.drop(columns=[column]) # Drop non-numeric columns directly.
                    logging.info(f'drop samples contains null features: {column}')
                    data = data.dropna(subset=[column]) # Drop rows with null values.
        logging.info(f'length of cleaned data: {len(data)}')

        return data, non_numeric_columns


    def standardize_data(self, X=None, y=None, cand_X=None, cand_y=None, minmax_feature_range=(0, 1), if_train=False, data_id=None):
        """
        Standardize or scale data based on the chosen method (standard/minmax).
        Args:
            X: Training features (optional).
            y: Training targets (optional).
            cand_X: Candidate features for prediction (optional).
            cand_y: Candidate targets for prediction (optional).
            minmax_feature_range: Feature range for MinMaxScaler.
            if_train: Flag to indicate whether it's training mode (default=False).
        Returns:
            Tuple of scaled inputs in the same order as provided.
        """
        assert not (X is None and y is None and cand_X is None and cand_y is None), \
            "At least one of X, y, cand_X, or cand_y must be provided."
    
        # Initialize variables
        X_scaled, y_scaled, cand_X_scaled, cand_y_scaled = None, None, None, None
    
        if if_train:
            # Training mode: fit scalers and save them
            if self.method == 'standard':
                if X is not None:
                    self.scaler_X.fit(X)
                    X_scaled = self.scaler_X.transform(X)
                if y is not None:
                    self.scaler_y.fit(y)
                    y_scaled = self.scaler_y.transform(y)
            elif self.method == 'minmax':
                self.scaler_X.feature_range = minmax_feature_range
                if X is not None:
                    self.scaler_X.fit(X)
                    X_scaled = self.scaler_X.transform(X)
                if y is not None:
                    self.scaler_y.fit(y)
                    y_scaled = self.scaler_y.transform(y)
            else:
                raise ValueError("Invalid method. Use 'standard' or 'minmax'.")
                
            zero_var_X = self.scaler_X.scale_ == 0
            self.scaler_X.scale_[zero_var_X] = 1.0
            zero_var_y = self.scaler_y.scale_ == 0
            self.scaler_y.scale_[zero_var_y] = 1.0

            if cand_X is not None:
                cand_X_scaled = self.scaler_X.transform(cand_X)
            if cand_y is not None:
                cand_y_scaled = self.scaler_y.transform(cand_y)
    
            # Save scalers after fitting
            self.save_scalers(data_id)
        else:
            # Prediction mode: load scalers and transform data
            self.load_scalers(data_id)
    
            if X is not None:
                X_scaled = self.scaler_X.transform(X)
            if y is not None:
                y_scaled = self.scaler_y.transform(y)
            if cand_X is not None:
                cand_X_scaled = self.scaler_X.transform(cand_X)
            if cand_y is not None:
                cand_y_scaled = self.scaler_y.transform(cand_y)
    
        # Dynamically construct the return tuple based on input arguments
        return_tuple = tuple(var for var in [X_scaled, y_scaled, cand_X_scaled, cand_y_scaled] if var is not None)
    
        # If there's only one element, return it directly
        return return_tuple[0] if len(return_tuple) == 1 else return_tuple


    def inverse_transform_X(self, X_scaled):
        return self.scaler_X.inverse_transform(X_scaled)

    def inverse_transform_y(self, y_scaled):
        return self.scaler_y.inverse_transform(y_scaled)

    def save_predictions(self, predictions, file_name):
        df = pd.DataFrame(predictions, columns=['Predictions'])
        df.to_csv(file_name, index=False)


