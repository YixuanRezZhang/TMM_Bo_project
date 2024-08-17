import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.exceptions import NotFittedError

class IOManager:
    def __init__(self, root=None, method='standard'):
        self.root = root if root else os.getcwd()
        self.method = method
        if method == 'standard':
            self.scaler_X = StandardScaler()           
        elif self.method == 'minmax':
            self.scaler_X = MinMaxScaler()
        else:
            raise ValueError("Invalid method. Use 'standard' or 'minmax'.")
        self.scaler_y = StandardScaler()

    def read_data(self, file_name, target_props, feature_props=None, descriptor_type='magpie', handle_null=True, drop_non_numeric=True):
        file_path = os.path.join(self.root, f'{file_name}')
        data = pd.read_csv(file_path)

        # 检查指定的列是否存在
        missing_cols = [col for col in target_props if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        target_props = list(set(target_props))  # Remove duplicates if any

        # Check for null values
        if data.isnull().values.any():
            print(f'data contains null value!')
            if not handle_null:
                raise ValueError("Data contains null values. Please handle them or set handle_null to True.")
            else:
                data = self.handle_null_values(data, target_props, drop_non_numeric=drop_non_numeric)
        else:
            data = self.handle_null_values(data, target_props, drop_non_numeric=drop_non_numeric)

        # 如果没有指明feature_props, 使用除target外所有列
        if feature_props is None:
            feature_props = [col for col in data.columns if col not in target_props]
        print(f'used feature set: {feature_props}')
        
        # 检查指定的feature列是否存在
        missing_feature_cols = [col for col in feature_props if col not in data.columns]
        if missing_feature_cols:
            raise ValueError(f"Missing feature columns: {missing_feature_cols}")

        X = data[feature_props].to_numpy()
        y = data[target_props].to_numpy()

        if y.ndim == 1:
            y = np.expand_dims(y, -1)

        return X, y

    def handle_null_values(self, data, target_props, drop_non_numeric):
        # 处理target列的null值，通过删除包含null的行
        for target in target_props:
            if data[target].isnull().any():
                data = data.dropna(subset=[target])
            if drop_non_numeric and not pd.api.types.is_numeric_dtype(data[target]):
                unique_values = data[target].nunique()
                if unique_values == 2:
                    data[target] = data[target].astype('category').cat.codes
                else:
                    raise ValueError(f"Target column {target} contains non-numeric data that cannot be converted to binary classification.")

        # 处理非target列的null值，通过删除包含null的行
        for column in data.columns:
            if column not in target_props:
                if data[column].isnull().any():
                    # print(f'drop feature column contains null: {column}')
                    # data = data.drop(columns=[column])
                    print(f'drop samples contains null: {column}')
                    data = data.dropna(subset=[column])
                elif drop_non_numeric and not pd.api.types.is_numeric_dtype(data[column]):
                    print(f'drop non numeric column: {column}')
                    data = data.drop(columns=[column])

        return data

    def standardize_data(self, X, y=None, feature_range=(0, 1), custom_min=None, custom_max=None):
        if self.method == 'standard':
            X_scaled = self.scaler_X.fit_transform(X)
        elif self.method == 'minmax':
            self.scaler_X.feature_range = feature_range
            if custom_min is not None and custom_max is not None:
                if len(custom_min) != X.shape[1] or len(custom_max) != X.shape[1]:
                    raise ValueError("custom_min and custom_max must have the same dimensions as the features.")

                # 计算缩放比例和偏移量
                scale_X = (feature_range[1] - feature_range[0]) / (custom_max - custom_min)
                min_X = feature_range[0] - custom_min * scale_X
                X_scaled = X * scale_X + min_X
            else:
                X_scaled = self.scaler_X.fit_transform(X)
        else:
            raise ValueError("Invalid method. Use 'standard' or 'minmax'.")

        if y is not None:
            y_scaled = self.scaler_y.fit_transform(y)       
            return X_scaled, y_scaled
        else:
            return X_scaled

    def inverse_transform_X(self, X_scaled):
        return self.scaler_X.inverse_transform(X_scaled)

    def inverse_transform_y(self, y_scaled):
        return self.scaler_y.inverse_transform(y_scaled)

    def save_predictions(self, predictions, file_name):
        df = pd.DataFrame(predictions, columns=['Predictions'])
        df.to_csv(file_name, index=False)



# Example usage
if __name__ == "__main__":
    io_manager = IOManager()

    drop_props = ['prop_to_drop1', 'prop_to_drop2']  # Replace with actual column names
    props = ['target_prop']  # Replace with actual target column name
    file_name = 'data.csv'  # Replace with actual file name

    X, y = io_manager.read_data(file_name, drop_props, props)

    # StandardScaler
    X_standard, y_standard = io_manager.standardize_data(X, y, method='standard')

    # MinMaxScaler with feature range (0, 1) and custom min/max
    custom_min = [0, -10, 0]  # Replace with actual custom min values matching feature dimensions
    custom_max = [100, 50, 100]  # Replace with actual custom max values matching feature dimensions
    X_minmax, y_minmax = io_manager.standardize_data(X, y, method='minmax', feature_range=(0, 1), custom_min=custom_min, custom_max=custom_max)

    # Inverse transform y for both methods
    y_inv_standard = io_manager.inverse_transform_y(y_standard, method='standard')
    y_inv_minmax = io_manager.inverse_transform_y(y_minmax, method='minmax', custom_min=custom_min, custom_max=custom_max)

    # Save predictions example
    predictions = np.array([1.0, 2.0, 3.0])  # Replace with actual predictions
    io_manager.save_predictions(predictions, 'predictions.csv')
