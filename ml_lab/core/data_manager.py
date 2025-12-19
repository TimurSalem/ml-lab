import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass, field

@dataclass
class DataInfo:
    filename: str
    shape: Tuple[int, int]
    columns: List[str]
    dtypes: dict
    target_column: Optional[str] = None
    feature_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    numeric_columns: List[str] = field(default_factory=list)

class DataManager:
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.data_encoded: Optional[pd.DataFrame] = None
        self.data_info: Optional[DataInfo] = None
        self.label_encoders: dict = {}
        
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        
        self.feature_names: List[str] = []
        self.target_name: str = ""
    
    def load_file(self, filepath: str) -> DataInfo:
        if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
            self.data = pd.read_excel(filepath)
        elif filepath.endswith('.csv'):
            self.data = pd.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")
        
        categorical_cols = []
        numeric_cols = []
        
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        
        self.data_info = DataInfo(
            filename=filepath,
            shape=self.data.shape,
            columns=list(self.data.columns),
            dtypes={col: str(dtype) for col, dtype in self.data.dtypes.items()},
            categorical_columns=categorical_cols,
            numeric_columns=numeric_cols
        )
        
        return self.data_info
    
    def set_target_and_features(self, target_column: str, 
                                 feature_columns: Optional[List[str]] = None) -> None:
        if self.data is None:
            raise ValueError("No data loaded. Call load_file() first.")
        
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        self.target_name = target_column
        self.data_info.target_column = target_column
        
        if feature_columns is None:
            feature_columns = [col for col in self.data.columns if col != target_column]
        
        self.feature_names = feature_columns
        self.data_info.feature_columns = feature_columns
    
    def encode_categorical(self, apply_log_transform: bool = False) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("No data loaded. Call load_file() first.")
        
        self.data_encoded = self.data.copy()
        self.label_encoders = {}
        
        for column in self.data_encoded.columns:
            if self.data_encoded[column].dtype == 'object':
                encoder = LabelEncoder()
                self.data_encoded[column] = self.data_encoded[column].fillna('MISSING')
                self.data_encoded[column] = encoder.fit_transform(
                    self.data_encoded[column].astype(str)
                )
                self.label_encoders[column] = encoder
        
        return self.data_encoded
    
    def handle_missing_values(self, strategy: str = 'drop') -> int:
        if self.data is None:
            raise ValueError("No data loaded. Call load_file() first.")
        
        original_rows = len(self.data)
        
        if strategy == 'drop':
            self.data = self.data.dropna()
            affected = original_rows - len(self.data)
        elif strategy == 'mean':
            for col in self.data.columns:
                if self.data[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    self.data[col] = self.data[col].fillna(self.data[col].mean())
                else:
                    mode_val = self.data[col].mode()
                    if len(mode_val) > 0:
                        self.data[col] = self.data[col].fillna(mode_val[0])
            affected = self.data.isnull().sum().sum()
        elif strategy == 'median':
            for col in self.data.columns:
                if self.data[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    self.data[col] = self.data[col].fillna(self.data[col].median())
                else:
                    mode_val = self.data[col].mode()
                    if len(mode_val) > 0:
                        self.data[col] = self.data[col].fillna(mode_val[0])
            affected = self.data.isnull().sum().sum()
        elif strategy == 'mode':
            for col in self.data.columns:
                mode_val = self.data[col].mode()
                if len(mode_val) > 0:
                    self.data[col] = self.data[col].fillna(mode_val[0])
            affected = self.data.isnull().sum().sum()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        self.data_encoded = None
        
        if self.data_info:
            self.data_info.shape = self.data.shape
        
        return affected
    
    def get_missing_count(self) -> int:
        if self.data is None:
            return 0
        return self.data.isnull().sum().sum()
    
    def get_missing_by_column(self) -> dict:
        if self.data is None:
            return {}
        return self.data.isnull().sum().to_dict()
    
    def prepare_data(self, test_size: float = 0.25, 
                     random_state: int = 42,
                     apply_log_transform: bool = False) -> Tuple[np.ndarray, ...]:
        if self.data is None:
            raise ValueError("No data loaded. Call load_file() first.")
        
        if not self.target_name:
            raise ValueError("Target column not set. Call set_target_and_features() first.")
        
        if self.target_name not in self.data.columns:
            raise ValueError(f"Target column '{self.target_name}' not found in data. "
                           f"Available columns: {list(self.data.columns)}")
        
        valid_features = [f for f in self.feature_names if f in self.data.columns]
        if not valid_features:
            raise ValueError(f"No valid feature columns found. "
                           f"Available columns: {list(self.data.columns)}")
        
        self.feature_names = valid_features
        
        # Always re-encode to ensure fresh encoding after any data changes
        self.encode_categorical()
        
        X = self.data_encoded[self.feature_names].values
        y = self.data_encoded[self.target_name].values
        
        # Apply log transform if requested (only for numeric targets)
        if apply_log_transform:
            if np.issubdtype(y.dtype, np.number) and np.all(y > 0):
                y = np.log(y)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_data_preview(self, n_rows: int = 10) -> pd.DataFrame:
        if self.data is None:
            return pd.DataFrame()
        return self.data.head(n_rows)
    
    def get_encoded_preview(self, n_rows: int = 10) -> pd.DataFrame:
        if self.data_encoded is None:
            return pd.DataFrame()
        return self.data_encoded.head(n_rows)
    
    def get_statistics(self) -> dict:
        if self.data is None:
            return {}
        
        return {
            'shape': self.data.shape,
            'columns': len(self.data.columns),
            'rows': len(self.data),
            'missing_values': self.data.isnull().sum().to_dict(),
            'dtypes': {col: str(dtype) for col, dtype in self.data.dtypes.items()}
        }
    
    def get_full_data(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.data_encoded is None:
            self.encode_categorical()
        
        X = self.data_encoded[self.feature_names].values
        y = self.data_encoded[self.target_name].values
        
        return X, y
