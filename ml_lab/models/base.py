from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable
import numpy as np

class ParameterType(Enum):
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    CHOICE = "choice"
    INT_RANGE = "int_range"
    FLOAT_RANGE = "float_range"

class ModelCategory(Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"

@dataclass
class ParameterConfig:
    name: str
    display_name: str
    param_type: ParameterType
    default: Any
    description: str = ""
    
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    
    choices: Optional[list] = None
    choice_labels: Optional[list] = None
    
    range_values: Optional[list] = None
    
    def __post_init__(self):
        if self.param_type == ParameterType.INT:
            self.step = self.step or 1
        elif self.param_type == ParameterType.FLOAT:
            self.step = self.step or 0.01

@dataclass
class TrainingResult:
    train_score: float
    test_score: float
    best_params: Optional[dict] = None
    cv_scores: Optional[list] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    feature_importances: Optional[np.ndarray] = None
    predictions: Optional[np.ndarray] = None
    training_time: float = 0.0
    additional_metrics: dict = field(default_factory=dict)

@dataclass 
class Visualization:
    name: str
    title: str
    plot_function: Callable
    description: str = ""

class BaseModel(ABC):
    
    name: str = "Base Model"
    display_name: str = "Base Model"
    category: ModelCategory = ModelCategory.REGRESSION
    description: str = ""
    
    def __init__(self):
        self.model = None
        self.is_fitted = False
        self._last_result: Optional[TrainingResult] = None
        self._scaler = None
        self._X_train = None
        self._X_test = None
        self._y_train = None
        self._y_test = None
        self._feature_names = None
    
    @classmethod
    @abstractmethod
    def get_parameters(cls) -> list[ParameterConfig]:
        pass
    
    @abstractmethod
    def create_model(self, params: dict) -> Any:
        pass
    
    def train(self, X_train: np.ndarray, X_test: np.ndarray, 
              y_train: np.ndarray, y_test: np.ndarray,
              params: dict, feature_names: list = None) -> TrainingResult:
        import time
        start_time = time.time()
        
        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test
        self._feature_names = feature_names or [f"x{i+1}" for i in range(X_train.shape[1])]
        
        self.model = self.create_model(params)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        predictions = self.model.predict(X_test)
        
        feature_importances = None
        n_features = X_train.shape[1]
        
        if hasattr(self.model, 'feature_importances_'):
            fi = self.model.feature_importances_
            if fi.shape[0] == n_features:
                feature_importances = fi
        elif hasattr(self.model, 'coef_'):
            coef = self.model.coef_
            # Handle multi-class case where coef_ is 2D (n_classes, n_features)
            if coef.ndim == 2:
                fi = np.abs(coef).mean(axis=0)
            else:
                fi = np.abs(coef)
            if len(fi) == n_features:
                feature_importances = fi
        
        training_time = time.time() - start_time
        
        self._last_result = TrainingResult(
            train_score=train_score,
            test_score=test_score,
            predictions=predictions,
            feature_importances=feature_importances,
            training_time=training_time
        )
        
        return self._last_result
    
    def train_with_cv(self, X_train: np.ndarray, X_test: np.ndarray,
                      y_train: np.ndarray, y_test: np.ndarray,
                      params: dict, cv: int = 5,
                      feature_names: list = None) -> TrainingResult:
        from sklearn.model_selection import cross_val_score
        import time
        
        start_time = time.time()
        
        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test
        self._feature_names = feature_names or [f"x{i+1}" for i in range(X_train.shape[1])]
        
        self.model = self.create_model(params)
        
        scoring = self._get_scoring()
        cv_scores = cross_val_score(self.model, X_train, y_train, 
                                     cv=cv, scoring=scoring, n_jobs=-1)
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        predictions = self.model.predict(X_test)
        
        feature_importances = None
        n_features = X_train.shape[1]
        
        if hasattr(self.model, 'feature_importances_'):
            fi = self.model.feature_importances_
            if fi.shape[0] == n_features:
                feature_importances = fi
        elif hasattr(self.model, 'coef_'):
            coef = self.model.coef_
            if coef.ndim == 2:
                fi = np.abs(coef).mean(axis=0)
            else:
                fi = np.abs(coef)
            if len(fi) == n_features:
                feature_importances = fi
        
        training_time = time.time() - start_time
        
        self._last_result = TrainingResult(
            train_score=train_score,
            test_score=test_score,
            cv_scores=cv_scores.tolist(),
            cv_mean=float(np.mean(cv_scores)),
            cv_std=float(np.std(cv_scores)),
            predictions=predictions,
            feature_importances=feature_importances,
            training_time=training_time
        )
        
        return self._last_result
    
    def train_with_grid_search(self, X_train: np.ndarray, X_test: np.ndarray,
                               y_train: np.ndarray, y_test: np.ndarray,
                               param_grid: dict, cv: int = 5,
                               feature_names: list = None) -> TrainingResult:
        from sklearn.model_selection import GridSearchCV
        import time
        
        start_time = time.time()
        
        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test
        self._feature_names = feature_names or [f"x{i+1}" for i in range(X_train.shape[1])]
        
        base_model = self.create_model({})
        
        scoring = self._get_scoring()
        grid = GridSearchCV(base_model, param_grid, cv=cv, 
                           scoring=scoring, n_jobs=-1, refit=True)
        grid.fit(X_train, y_train)
        
        self.model = grid.best_estimator_
        self.is_fitted = True
        
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        predictions = self.model.predict(X_test)
        
        feature_importances = None
        n_features = X_train.shape[1]
        
        if hasattr(self.model, 'feature_importances_'):
            fi = self.model.feature_importances_
            if fi.shape[0] == n_features:
                feature_importances = fi
        elif hasattr(self.model, 'coef_'):
            coef = self.model.coef_
            if coef.ndim == 2:
                fi = np.abs(coef).mean(axis=0)
            else:
                fi = np.abs(coef)
            if len(fi) == n_features:
                feature_importances = fi
        
        training_time = time.time() - start_time
        
        self._last_result = TrainingResult(
            train_score=train_score,
            test_score=test_score,
            best_params=grid.best_params_,
            cv_mean=float(grid.best_score_),
            predictions=predictions,
            feature_importances=feature_importances,
            training_time=training_time
        )
        
        return self._last_result
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call train() first.")
        return self.model.predict(X)
    
    def _get_scoring(self) -> str:
        if self.category == ModelCategory.REGRESSION:
            return "r2"
        elif self.category == ModelCategory.CLASSIFICATION:
            return "accuracy"
        else:
            return "r2"
    
    def get_visualizations(self) -> list[Visualization]:
        return []
    
    def get_last_result(self) -> Optional[TrainingResult]:
        return self._last_result
    
    def save(self, filepath: str) -> None:
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'is_fitted': self.is_fitted,
                'scaler': self._scaler,
                'feature_names': self._feature_names
            }, f)
    
    def load(self, filepath: str) -> None:
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.is_fitted = data['is_fitted']
            self._scaler = data.get('scaler')
            self._feature_names = data.get('feature_names')
