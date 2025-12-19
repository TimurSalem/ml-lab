import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from ..base import BaseModel, ParameterConfig, ParameterType, ModelCategory, TrainingResult
from ...core.model_manager import ModelManager

@ModelManager.register
class DBSCANModel(BaseModel):
    
    name = "dbscan"
    display_name = "DBSCAN"
    category = ModelCategory.CLUSTERING
    description = "Кластеризация на основе плотности (DBSCAN)"
    
    @classmethod
    def get_parameters(cls) -> list[ParameterConfig]:
        return [
            ParameterConfig(
                name="eps",
                display_name="Epsilon (ε)",
                param_type=ParameterType.FLOAT,
                default=0.5,
                min_value=0.01,
                max_value=10.0,
                step=0.1,
                description="Максимальное расстояние между соседними точками"
            ),
            ParameterConfig(
                name="min_samples",
                display_name="Мин. образцов",
                param_type=ParameterType.INT,
                default=5,
                min_value=2,
                max_value=50,
                step=1,
                description="Минимальное число образцов для формирования плотной области"
            ),
            ParameterConfig(
                name="metric",
                display_name="Метрика",
                param_type=ParameterType.CHOICE,
                default="euclidean",
                choices=["euclidean", "manhattan", "chebyshev", "minkowski"],
                choice_labels=["Евклидова", "Манхэттенская", "Чебышёва", "Minkowski"],
                description="Метрика расстояния"
            ),
            ParameterConfig(
                name="algorithm",
                display_name="Алгоритм",
                param_type=ParameterType.CHOICE,
                default="auto",
                choices=["auto", "ball_tree", "kd_tree", "brute"],
                choice_labels=["Авто", "Ball Tree", "KD Tree", "Brute Force"],
                description="Алгоритм поиска соседей"
            ),
        ]
    
    def create_model(self, params: dict):
        defaults = {
            'eps': 0.5,
            'min_samples': 5,
            'metric': 'euclidean',
            'algorithm': 'auto',
            'n_jobs': -1
        }
        defaults.update(params)
        return DBSCAN(**defaults)
    
    def train(self, X_train: np.ndarray, X_test: np.ndarray,
              y_train: np.ndarray, y_test: np.ndarray,
              params: dict, feature_names: list = None) -> TrainingResult:
        import time
        start_time = time.time()
        
        self._X_train = X_train
        self._feature_names = feature_names or [f"x{i+1}" for i in range(X_train.shape[1])]
        
        X_full = np.vstack([X_train, X_test])
        
        self.model = self.create_model(params)
        labels = self.model.fit_predict(X_full)
        self.is_fitted = True
        
        # Calculate silhouette score (only if more than 1 cluster and no -1 only)
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        
        if n_clusters > 1 and len(labels[labels != -1]) > 1:
            mask = labels != -1
            score = silhouette_score(X_full[mask], labels[mask])
        else:
            score = 0.0
        
        n_noise = list(labels).count(-1)
        
        training_time = time.time() - start_time
        
        self._last_result = TrainingResult(
            train_score=score,
            test_score=score,
            predictions=labels,
            training_time=training_time,
            additional_metrics={
                'n_clusters': n_clusters,
                'n_noise_points': n_noise,
                'noise_ratio': n_noise / len(labels)
            }
        )
        
        return self._last_result
    
    def train_with_cv(self, X_train: np.ndarray, X_test: np.ndarray,
                      y_train: np.ndarray, y_test: np.ndarray,
                      params: dict, cv: int = 5,
                      feature_names: list = None) -> TrainingResult:
        return self.train(X_train, X_test, y_train, y_test, params, feature_names)
    
    @classmethod
    def get_grid_search_params(cls) -> dict:
        return {
            'eps': [0.1, 0.3, 0.5, 0.7, 1.0],
            'min_samples': [3, 5, 10, 15]
        }
