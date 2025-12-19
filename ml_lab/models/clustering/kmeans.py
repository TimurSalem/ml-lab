import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from ..base import BaseModel, ParameterConfig, ParameterType, ModelCategory, TrainingResult
from ...core.model_manager import ModelManager

@ModelManager.register
class KMeansModel(BaseModel):
    
    name = "kmeans"
    display_name = "KMeans Clustering"
    category = ModelCategory.CLUSTERING
    description = "Кластеризация методом K-средних"
    
    @classmethod
    def get_parameters(cls) -> list[ParameterConfig]:
        return [
            ParameterConfig(
                name="n_clusters",
                display_name="Количество кластеров",
                param_type=ParameterType.INT,
                default=3,
                min_value=2,
                max_value=20,
                step=1,
                description="Количество кластеров для поиска"
            ),
            ParameterConfig(
                name="init",
                display_name="Инициализация",
                param_type=ParameterType.CHOICE,
                default="k-means++",
                choices=["k-means++", "random"],
                choice_labels=["K-Means++", "Случайная"],
                description="Метод инициализации центроидов"
            ),
            ParameterConfig(
                name="n_init",
                display_name="Число инициализаций",
                param_type=ParameterType.INT,
                default=10,
                min_value=1,
                max_value=100,
                step=5,
                description="Количество запусков с разными начальными центроидами"
            ),
            ParameterConfig(
                name="max_iter",
                display_name="Макс. итераций",
                param_type=ParameterType.INT,
                default=300,
                min_value=100,
                max_value=3000,
                step=100,
                description="Максимальное количество итераций"
            ),
            ParameterConfig(
                name="algorithm",
                display_name="Алгоритм",
                param_type=ParameterType.CHOICE,
                default="lloyd",
                choices=["lloyd", "elkan"],
                choice_labels=["Lloyd", "Elkan"],
                description="Алгоритм K-Means"
            ),
        ]
    
    def create_model(self, params: dict):
        defaults = {
            'n_clusters': 3,
            'init': 'k-means++',
            'n_init': 10,
            'max_iter': 300,
            'algorithm': 'lloyd',
            'random_state': 42
        }
        defaults.update(params)
        return KMeans(**defaults)
    
    def train(self, X_train: np.ndarray, X_test: np.ndarray,
              y_train: np.ndarray, y_test: np.ndarray,
              params: dict, feature_names: list = None) -> TrainingResult:
        import time
        start_time = time.time()
        
        self._X_train = X_train
        self._feature_names = feature_names or [f"x{i+1}" for i in range(X_train.shape[1])]
        
        # For clustering, we use all data (X_train + X_test combined)
        X_full = np.vstack([X_train, X_test])
        
        self.model = self.create_model(params)
        labels = self.model.fit_predict(X_full)
        self.is_fitted = True
        
        if len(np.unique(labels)) > 1:
            score = silhouette_score(X_full, labels)
        else:
            score = 0.0
        
        training_time = time.time() - start_time
        
        self._last_result = TrainingResult(
            train_score=score,
            test_score=score,
            predictions=labels,
            training_time=training_time,
            additional_metrics={
                'inertia': self.model.inertia_,
                'n_clusters': params.get('n_clusters', 3),
                'cluster_centers': self.model.cluster_centers_
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
            'n_clusters': [2, 3, 4, 5, 6, 7, 8],
            'init': ['k-means++', 'random'],
            'algorithm': ['lloyd', 'elkan']
        }
