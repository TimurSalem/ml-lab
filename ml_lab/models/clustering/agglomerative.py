import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from ..base import BaseModel, ParameterConfig, ParameterType, ModelCategory, TrainingResult
from ...core.model_manager import ModelManager

@ModelManager.register
class AgglomerativeModel(BaseModel):
    
    name = "agglomerative"
    display_name = "Agglomerative Clustering"
    category = ModelCategory.CLUSTERING
    description = "Иерархическая агломеративная кластеризация"
    
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
                name="linkage",
                display_name="Связь",
                param_type=ParameterType.CHOICE,
                default="ward",
                choices=["ward", "complete", "average", "single"],
                choice_labels=["Ward", "Complete", "Average", "Single"],
                description="Критерий связи между кластерами"
            ),
            ParameterConfig(
                name="metric",
                display_name="Метрика",
                param_type=ParameterType.CHOICE,
                default="euclidean",
                choices=["euclidean", "manhattan", "cosine"],
                choice_labels=["Евклидова", "Манхэттенская", "Косинусная"],
                description="Метрика расстояния (не для Ward)"
            ),
        ]
    
    def create_model(self, params: dict):
        defaults = {
            'n_clusters': 3,
            'linkage': 'ward',
        }
        defaults.update(params)
        
        if defaults.get('linkage') == 'ward':
            defaults.pop('metric', None)
        elif 'metric' not in defaults:
            defaults['metric'] = 'euclidean'
            
        return AgglomerativeClustering(**defaults)
    
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
                'n_clusters': params.get('n_clusters', 3),
                'n_leaves': self.model.n_leaves_,
                'n_connected_components': self.model.n_connected_components_
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
            'n_clusters': [2, 3, 4, 5, 6],
            'linkage': ['ward', 'complete', 'average']
        }
