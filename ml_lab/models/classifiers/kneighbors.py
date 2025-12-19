from sklearn.neighbors import KNeighborsClassifier
from ..base import BaseModel, ParameterConfig, ParameterType, ModelCategory
from ...core.model_manager import ModelManager

@ModelManager.register
class KNeighborsClassifierModel(BaseModel):
    
    name = "kneighbors_classifier"
    display_name = "K-Neighbors Classifier"
    category = ModelCategory.CLASSIFICATION
    description = "Классификация на основе k ближайших соседей"
    
    @classmethod
    def get_parameters(cls) -> list[ParameterConfig]:
        return [
            ParameterConfig(
                name="n_neighbors",
                display_name="Количество соседей",
                param_type=ParameterType.INT,
                default=5,
                min_value=1,
                max_value=50,
                step=1,
                description="Количество соседей для голосования"
            ),
            ParameterConfig(
                name="weights",
                display_name="Веса",
                param_type=ParameterType.CHOICE,
                default="uniform",
                choices=["uniform", "distance"],
                choice_labels=["Равные", "По расстоянию"],
                description="Способ взвешивания соседей"
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
            ParameterConfig(
                name="leaf_size",
                display_name="Размер листа",
                param_type=ParameterType.INT,
                default=30,
                min_value=1,
                max_value=100,
                step=5,
                description="Размер листа для ball_tree/kd_tree"
            ),
            ParameterConfig(
                name="metric",
                display_name="Метрика",
                param_type=ParameterType.CHOICE,
                default="minkowski",
                choices=["minkowski", "euclidean", "manhattan", "chebyshev"],
                choice_labels=["Minkowski", "Евклидова", "Манхэттенская", "Чебышёва"],
                description="Метрика расстояния"
            ),
        ]
    
    def create_model(self, params: dict):
        defaults = {
            'n_neighbors': 5,
            'weights': 'uniform',
            'algorithm': 'auto',
            'leaf_size': 30,
            'metric': 'minkowski',
            'n_jobs': -1
        }
        defaults.update(params)
        return KNeighborsClassifier(**defaults)
    
    def _get_scoring(self) -> str:
        return "accuracy"
    
    @classmethod
    def get_grid_search_params(cls) -> dict:
        return {
            'n_neighbors': [3, 5, 7, 10, 15],
            'weights': ['uniform', 'distance'],
            'algorithm': ['ball_tree', 'kd_tree']
        }
