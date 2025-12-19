from sklearn.ensemble import GradientBoostingRegressor
from ..base import BaseModel, ParameterConfig, ParameterType, ModelCategory
from ...core.model_manager import ModelManager

@ModelManager.register
class GradientBoostingRegressorModel(BaseModel):
    
    name = "gradient_boosting_regressor"
    display_name = "Gradient Boosting Regressor"
    category = ModelCategory.REGRESSION
    description = "Ансамбль деревьев решений, обучаемых последовательно для минимизации ошибки"
    
    @classmethod
    def get_parameters(cls) -> list[ParameterConfig]:
        return [
            ParameterConfig(
                name="n_estimators",
                display_name="Количество деревьев",
                param_type=ParameterType.INT,
                default=100,
                min_value=1,
                max_value=500,
                step=5,
                description="Количество этапов бустинга"
            ),
            ParameterConfig(
                name="learning_rate",
                display_name="Скорость обучения",
                param_type=ParameterType.FLOAT,
                default=0.1,
                min_value=0.01,
                max_value=1.0,
                step=0.01,
                description="Сокращает вклад каждого дерева"
            ),
            ParameterConfig(
                name="max_depth",
                display_name="Максимальная глубина",
                param_type=ParameterType.INT,
                default=3,
                min_value=1,
                max_value=50,
                step=1,
                description="Максимальная глубина деревьев"
            ),
            ParameterConfig(
                name="min_samples_split",
                display_name="Мин. для разделения",
                param_type=ParameterType.INT,
                default=2,
                min_value=2,
                max_value=20,
                step=1,
                description="Минимальное число образцов для разделения узла"
            ),
            ParameterConfig(
                name="min_samples_leaf",
                display_name="Мин. в листе",
                param_type=ParameterType.INT,
                default=1,
                min_value=1,
                max_value=20,
                step=1,
                description="Минимальное число образцов в листе"
            ),
            ParameterConfig(
                name="max_features",
                display_name="Макс. признаков",
                param_type=ParameterType.CHOICE,
                default="sqrt",
                choices=["sqrt", "log2", None],
                choice_labels=["sqrt", "log2", "Все"],
                description="Количество признаков при разделении"
            ),
        ]
    
    def create_model(self, params: dict):
        defaults = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'warm_start': True,
            'n_iter_no_change': 5,
            'validation_fraction': 0.1
        }
        defaults.update(params)
        return GradientBoostingRegressor(**defaults)
    
    @classmethod
    def get_grid_search_params(cls) -> dict:
        return {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_leaf': [1, 2, 4]
        }
