from sklearn.ensemble import RandomForestRegressor
from ..base import BaseModel, ParameterConfig, ParameterType, ModelCategory
from ...core.model_manager import ModelManager

@ModelManager.register
class RandomForestRegressorModel(BaseModel):
    
    name = "random_forest_regressor"
    display_name = "Random Forest Regressor"
    category = ModelCategory.REGRESSION
    description = "Ансамбль деревьев решений, обучаемых на случайных подвыборках"
    
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
                step=10,
                description="Количество деревьев в лесу"
            ),
            ParameterConfig(
                name="max_depth",
                display_name="Максимальная глубина",
                param_type=ParameterType.INT,
                default=10,
                min_value=1,
                max_value=100,
                step=1,
                description="Максимальная глубина деревьев (None = без ограничений)"
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
                choices=["sqrt", "log2", 1.0],
                choice_labels=["sqrt", "log2", "Все"],
                description="Количество признаков при разделении"
            ),
        ]
    
    def create_model(self, params: dict):
        defaults = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'n_jobs': -1,
            'oob_score': True,
            'warm_start': True
        }
        defaults.update(params)
        return RandomForestRegressor(**defaults)
    
    @classmethod
    def get_grid_search_params(cls) -> dict:
        return {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 20, None],
            'min_samples_leaf': [1, 2, 4]
        }
