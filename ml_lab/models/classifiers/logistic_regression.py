from sklearn.linear_model import LogisticRegression
from ..base import BaseModel, ParameterConfig, ParameterType, ModelCategory
from ...core.model_manager import ModelManager

@ModelManager.register
class LogisticRegressionModel(BaseModel):
    
    name = "logistic_regression"
    display_name = "Logistic Regression"
    category = ModelCategory.CLASSIFICATION
    description = "Логистическая регрессия для бинарной и мультиклассовой классификации"
    
    @classmethod
    def get_parameters(cls) -> list[ParameterConfig]:
        return [
            ParameterConfig(
                name="C",
                display_name="Обратная регуляризация (C)",
                param_type=ParameterType.FLOAT,
                default=1.0,
                min_value=0.001,
                max_value=100.0,
                step=0.1,
                description="Обратная сила регуляризации (меньше = сильнее)"
            ),
            ParameterConfig(
                name="max_iter",
                display_name="Макс. итераций",
                param_type=ParameterType.INT,
                default=1000,
                min_value=100,
                max_value=10000,
                step=100,
                description="Максимальное количество итераций для сходимости"
            ),
            ParameterConfig(
                name="solver",
                display_name="Решатель",
                param_type=ParameterType.CHOICE,
                default="lbfgs",
                choices=["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
                choice_labels=["L-BFGS", "Liblinear", "Newton-CG", "SAG", "SAGA"],
                description="Алгоритм оптимизации"
            ),
            ParameterConfig(
                name="penalty",
                display_name="Регуляризация",
                param_type=ParameterType.CHOICE,
                default="l2",
                choices=["l2", "l1", "elasticnet", None],
                choice_labels=["L2", "L1", "ElasticNet", "Нет"],
                description="Тип регуляризации"
            ),
        ]
    
    def create_model(self, params: dict):
        defaults = {
            'C': 1.0,
            'max_iter': 1000,
            'solver': 'lbfgs',
            'penalty': 'l2',
            'n_jobs': -1
        }
        defaults.update(params)
        
        # Remove multi_class if present (deprecated in sklearn 1.5+)
        defaults.pop('multi_class', None)
        
        solver = defaults.get('solver', 'lbfgs')
        penalty = defaults.get('penalty', 'l2')
        
        if solver == 'lbfgs' and penalty == 'l1':
            defaults['penalty'] = 'l2'
        
        if penalty == 'l1' and solver not in ['liblinear', 'saga']:
            defaults['solver'] = 'saga'
        
        if penalty == 'elasticnet':
            defaults['solver'] = 'saga'
            defaults['l1_ratio'] = 0.5
        
        return LogisticRegression(**defaults)
    
    def _get_scoring(self) -> str:
        return "accuracy"
    
    @classmethod
    def get_grid_search_params(cls) -> dict:
        return {
            'C': [0.01, 0.1, 1.0, 10.0],
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [100, 500]
        }
