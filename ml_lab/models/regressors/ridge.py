from sklearn.linear_model import Ridge
from ..base import BaseModel, ParameterConfig, ParameterType, ModelCategory
from ...core.model_manager import ModelManager

@ModelManager.register
class RidgeModel(BaseModel):
    
    name = "ridge"
    display_name = "Ridge (L2 Regularization)"
    category = ModelCategory.REGRESSION
    description = "Линейная регрессия с L2-регуляризацией"
    
    @classmethod
    def get_parameters(cls) -> list[ParameterConfig]:
        return [
            ParameterConfig(
                name="alpha",
                display_name="Alpha (λ)",
                param_type=ParameterType.FLOAT,
                default=1.0,
                min_value=0.001,
                max_value=100.0,
                step=0.1,
                description="Сила регуляризации (больше = сильнее)"
            ),
            ParameterConfig(
                name="max_iter",
                display_name="Макс. итераций",
                param_type=ParameterType.INT,
                default=1000,
                min_value=100,
                max_value=100000,
                step=100,
                description="Максимальное количество итераций"
            ),
            ParameterConfig(
                name="tol",
                display_name="Толерантность",
                param_type=ParameterType.FLOAT,
                default=0.0001,
                min_value=0.00001,
                max_value=0.01,
                step=0.0001,
                description="Порог сходимости"
            ),
            ParameterConfig(
                name="solver",
                display_name="Решатель",
                param_type=ParameterType.CHOICE,
                default="auto",
                choices=["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
                choice_labels=["Авто", "SVD", "Cholesky", "LSQR", "Sparse CG", "SAG", "SAGA"],
                description="Алгоритм решения"
            ),
        ]
    
    def create_model(self, params: dict):
        defaults = {
            'alpha': 1.0,
            'max_iter': 1000,
            'tol': 0.0001,
            'solver': 'auto'
        }
        defaults.update(params)
        return Ridge(**defaults)
    
    @classmethod
    def get_grid_search_params(cls) -> dict:
        return {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'solver': ['auto', 'svd', 'lsqr']
        }
