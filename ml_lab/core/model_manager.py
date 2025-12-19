from typing import Dict, List, Type, Optional
from ..models.base import BaseModel, ModelCategory

class ModelManager:
    
    _registry: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register(cls, model_class: Type[BaseModel]) -> Type[BaseModel]:
        cls._registry[model_class.name] = model_class
        return model_class
    
    @classmethod
    def get_model(cls, name: str) -> BaseModel:
        if name not in cls._registry:
            raise ValueError(f"Model '{name}' not found. Available: {list(cls._registry.keys())}")
        return cls._registry[name]()
    
    @classmethod
    def get_model_class(cls, name: str) -> Type[BaseModel]:
        if name not in cls._registry:
            raise ValueError(f"Model '{name}' not found")
        return cls._registry[name]
    
    @classmethod
    def list_models(cls) -> List[str]:
        return list(cls._registry.keys())
    
    @classmethod
    def list_models_by_category(cls) -> Dict[ModelCategory, List[str]]:
        result = {cat: [] for cat in ModelCategory}
        for name, model_cls in cls._registry.items():
            result[model_cls.category].append(name)
        return result
    
    @classmethod
    def get_models_info(cls) -> List[dict]:
        result = []
        for name, model_cls in cls._registry.items():
            result.append({
                'name': name,
                'display_name': model_cls.display_name,
                'category': model_cls.category.value,
                'description': model_cls.description,
                'parameters': model_cls.get_parameters()
            })
        return result
    
    @classmethod  
    def clear_registry(cls):
        cls._registry.clear()

def register_all_models():
    from ..models.regressors import (
        GradientBoostingRegressorModel,
        RandomForestRegressorModel,
        KNeighborsRegressorModel,
        LassoModel,
        RidgeModel
    )
    from ..models.clustering import (
        KMeansModel,
        DBSCANModel,
        AgglomerativeModel
    )
    from ..models.classifiers import (
        KNeighborsClassifierModel,
        LogisticRegressionModel
    )
