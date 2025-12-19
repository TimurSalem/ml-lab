from typing import Optional, Type
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, MaxAbsScaler, 
    RobustScaler, Normalizer
)

AVAILABLE_SCALERS = {
    'none': {
        'name': 'Без масштабирования',
        'class': None,
        'description': 'Использовать данные без преобразования'
    },
    'standard': {
        'name': 'StandardScaler',
        'class': StandardScaler,
        'description': 'Стандартизация: среднее = 0, СКО = 1'
    },
    'minmax': {
        'name': 'MinMaxScaler', 
        'class': MinMaxScaler,
        'description': 'Масштабирование в диапазон [0, 1]'
    },
    'maxabs': {
        'name': 'MaxAbsScaler',
        'class': MaxAbsScaler,
        'description': 'Масштабирование по максимальному абсолютному значению'
    },
    'robust': {
        'name': 'RobustScaler',
        'class': RobustScaler,
        'description': 'Устойчив к выбросам (использует медиану и IQR)'
    },
    'normalizer': {
        'name': 'Normalizer',
        'class': Normalizer,
        'description': 'Нормализация каждого образца по единичной норме'
    }
}

class ScalerFactory:
    
    @staticmethod
    def get_scaler(scaler_key: str):
        if scaler_key not in AVAILABLE_SCALERS:
            raise ValueError(f"Unknown scaler: {scaler_key}")
        
        scaler_info = AVAILABLE_SCALERS[scaler_key]
        scaler_class = scaler_info['class']
        
        if scaler_class is None:
            return None
        
        return scaler_class()
    
    @staticmethod
    def get_scaler_names() -> list[tuple[str, str]]:
        return [(key, info['name']) for key, info in AVAILABLE_SCALERS.items()]
    
    @staticmethod
    def get_all_scaler_instances() -> list[tuple[str, object]]:
        result = []
        for key, info in AVAILABLE_SCALERS.items():
            if info['class'] is not None:
                result.append((info['name'], info['class']()))
        return result

def scale_data(X_train, X_test, scaler_key: str = 'none'):
    scaler = ScalerFactory.get_scaler(scaler_key)
    
    if scaler is None:
        return X_train, X_test, None
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler
