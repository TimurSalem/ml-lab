from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class AppConfig:
    app_name: str = "ML Lab"
    app_version: str = "1.2.0"
    window_width: int = 1400
    window_height: int = 900
    min_window_width: int = 1200
    min_window_height: int = 700
    
    default_test_size: float = 0.25
    default_random_state: int = 42
    default_cv_folds: int = 5
    
    models_dir: str = "saved_models"
    exports_dir: str = "exports"
    
    def __post_init__(self):
        for dir_path in [self.models_dir, self.exports_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)

CONFIG = AppConfig()

COLORS = {
    'primary': '#90A4AE',        # сине-серый для большей части UI
    'primary_dark': '#78909C',
    'secondary': '#78909C',      # сине-серый
    'accent': '#FFB74D',         # желтый
    'success': '#81C784',        # зеленый
    'error': '#E57373',          # красный
    'warning': '#FFB74D',        # желтый
    'background': '#1E1E1E',
    'surface': '#2D2D2D',
    'text_primary': '#E0E0E0',
    'text_secondary': '#BDBDBD',
    'border': '#424242',
    'card': '#373737',
}

MODEL_CATEGORIES = {
    'regression': {
        'name': 'Регрессия',
        'icon': '',
        'description': 'Модели для предсказания непрерывных значений'
    },
    'classification': {
        'name': 'Классификация', 
        'icon': '',
        'description': 'Модели для предсказания категорий'
    },
    'clustering': {
        'name': 'Кластеризация',
        'icon': '',
        'description': 'Модели для группировки данных'
    }
}
