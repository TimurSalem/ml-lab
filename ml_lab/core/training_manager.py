from typing import Optional, Callable
from dataclasses import dataclass
from PyQt6.QtCore import QObject, QThread, pyqtSignal
import numpy as np
import time

from ..models.base import BaseModel, TrainingResult
from ..utils.scalers import scale_data

@dataclass
class TrainingConfig:
    use_cv: bool = False
    cv_folds: int = 5
    use_grid_search: bool = False
    param_grid: Optional[dict] = None
    scaler_key: str = 'none'

class TrainingWorker(QObject):
    
    progress = pyqtSignal(int, str)  # progress percentage, message
    finished = pyqtSignal(object)  # TrainingResult
    error = pyqtSignal(str)  # error message
    log = pyqtSignal(str)  # log message
    
    def __init__(self, model: BaseModel, X_train: np.ndarray, X_test: np.ndarray,
                 y_train: np.ndarray, y_test: np.ndarray, params: dict,
                 config: TrainingConfig, feature_names: list = None):
        super().__init__()
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.params = params
        self.config = config
        self.feature_names = feature_names
        self._is_cancelled = False
    
    def cancel(self):
        self._is_cancelled = True
    
    def run(self):
        try:
            self.progress.emit(0, "Подготовка данных...")
            self.log.emit(f"Начало обучения модели: {self.model.display_name}")
            
            self.log.emit("--- ОТЛАДОЧНАЯ ИНФОРМАЦИЯ ---")
            self.log.emit(f"X_train shape: {self.X_train.shape}")
            self.log.emit(f"X_test shape: {self.X_test.shape}")
            self.log.emit(f"y_train: min={self.y_train.min():.4f}, max={self.y_train.max():.4f}, mean={self.y_train.mean():.4f}")
            self.log.emit(f"y_test: min={self.y_test.min():.4f}, max={self.y_test.max():.4f}, mean={self.y_test.mean():.4f}")
            self.log.emit(f"Параметры модели: {self.params}")
            if self.feature_names:
                self.log.emit(f"Признаки ({len(self.feature_names)}): {self.feature_names[:5]}{'...' if len(self.feature_names) > 5 else ''}")
            self.log.emit("--- КОНЕЦ ОТЛАДКИ ---")
            
            X_train, X_test = self.X_train, self.X_test
            if self.config.scaler_key != 'none':
                self.log.emit(f"Масштабирование данных: {self.config.scaler_key}")
                X_train, X_test, scaler = scale_data(
                    self.X_train, self.X_test, self.config.scaler_key
                )
                self.model._scaler = scaler
                self.log.emit(f"После масштабирования: X_train mean={X_train.mean():.4f}, std={X_train.std():.4f}")
            
            self.progress.emit(20, "Обучение модели...")
            
            if self._is_cancelled:
                self.error.emit("Обучение отменено")
                return
            
            if self.config.use_grid_search and self.config.param_grid:
                self.log.emit(f"Используется GridSearchCV с {self.config.cv_folds} фолдами")
                self.log.emit(f"Сетка параметров: {self.config.param_grid}")
                self.progress.emit(30, "GridSearchCV...")
                
                result = self.model.train_with_grid_search(
                    X_train, X_test,
                    self.y_train, self.y_test,
                    self.config.param_grid,
                    cv=self.config.cv_folds,
                    feature_names=self.feature_names
                )
                
            elif self.config.use_cv:
                self.log.emit(f"Используется кросс-валидация с {self.config.cv_folds} фолдами")
                self.progress.emit(30, f"Кросс-валидация ({self.config.cv_folds} фолдов)...")
                
                result = self.model.train_with_cv(
                    X_train, X_test,
                    self.y_train, self.y_test,
                    self.params,
                    cv=self.config.cv_folds,
                    feature_names=self.feature_names
                )
                
            else:
                self.log.emit("Стандартное обучение")
                result = self.model.train(
                    X_train, X_test,
                    self.y_train, self.y_test,
                    self.params,
                    feature_names=self.feature_names
                )
            
            self.progress.emit(90, "Завершение...")
            
            self.log.emit(f"Обучение завершено за {result.training_time:.2f} сек")
            self.log.emit(f"Правильность на обучающем наборе: {result.train_score:.4f}")
            self.log.emit(f"Правильность на тестовом наборе: {result.test_score:.4f}")
            
            if result.cv_mean is not None and result.cv_std is not None:
                self.log.emit(f"Средняя CV правильность: {result.cv_mean:.4f} ± {result.cv_std:.4f}")
            elif result.cv_mean is not None:
                self.log.emit(f"Лучший CV score: {result.cv_mean:.4f}")
            
            if result.best_params:
                self.log.emit(f"Лучшие параметры: {result.best_params}")
            
            self.progress.emit(100, "Готово!")
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))

class TrainingManager(QObject):
    
    training_started = pyqtSignal()
    training_progress = pyqtSignal(int, str)
    training_finished = pyqtSignal(object)
    training_error = pyqtSignal(str)
    training_log = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self._thread: Optional[QThread] = None
        self._worker: Optional[TrainingWorker] = None
        self._is_training = False
    
    @property
    def is_training(self) -> bool:
        return self._is_training
    
    def start_training(self, model: BaseModel,
                       X_train: np.ndarray, X_test: np.ndarray,
                       y_train: np.ndarray, y_test: np.ndarray,
                       params: dict, config: TrainingConfig,
                       feature_names: list = None):
        if self._is_training:
            raise RuntimeError("Training already in progress")
        
        self._is_training = True
        self.training_started.emit()
        
        self._thread = QThread()
        self._worker = TrainingWorker(
            model, X_train, X_test, y_train, y_test,
            params, config, feature_names
        )
        self._worker.moveToThread(self._thread)
        
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.log.connect(self._on_log)
        
        self._worker.finished.connect(self._cleanup)
        self._worker.error.connect(self._cleanup)
        
        self._thread.start()
    
    def cancel_training(self):
        if self._worker:
            self._worker.cancel()
        self._cleanup()
    
    def _on_progress(self, value: int, message: str):
        self.training_progress.emit(value, message)
    
    def _on_finished(self, result: TrainingResult):
        self._is_training = False
        self.training_finished.emit(result)
    
    def _on_error(self, message: str):
        self._is_training = False
        self.training_error.emit(message)
    
    def _on_log(self, message: str):
        self.training_log.emit(message)
    
    def _cleanup(self):
        self._is_training = False
        if self._thread:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None
