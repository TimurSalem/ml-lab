"""
Main Window for ML Lab application
"""
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTabWidget, QMenuBar, QMenu, QToolBar, QStatusBar, QMessageBox,
    QFileDialog, QLabel, QFrame, QScrollArea, QApplication
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QFont, QIcon
import numpy as np
import matplotlib.pyplot as plt

from .widgets.model_selector import ModelSelector
from .widgets.data_loader import DataLoaderWidget
from .widgets.parameters_panel import ParametersPanel
from .widgets.scaling_panel import ScalingPanel
from .widgets.training_panel import TrainingPanel
from .widgets.results_panel import ResultsPanel
from .widgets.visualization_panel import VisualizationPanel

from ..core.data_manager import DataManager
from ..core.model_manager import ModelManager, register_all_models
from ..core.training_manager import TrainingManager, TrainingConfig
from ..core.viz_manager import VizManager
from ..models.base import TrainingResult, ModelCategory
from ..config import CONFIG, COLORS


class MainWindow(QMainWindow):
    """Main application window for ML Lab"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize managers
        self.data_manager = DataManager()
        self.training_manager = TrainingManager()
        self.viz_manager = VizManager()
        
        # Register all models
        register_all_models()
        
        # Current state
        self._current_model = None
        self._current_model_name = None
        
        # Setup UI
        self._init_ui()
        self._connect_signals()
        self._apply_styles()
    
    def _init_ui(self):
        self.setWindowTitle(f"{CONFIG.app_name} v{CONFIG.app_version}")
        self.setMinimumSize(CONFIG.min_window_width, CONFIG.min_window_height)
        self.resize(CONFIG.window_width, CONFIG.window_height)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # === LEFT PANEL ===
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)
        
        # Left tabs: Model Selection, Data Loading
        left_tabs = QTabWidget()
        
        # Model selector
        self.model_selector = ModelSelector()
        left_tabs.addTab(self.model_selector, "Алгоритм")
        
        # Data loader
        self.data_loader = DataLoaderWidget(self.data_manager)
        left_tabs.addTab(self.data_loader, "Данные")
        
        left_layout.addWidget(left_tabs, 5)  # 50% of space
        
        # Parameters panel
        self.parameters_panel = ParametersPanel()
        left_layout.addWidget(self.parameters_panel, 3)  # 30% of space
        
        # Scaling panel (wrapped in scroll area for small screens)
        scaling_scroll = QScrollArea()
        scaling_scroll.setWidgetResizable(True)
        scaling_scroll.setFrameShape(QFrame.Shape.NoFrame)
        scaling_scroll.setMaximumHeight(180)
        self.scaling_panel = ScalingPanel()
        scaling_scroll.setWidget(self.scaling_panel)
        left_layout.addWidget(scaling_scroll, 2)  # 20% of space
        
        left_panel.setMaximumWidth(350)
        left_panel.setMinimumWidth(280)
        splitter.addWidget(left_panel)
        
        # === CENTER PANEL ===
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        
        # Visualization
        self.visualization_panel = VisualizationPanel()
        center_layout.addWidget(self.visualization_panel)
        
        splitter.addWidget(center_panel)
        
        # === RIGHT PANEL ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        
        # Training controls
        self.training_panel = TrainingPanel()
        right_layout.addWidget(self.training_panel)
        
        # Results
        self.results_panel = ResultsPanel()
        right_layout.addWidget(self.results_panel)
        
        right_panel.setMaximumWidth(380)
        right_panel.setMinimumWidth(320)
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([300, 700, 350])
        main_layout.addWidget(splitter)
        
        # === MENU BAR ===
        self._create_menu_bar()
        
        # === STATUS BAR ===
        self.statusBar().showMessage("Готов к работе")
    
    def _create_menu_bar(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("Файл")
        
        load_data_action = QAction("Загрузить данные...", self)
        load_data_action.setShortcut("Ctrl+O")
        file_menu.addAction(load_data_action)
        
        file_menu.addSeparator()
        
        save_model_action = QAction("Сохранить модель...", self)
        save_model_action.setShortcut("Ctrl+S")
        save_model_action.triggered.connect(self._save_model)
        file_menu.addAction(save_model_action)
        
        load_model_action = QAction("Загрузить модель...", self)
        load_model_action.setShortcut("Ctrl+L")
        load_model_action.triggered.connect(self._load_model)
        file_menu.addAction(load_model_action)
        
        file_menu.addSeparator()
        
        export_results_action = QAction("Экспорт результатов...", self)
        export_results_action.triggered.connect(self._export_results)
        file_menu.addAction(export_results_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Выход", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu("Помощь")
        
        about_action = QAction("О программе", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _connect_signals(self):
        # Model selection
        self.model_selector.model_changed.connect(self._on_model_changed)
        
        # Data loading
        self.data_loader.data_configured.connect(self._on_data_configured)
        
        # Training
        self.training_panel.train_clicked.connect(self._on_train_clicked)
        self.training_panel.stop_clicked.connect(self._on_stop_clicked)
        
        # Training manager signals
        self.training_manager.training_started.connect(
            lambda: self.training_panel.set_training_state(True)
        )
        self.training_manager.training_progress.connect(
            self.training_panel.set_progress
        )
        self.training_manager.training_finished.connect(self._on_training_finished)
        self.training_manager.training_error.connect(self._on_training_error)
        self.training_manager.training_log.connect(self.results_panel.add_log)
        
        # Model export/import buttons in visualization panel
        self.visualization_panel._export_model_btn.clicked.connect(self._save_model)
        self.visualization_panel._import_model_btn.clicked.connect(self._load_model)
    
    def _apply_styles(self):
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {COLORS['background']};
                color: {COLORS['text_primary']};
            }}
            QWidget {{
                background-color: {COLORS['background']};
                color: {COLORS['text_primary']};
            }}
            QLabel {{
                color: {COLORS['text_primary']};
            }}
            QGroupBox {{
                font-weight: bold;
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                background-color: {COLORS['surface']};
                color: {COLORS['text_primary']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {COLORS['primary']};
            }}
            QTabWidget::pane {{
                border: 1px solid {COLORS['border']};
                background-color: {COLORS['surface']};
                border-radius: 4px;
            }}
            QTabBar::tab {{
                background-color: {COLORS['background']};
                color: {COLORS['text_secondary']};
                padding: 8px 16px;
                margin-right: 2px;
                border: 1px solid {COLORS['border']};
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background-color: {COLORS['surface']};
                color: {COLORS['text_primary']};
                border-bottom: 2px solid {COLORS['primary']};
            }}
            QScrollArea {{
                border: none;
                background-color: {COLORS['background']};
            }}
            QSplitter::handle {{
                background-color: {COLORS['border']};
                width: 2px;
            }}
            QStatusBar {{
                background-color: {COLORS['surface']};
                color: {COLORS['text_primary']};
                border-top: 1px solid {COLORS['border']};
            }}
            QMenuBar {{
                background-color: {COLORS['surface']};
                color: {COLORS['text_primary']};
            }}
            QMenuBar::item:selected {{
                background-color: {COLORS['primary']};
            }}
            QMenu {{
                background-color: {COLORS['surface']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
            }}
            QMenu::item:selected {{
                background-color: {COLORS['primary']};
            }}
            QComboBox {{
                background-color: {COLORS['surface']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 5px;
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QComboBox QAbstractItemView {{
                background-color: {COLORS['surface']};
                color: {COLORS['text_primary']};
                selection-background-color: {COLORS['primary']};
            }}
            QSpinBox, QDoubleSpinBox {{
                background-color: {COLORS['surface']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 5px;
            }}
            QCheckBox {{
                color: {COLORS['text_primary']};
            }}
            QRadioButton {{
                color: {COLORS['text_primary']};
            }}
            QTableWidget {{
                background-color: {COLORS['surface']};
                color: {COLORS['text_primary']};
                gridline-color: {COLORS['border']};
                border: 1px solid {COLORS['border']};
            }}
            QTableWidget::item {{
                color: {COLORS['text_primary']};
            }}
            QHeaderView::section {{
                background-color: {COLORS['background']};
                color: {COLORS['text_primary']};
                padding: 6px;
                border: none;
                border-bottom: 1px solid {COLORS['border']};
            }}
            QListWidget {{
                background-color: {COLORS['surface']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
            }}
            QListWidget::item {{
                color: {COLORS['text_primary']};
            }}
            QListWidget::item:selected {{
                background-color: {COLORS['primary']};
            }}
            QProgressBar {{
                background-color: {COLORS['surface']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                border-radius: 5px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['success']};
                border-radius: 4px;
            }}
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary_dark']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['border']};
                color: {COLORS['text_secondary']};
            }}
            QTextEdit {{
                background-color: #1a1a2e;
                color: #E0E0E0;
                border: 1px solid {COLORS['border']};
            }}
        """)
    
    def _on_model_changed(self, model_name: str):
        """Handle model selection change"""
        self._current_model_name = model_name
        self._current_model = ModelManager.get_model(model_name)
        self.parameters_panel.set_model(model_name)
        self.results_panel.clear()
        self.visualization_panel.clear_all()
        self.statusBar().showMessage(f"Выбран алгоритм: {self._current_model.display_name}")
    
    def _on_data_configured(self):
        """Handle data configuration completion"""
        self.statusBar().showMessage("Данные готовы к обучению")
    
    def _on_train_clicked(self):
        """Start training"""
        # Validation
        if self._current_model is None:
            QMessageBox.warning(self, "Ошибка", "Выберите алгоритм для обучения")
            return
        
        if self.data_manager.X_train is None:
            QMessageBox.warning(self, "Ошибка", "Загрузите и подготовьте данные")
            return
        
        # Get parameters
        params = self.parameters_panel.get_parameters()
        scaler_key = self.scaling_panel.get_selected_scaler()
        
        # Create training config
        config = TrainingConfig(
            use_cv=self.training_panel.use_cv(),
            cv_folds=self.training_panel.get_cv_folds(),
            use_grid_search=self.training_panel.use_grid_search(),
            scaler_key=scaler_key
        )
        
        # Get param grid for GridSearchCV
        if config.use_grid_search:
            model_class = ModelManager.get_model_class(self._current_model_name)
            if hasattr(model_class, 'get_grid_search_params'):
                config.param_grid = model_class.get_grid_search_params()
        
        # Clear previous results
        self.results_panel.clear_logs()
        self.visualization_panel.clear_all()
        
        # Start training
        self.training_manager.start_training(
            self._current_model,
            self.data_manager.X_train,
            self.data_manager.X_test,
            self.data_manager.y_train,
            self.data_manager.y_test,
            params,
            config,
            self.data_manager.feature_names
        )
    
    def _on_stop_clicked(self):
        """Stop training"""
        self.training_manager.cancel_training()
        self.training_panel.set_training_state(False)
        self.statusBar().showMessage("Обучение отменено")
    
    def _on_training_finished(self, result: TrainingResult):
        """Handle training completion"""
        self.training_panel.set_training_state(False)
        self.results_panel.update_results(result)
        self.statusBar().showMessage(
            f"Обучение завершено! Test score: {result.test_score:.4f}"
        )
        
        # Update visualizations
        self._update_visualizations(result)
    
    def _on_training_error(self, message: str):
        """Handle training error"""
        self.training_panel.set_training_state(False)
        QMessageBox.critical(self, "Ошибка обучения", message)
        self.statusBar().showMessage("Ошибка обучения")
    
    def _update_visualizations(self, result: TrainingResult):
        """Update visualization plots after training based on model category"""
        from sklearn.metrics import confusion_matrix
        import numpy as np
        
        category = self._current_model.category
        
        # For REGRESSION: scatter plot of predictions vs actual
        if category == ModelCategory.REGRESSION:
            if result.predictions is not None:
                fig = self.viz_manager.plot_predictions_vs_actual(
                    self.data_manager.y_test,
                    result.predictions,
                    title="Прогноз vs Реальные значения (Регрессия)"
                )
                if fig:
                    self.visualization_panel.update_plot('predictions', fig)
            
            # Residuals make sense for regression
            if result.predictions is not None:
                fig = self.viz_manager.plot_residuals(
                    self.data_manager.y_test,
                    result.predictions
                )
                if fig:
                    self.visualization_panel.update_plot('residuals', fig)
        
        # For CLASSIFICATION: confusion matrix
        elif category == ModelCategory.CLASSIFICATION:
            if result.predictions is not None:
                y_true = self.data_manager.y_test
                y_pred = result.predictions
                
                # Get unique labels
                unique_labels = np.unique(np.concatenate([y_true, y_pred]))
                n_classes = len(unique_labels)
                
                # Create confusion matrix for all classes
                cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
                
                # Use numeric labels for large class counts
                if n_classes > 15:
                    labels = [str(i) for i in range(n_classes)]
                else:
                    labels = [str(l) for l in unique_labels]
                
                fig = self.viz_manager.plot_confusion_matrix(cm, labels)
                if fig:
                    self.visualization_panel.update_plot('predictions', fig)
                
                # Clear residuals tab (not applicable for classification)
                self.visualization_panel._plots['residuals'].clear()
        
        # For CLUSTERING: cluster scatter plot
        elif category == ModelCategory.CLUSTERING:
            if result.predictions is not None:
                # Get data from model (which combines train+test internally)
                # The model stores the original X_train, we need to combine with X_test
                X_train = self._current_model._X_train
                X_test = self._current_model._X_test if self._current_model._X_test is not None else np.array([]).reshape(0, X_train.shape[1])
                X_full = np.vstack([X_train, X_test]) if X_test.shape[0] > 0 else X_train
                labels = result.predictions
                
                # Verify sizes match
                if len(labels) != X_full.shape[0]:
                    # If mismatch, use data_manager and hope for the best
                    print(f"Warning: labels size {len(labels)} != X_full size {X_full.shape[0]}, using data_manager")
                    X_full = np.vstack([self.data_manager.X_train, self.data_manager.X_test])
                
                # Get cluster centers if available
                centers = None
                if 'cluster_centers' in result.additional_metrics:
                    centers = result.additional_metrics['cluster_centers']
                
                try:
                    fig = self.viz_manager.plot_cluster_scatter(
                        X_full, labels, centers
                    )
                    if fig:
                        self.visualization_panel.update_plot('predictions', fig)
                except Exception as e:
                    print(f"Error plotting clusters: {e}")
                
                # Clear residuals tab (not applicable)
                self.visualization_panel._plots['residuals'].clear()
        
        # Feature importance (works for all if available)
        if result.feature_importances is not None:
            fig = self.viz_manager.plot_feature_importance(self._current_model)
            if fig:
                self.visualization_panel.update_plot('feature_importance', fig)
        
        # Comparison bar chart
        scores = {
            'Train': result.train_score,
            'Test': result.test_score
        }
        if result.cv_mean:
            scores['CV Mean'] = result.cv_mean
        
        # Add info about metric type
        if category == ModelCategory.REGRESSION:
            title = "Сравнение R² метрик"
        elif category == ModelCategory.CLASSIFICATION:
            title = "Сравнение Accuracy"
        else:
            title = "Сравнение Silhouette Score"
        
        fig = self.viz_manager.plot_scores_comparison(scores, title)
        if fig:
            self.visualization_panel.update_plot('comparison', fig)
    
    def _save_model(self):
        """Save trained model to file"""
        if not self._current_model or not self._current_model.is_fitted:
            QMessageBox.warning(self, "Ошибка", "Нет обученной модели для сохранения")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Сохранить модель",
            f"{self._current_model_name}.pkl",
            "Pickle Files (*.pkl *.sav)"
        )
        
        if filepath:
            try:
                self._current_model.save(filepath)
                QMessageBox.information(self, "Успех", f"Модель сохранена: {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить: {str(e)}")
    
    def _load_model(self):
        """Load model from file"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Загрузить модель",
            "",
            "Pickle Files (*.pkl *.sav)"
        )
        
        if filepath:
            if not self._current_model:
                QMessageBox.warning(self, "Ошибка", "Сначала выберите тип модели")
                return
            
            try:
                self._current_model.load(filepath)
                QMessageBox.information(self, "Успех", f"Модель загружена: {filepath}")
                self.statusBar().showMessage(f"Модель загружена из {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить: {str(e)}")
    
    def _export_results(self):
        """Export results to file"""
        if not self._current_model or not self._current_model.get_last_result():
            QMessageBox.warning(self, "Ошибка", "Нет результатов для экспорта")
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Экспорт результатов",
            "results.csv",
            "CSV Files (*.csv);;Excel Files (*.xlsx)"
        )
        
        if filepath:
            try:
                result = self._current_model.get_last_result()
                import pandas as pd
                
                data = {
                    'Метрика': ['Train Score', 'Test Score', 'Training Time'],
                    'Значение': [result.train_score, result.test_score, result.training_time]
                }
                
                if result.cv_mean:
                    data['Метрика'].extend(['CV Mean', 'CV Std'])
                    data['Значение'].extend([result.cv_mean, result.cv_std])
                
                df = pd.DataFrame(data)
                
                if filepath.endswith('.xlsx'):
                    df.to_excel(filepath, index=False)
                else:
                    df.to_csv(filepath, index=False)
                
                QMessageBox.information(self, "Успех", f"Результаты экспортированы: {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось экспортировать: {str(e)}")
    
    def _show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            f"О программе {CONFIG.app_name}",
            f"""<h2>{CONFIG.app_name}</h2>
            <p>Версия: {CONFIG.app_version}</p>
            <p>Интерфейс для экспериментов с машинным обучением.</p>
            <p>Поддерживаемые алгоритмы:</p>
            <ul>
                <li>Регрессия: GradientBoosting, RandomForest, KNN, Lasso, Ridge</li>
                <li>Классификация: KNN, Logistic Regression</li>
                <li>Кластеризация: KMeans, DBSCAN, Agglomerative</li>
            </ul>
            <p>Разработано для обучения студентов.</p>
            """
        )
