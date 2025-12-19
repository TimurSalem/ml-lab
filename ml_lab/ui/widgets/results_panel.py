"""
Results Panel widget for ML Lab
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QGroupBox, QTableWidget, QTableWidgetItem, QTabWidget,
    QHeaderView
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont, QColor

from ...models.base import TrainingResult


class ResultsPanel(QWidget):
    """Widget for displaying training results and logs"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("Результаты")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #E0E0E0;")
        layout.addWidget(title)
        
        # Tabs for metrics and logs
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #424242;
                border-radius: 4px;
                background: #2D2D2D;
            }
            QTabBar::tab {
                padding: 8px 16px;
                margin-right: 2px;
                background: #1E1E1E;
                border: 1px solid #424242;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                color: #BDBDBD;
            }
            QTabBar::tab:selected {
                background: #2D2D2D;
                border-bottom: 2px solid #555555;
                color: #E0E0E0;
            }
        """)
        
        # Metrics tab
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(metrics_widget)
        
        # Main scores
        self._scores_group = QGroupBox("Метрики качества")
        scores_layout = QVBoxLayout(self._scores_group)
        
        self._metrics_table = QTableWidget()
        self._metrics_table.setColumnCount(2)
        self._metrics_table.setHorizontalHeaderLabels(["Метрика", "Значение"])
        self._metrics_table.horizontalHeader().setStretchLastSection(True)
        self._metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._metrics_table.setMaximumHeight(200)
        self._metrics_table.setStyleSheet("""
            QTableWidget {
                border: none;
                font-size: 11px;
                background-color: #2D2D2D;
                color: #E0E0E0;
            }
            QHeaderView::section {
                background-color: #1E1E1E;
                padding: 6px;
                border: none;
                border-bottom: 1px solid #424242;
                font-weight: bold;
                color: #E0E0E0;
            }
        """)
        scores_layout.addWidget(self._metrics_table)
        metrics_layout.addWidget(self._scores_group)
        
        # Best parameters (for GridSearchCV)
        self._params_group = QGroupBox("Лучшие параметры (GridSearchCV)")
        self._params_group.setVisible(False)
        params_layout = QVBoxLayout(self._params_group)
        
        self._params_table = QTableWidget()
        self._params_table.setColumnCount(2)
        self._params_table.setHorizontalHeaderLabels(["Параметр", "Значение"])
        self._params_table.horizontalHeader().setStretchLastSection(True)
        self._params_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._params_table.setMaximumHeight(150)
        params_layout.addWidget(self._params_table)
        metrics_layout.addWidget(self._params_group)
        
        metrics_layout.addStretch()
        tabs.addTab(metrics_widget, "📊 Метрики")
        
        # Logs tab
        logs_widget = QWidget()
        logs_layout = QVBoxLayout(logs_widget)
        
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                background-color: #263238;
                color: #ECEFF1;
                border: none;
                padding: 10px;
            }
        """)
        logs_layout.addWidget(self._log_text)
        
        # Clear button
        clear_btn = QPushButton("Очистить логи")
        clear_btn.clicked.connect(self._log_text.clear)
        logs_layout.addWidget(clear_btn)
        
        tabs.addTab(logs_widget, "📝 Логи")
        
        layout.addWidget(tabs)
    
    def update_results(self, result: TrainingResult):
        """Update display with training results"""
        # Clear and populate metrics table
        self._metrics_table.setRowCount(0)
        
        metrics = [
            ("Правильность (train)", f"{result.train_score:.4f}"),
            ("Правильность (test)", f"{result.test_score:.4f}"),
            ("Время обучения", f"{result.training_time:.2f} сек"),
        ]
        
        if result.cv_mean is not None and result.cv_std is not None:
            metrics.append(("CV Mean ± Std", f"{result.cv_mean:.4f} ± {result.cv_std:.4f}"))
        elif result.cv_mean is not None:
            metrics.append(("Лучший CV score", f"{result.cv_mean:.4f}"))
        
        # Add additional metrics
        for key, value in result.additional_metrics.items():
            if isinstance(value, (int, float)):
                metrics.append((key, f"{value:.4f}" if isinstance(value, float) else str(value)))
        
        self._metrics_table.setRowCount(len(metrics))
        for i, (name, value) in enumerate(metrics):
            name_item = QTableWidgetItem(name)
            value_item = QTableWidgetItem(value)
            
            # Color code test score
            if "test" in name.lower():
                score = result.test_score
                if score >= 0.9:
                    value_item.setBackground(QColor("#1B5E20"))
                elif score >= 0.7:
                    value_item.setBackground(QColor("#5D4037"))
                else:
                    value_item.setBackground(QColor("#5D0000"))
            
            self._metrics_table.setItem(i, 0, name_item)
            self._metrics_table.setItem(i, 1, value_item)
        
        # Update best parameters if available
        if result.best_params:
            self._params_group.setVisible(True)
            self._params_table.setRowCount(len(result.best_params))
            
            for i, (key, value) in enumerate(result.best_params.items()):
                self._params_table.setItem(i, 0, QTableWidgetItem(key))
                self._params_table.setItem(i, 1, QTableWidgetItem(str(value)))
        else:
            self._params_group.setVisible(False)
    
    def add_log(self, message: str):
        """Add a log message"""
        self._log_text.append(message)
        # Scroll to bottom
        scrollbar = self._log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear_logs(self):
        """Clear all logs"""
        self._log_text.clear()
    
    def clear(self):
        """Clear all results"""
        self._metrics_table.setRowCount(0)
        self._params_table.setRowCount(0)
        self._params_group.setVisible(False)
        self._log_text.clear()


from PyQt6.QtWidgets import QPushButton
