"""
Training Panel widget for ML Lab
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QCheckBox, QSpinBox, QGroupBox, QFrame
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont


class TrainingPanel(QWidget):
    """Widget for training controls"""
    
    train_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_training = False
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("Обучение")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #E0E0E0;")
        layout.addWidget(title)
        
        # Options group
        options_group = QGroupBox("Опции")
        options_layout = QVBoxLayout(options_group)
        
        # Cross-validation
        cv_layout = QHBoxLayout()
        self._cv_check = QCheckBox("Кросс-валидация")
        self._cv_check.setChecked(True)
        self._cv_check.stateChanged.connect(self._on_cv_changed)
        cv_layout.addWidget(self._cv_check)
        
        cv_layout.addWidget(QLabel("Фолдов:"))
        self._cv_folds_spin = QSpinBox()
        self._cv_folds_spin.setRange(2, 20)
        self._cv_folds_spin.setValue(5)
        cv_layout.addWidget(self._cv_folds_spin)
        cv_layout.addStretch()
        options_layout.addLayout(cv_layout)
        
        # GridSearchCV
        grid_layout = QHBoxLayout()
        self._grid_search_check = QCheckBox("GridSearchCV (автоподбор параметров)")
        self._grid_search_check.setToolTip("Автоматический подбор лучших параметров")
        grid_layout.addWidget(self._grid_search_check)
        grid_layout.addStretch()
        options_layout.addLayout(grid_layout)
        
        layout.addWidget(options_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self._train_btn = QPushButton("Обучить модель")
        self._train_btn.setMinimumHeight(45)
        self._train_btn.setStyleSheet("""
            QPushButton {
                background-color: #81C784;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #66BB6A;
            }
            QPushButton:pressed {
                background-color: #4CAF50;
            }
            QPushButton:disabled {
                background-color: #424242;
                color: #757575;
            }
        """)
        self._train_btn.clicked.connect(self._on_train_clicked)
        btn_layout.addWidget(self._train_btn)
        
        self._stop_btn = QPushButton("Остановить")
        self._stop_btn.setMinimumHeight(45)
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #555555;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
            QPushButton:disabled {
                background-color: #424242;
                color: #757575;
            }
        """)
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        btn_layout.addWidget(self._stop_btn)
        
        layout.addLayout(btn_layout)
        
        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimumHeight(25)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setFormat("%p% - Готов к обучению")
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #424242;
                border-radius: 5px;
                background-color: #2D2D2D;
                text-align: center;
                color: #E0E0E0;
            }
            QProgressBar::chunk {
                background-color: #81C784;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self._progress_bar)
        
        # Status label
        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("""
            QLabel {
                color: #BDBDBD;
                font-size: 11px;
            }
        """)
        layout.addWidget(self._status_label)
        
        layout.addStretch()
    
    def _on_cv_changed(self, state):
        self._cv_folds_spin.setEnabled(state == Qt.CheckState.Checked.value)
    
    def _on_train_clicked(self):
        self.train_clicked.emit()
    
    def _on_stop_clicked(self):
        self.stop_clicked.emit()
    
    def set_training_state(self, is_training: bool):
        """Update UI for training state"""
        self._is_training = is_training
        self._train_btn.setEnabled(not is_training)
        self._stop_btn.setEnabled(is_training)
        self._cv_check.setEnabled(not is_training)
        self._cv_folds_spin.setEnabled(not is_training and self._cv_check.isChecked())
        self._grid_search_check.setEnabled(not is_training)
        
        if is_training:
            self._progress_bar.setFormat("%p% - Обучение...")
        else:
            self._progress_bar.setFormat("%p% - Готов")
    
    def set_progress(self, value: int, message: str = ""):
        """Update progress bar"""
        self._progress_bar.setValue(value)
        if message:
            self._progress_bar.setFormat(f"%p% - {message}")
    
    def set_status(self, status: str):
        """Set status message"""
        self._status_label.setText(status)
    
    def use_cv(self) -> bool:
        return self._cv_check.isChecked()
    
    def get_cv_folds(self) -> int:
        return self._cv_folds_spin.value()
    
    def use_grid_search(self) -> bool:
        return self._grid_search_check.isChecked()
    
    def reset(self):
        """Reset to initial state"""
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("%p% - Готов к обучению")
        self._status_label.clear()
        self.set_training_state(False)
