"""
Parameters Panel widget for ML Lab
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
    QDoubleSpinBox, QComboBox, QCheckBox, QGroupBox,
    QScrollArea, QFrame, QSlider, QGridLayout
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont

from ...models.base import ParameterConfig, ParameterType
from ...core.model_manager import ModelManager


class ParameterWidget(QWidget):
    """Single parameter input widget"""
    
    value_changed = pyqtSignal()
    
    def __init__(self, config: ParameterConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._init_ui()
    
    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 5, 0, 5)
        
        # Label
        label = QLabel(f"{self.config.display_name}:")
        label.setMinimumWidth(140)
        label.setToolTip(self.config.description)
        layout.addWidget(label)
        
        # Input widget based on type
        if self.config.param_type == ParameterType.INT:
            self._input = QSpinBox()
            self._input.setRange(
                int(self.config.min_value or 1),
                int(self.config.max_value or 1000)
            )
            self._input.setSingleStep(int(self.config.step or 1))
            self._input.setValue(int(self.config.default))
            self._input.valueChanged.connect(self.value_changed.emit)
            
        elif self.config.param_type == ParameterType.FLOAT:
            self._input = QDoubleSpinBox()
            self._input.setRange(
                self.config.min_value or 0.0,
                self.config.max_value or 100.0
            )
            self._input.setSingleStep(self.config.step or 0.1)
            self._input.setDecimals(4)
            self._input.setValue(float(self.config.default))
            self._input.valueChanged.connect(self.value_changed.emit)
            
        elif self.config.param_type == ParameterType.BOOL:
            self._input = QCheckBox()
            self._input.setChecked(bool(self.config.default))
            self._input.stateChanged.connect(self.value_changed.emit)
            
        elif self.config.param_type == ParameterType.CHOICE:
            self._input = QComboBox()
            labels = self.config.choice_labels or self.config.choices
            for i, choice in enumerate(self.config.choices):
                label_text = labels[i] if i < len(labels) else str(choice)
                self._input.addItem(label_text, choice)
            
            # Set default
            default_idx = 0
            if self.config.default in self.config.choices:
                default_idx = self.config.choices.index(self.config.default)
            self._input.setCurrentIndex(default_idx)
            self._input.currentIndexChanged.connect(self.value_changed.emit)
        else:
            self._input = QLabel(str(self.config.default))
        
        layout.addWidget(self._input, 1)
    
    def get_value(self):
        """Get current parameter value"""
        if self.config.param_type == ParameterType.INT:
            return self._input.value()
        elif self.config.param_type == ParameterType.FLOAT:
            return self._input.value()
        elif self.config.param_type == ParameterType.BOOL:
            return self._input.isChecked()
        elif self.config.param_type == ParameterType.CHOICE:
            return self._input.currentData()
        return self.config.default
    
    def set_value(self, value):
        """Set parameter value"""
        if self.config.param_type == ParameterType.INT:
            self._input.setValue(int(value))
        elif self.config.param_type == ParameterType.FLOAT:
            self._input.setValue(float(value))
        elif self.config.param_type == ParameterType.BOOL:
            self._input.setChecked(bool(value))
        elif self.config.param_type == ParameterType.CHOICE:
            idx = self._input.findData(value)
            if idx >= 0:
                self._input.setCurrentIndex(idx)
    
    def reset(self):
        """Reset to default value"""
        self.set_value(self.config.default)


class ParametersPanel(QWidget):
    """Panel for configuring model parameters"""
    
    parameters_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_model_name = None
        self._param_widgets = {}
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("Параметры модели")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #E0E0E0;")
        layout.addWidget(title)
        
        # Model name label
        self._model_label = QLabel("Модель не выбрана")
        self._model_label.setStyleSheet("color: #BDBDBD; font-style: italic;")
        layout.addWidget(self._model_label)
        
        # Scroll area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        self._params_container = QWidget()
        self._params_layout = QVBoxLayout(self._params_container)
        self._params_layout.setSpacing(0)
        scroll.setWidget(self._params_container)
        layout.addWidget(scroll)
        
        # Reset button
        reset_btn = QPushButton("↻ Сбросить к значениям по умолчанию")
        reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #373737;
                border: 1px solid #424242;
                padding: 8px;
                border-radius: 4px;
                color: #E0E0E0;
            }
            QPushButton:hover {
                background-color: #424242;
            }
        """)
        reset_btn.clicked.connect(self._reset_all)
        layout.addWidget(reset_btn)
    
    def set_model(self, model_name: str):
        """Update parameters for the selected model"""
        self._current_model_name = model_name
        
        # Clear existing widgets
        self._param_widgets.clear()
        while self._params_layout.count():
            child = self._params_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        if not model_name:
            self._model_label.setText("Модель не выбрана")
            return
        
        model_class = ModelManager.get_model_class(model_name)
        self._model_label.setText(f"📊 {model_class.display_name}")
        self._model_label.setStyleSheet("color: #E0E0E0; font-weight: bold;")
        
        # Add parameter widgets
        params = model_class.get_parameters()
        for config in params:
            widget = ParameterWidget(config)
            widget.value_changed.connect(self.parameters_changed.emit)
            self._param_widgets[config.name] = widget
            self._params_layout.addWidget(widget)
        
        self._params_layout.addStretch()
    
    def get_parameters(self) -> dict:
        """Get all current parameter values"""
        return {name: widget.get_value() 
                for name, widget in self._param_widgets.items()}
    
    def set_parameters(self, params: dict):
        """Set parameter values"""
        for name, value in params.items():
            if name in self._param_widgets:
                self._param_widgets[name].set_value(value)
    
    def _reset_all(self):
        """Reset all parameters to defaults"""
        for widget in self._param_widgets.values():
            widget.reset()
        self.parameters_changed.emit()


from PyQt6.QtWidgets import QPushButton
