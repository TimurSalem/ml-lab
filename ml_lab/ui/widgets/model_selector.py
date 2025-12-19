"""
Model Selector widget for ML Lab
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QGroupBox, QRadioButton, QButtonGroup, QScrollArea, QFrame
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont

from ...models.base import ModelCategory
from ...core.model_manager import ModelManager


class ModelSelector(QWidget):
    """Widget for selecting ML model"""
    
    model_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_model = None
        self._model_buttons = {}
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("Выбор алгоритма")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #E0E0E0;")
        layout.addWidget(title)
        
        # Scroll area for model categories
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(15)
        
        # Create button group
        self._button_group = QButtonGroup(self)
        self._button_group.buttonClicked.connect(self._on_model_selected)
        
        # Get models by category
        models_by_category = ModelManager.list_models_by_category()
        
        category_info = {
            ModelCategory.REGRESSION: ("Регрессия", "#555555"),
            ModelCategory.CLASSIFICATION: ("Классификация", "#555555"),
            ModelCategory.CLUSTERING: ("Кластеризация", "#555555")
        }
        
        for category, models in models_by_category.items():
            if not models:
                continue
            
            cat_name, cat_color = category_info.get(category, ("Другое", "#555555"))
            
            # Category group
            group = QGroupBox(cat_name)
            group.setStyleSheet(f"""
                QGroupBox {{
                    font-weight: bold;
                    font-size: 12px;
                    border: 1px solid {cat_color};
                    border-radius: 8px;
                    margin-top: 12px;
                    padding-top: 10px;
                    background-color: #2D2D2D;
                    color: #E0E0E0;
                }}
                QGroupBox::title {{
                    subcontrol-origin: margin;
                    left: 15px;
                    padding: 0 8px;
                    color: #BDBDBD;
                }}
            """)
            
            group_layout = QVBoxLayout(group)
            group_layout.setSpacing(8)
            
            for model_name in models:
                model_class = ModelManager.get_model_class(model_name)
                
                radio = QRadioButton(model_class.display_name)
                radio.setToolTip(model_class.description)
                radio.setStyleSheet("""
                    QRadioButton {
                        font-size: 11px;
                        padding: 5px;
                        color: #E0E0E0;
                    }
                    QRadioButton:hover {
                        background-color: #424242;
                        border-radius: 4px;
                    }
                """)
                
                self._button_group.addButton(radio)
                self._model_buttons[radio] = model_name
                group_layout.addWidget(radio)
            
            scroll_layout.addWidget(group)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        self._description_label = QLabel()
        self._description_label.setWordWrap(True)
        self._description_label.setStyleSheet("""
            QLabel {
                background-color: #373737;
                padding: 10px;
                border-radius: 5px;
                color: #BDBDBD;
                font-size: 11px;
            }
        """)
        layout.addWidget(self._description_label)
    
    def _on_model_selected(self, button):
        model_name = self._model_buttons.get(button)
        if model_name:
            self._current_model = model_name
            model_class = ModelManager.get_model_class(model_name)
            self._description_label.setText(model_class.description)
            self.model_changed.emit(model_name)
    
    def get_selected_model(self) -> str:
        """Get currently selected model name"""
        return self._current_model
    
    def select_model(self, model_name: str):
        """Programmatically select a model"""
        for button, name in self._model_buttons.items():
            if name == model_name:
                button.setChecked(True)
                self._on_model_selected(button)
                break
