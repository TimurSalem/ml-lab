"""
Scaling Panel widget for ML Lab
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QGroupBox, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont

from ...utils.scalers import AVAILABLE_SCALERS


class ScalingPanel(QWidget):
    """Widget for selecting data scaling method"""
    
    scaler_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_scaler = 'none'
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        title = QLabel("Масштабирование")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #E0E0E0;")
        layout.addWidget(title)
        
        # Button group for scalers
        self._button_group = QButtonGroup(self)
        self._button_group.buttonClicked.connect(self._on_scaler_selected)
        
        self._scaler_buttons = {}
        
        for key, info in AVAILABLE_SCALERS.items():
            radio = QRadioButton(info['name'])
            radio.setToolTip(info['description'])
            radio.setStyleSheet("""
                QRadioButton {
                    font-size: 11px;
                    padding: 6px;
                    color: #E0E0E0;
                }
                QRadioButton:hover {
                    background-color: #424242;
                    border-radius: 4px;
                }
            """)
            
            if key == 'none':
                radio.setChecked(True)
            
            self._button_group.addButton(radio)
            self._scaler_buttons[radio] = key
            layout.addWidget(radio)
        
        # Description
        self._description_label = QLabel(AVAILABLE_SCALERS['none']['description'])
        self._description_label.setWordWrap(True)
        self._description_label.setStyleSheet("""
            QLabel {
                background-color: #373737;
                color: #BDBDBD;
                padding: 10px;
                border-radius: 5px;
                font-size: 11px;
            }
        """)
        layout.addWidget(self._description_label)
        
        layout.addStretch()
    
    def _on_scaler_selected(self, button):
        scaler_key = self._scaler_buttons.get(button)
        if scaler_key:
            self._current_scaler = scaler_key
            self._description_label.setText(AVAILABLE_SCALERS[scaler_key]['description'])
            self.scaler_changed.emit(scaler_key)
    
    def get_selected_scaler(self) -> str:
        """Get currently selected scaler key"""
        return self._current_scaler
    
    def set_scaler(self, scaler_key: str):
        """Programmatically select a scaler"""
        for button, key in self._scaler_buttons.items():
            if key == scaler_key:
                button.setChecked(True)
                self._on_scaler_selected(button)
                break
