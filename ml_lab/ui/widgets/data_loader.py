"""
Data Loader widget for ML Lab"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QTableWidget, QTableWidgetItem, QComboBox,
    QGroupBox, QListWidget, QListWidgetItem, QAbstractItemView,
    QSpinBox, QDoubleSpinBox, QCheckBox, QMessageBox, QSplitter,
    QScrollArea
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont

from ...core.data_manager import DataManager


class DataLoaderWidget(QWidget):
    """Widget for loading and configuring data"""
    
    data_loaded = pyqtSignal(object)  # Emits DataInfo when data is loaded
    data_configured = pyqtSignal()  # Emits when target/features are configured
    
    def __init__(self, data_manager: DataManager, parent=None):
        super().__init__(parent)
        self.data_manager = data_manager
        self._init_ui()
    
    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create scroll area to contain all content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #2D2D2D;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background-color: #555555;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #666666;
            }
        """)
        
        # Content widget inside scroll area
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title = QLabel("Данные")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #E0E0E0;")
        layout.addWidget(title)
        
        # Load button
        load_btn = QPushButton("Загрузить файл (Excel/CSV)")
        load_btn.setStyleSheet("""
            QPushButton {
                background-color: #424242;
                color: white;
                border: 1px solid #555555;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #555555;
                border-color: #FFB74D;
            }
            QPushButton:pressed {
                background-color: #333333;
            }
        """)
        load_btn.clicked.connect(self._load_file)
        layout.addWidget(load_btn)
        
        # File info label
        self._file_label = QLabel("Файл не загружен")
        self._file_label.setStyleSheet("color: #BDBDBD; font-style: italic;")
        layout.addWidget(self._file_label)
        
        self._preview_group = QGroupBox("Предпросмотр данных")
        self._preview_group.setVisible(False)
        preview_layout = QVBoxLayout(self._preview_group)
        
        self._preview_table = QTableWidget()
        self._preview_table.setMaximumHeight(150)
        self._preview_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #424242;
                border-radius: 4px;
                font-size: 10px;
                background-color: #2D2D2D;
                color: #E0E0E0;
            }
            QHeaderView::section {
                background-color: #1E1E1E;
                padding: 5px;
                border: none;
                border-bottom: 1px solid #424242;
                font-weight: bold;
                color: #E0E0E0;
            }
        """)
        preview_layout.addWidget(self._preview_table)
        layout.addWidget(self._preview_group)
        
        # Missing values warning
        self._missing_group = QGroupBox("Пропущенные значения")
        self._missing_group.setVisible(False)
        self._missing_group.setStyleSheet("""
            QGroupBox {
                background-color: #5D4037;
                border: 2px solid #FF8A65;
                border-radius: 6px;
                color: #FFCCBC;
            }
            QGroupBox::title {
                color: #FF8A65;
            }
        """)
        missing_layout = QVBoxLayout(self._missing_group)
        
        self._missing_label = QLabel()
        self._missing_label.setWordWrap(True)
        self._missing_label.setStyleSheet("color: #FFCCBC;")
        missing_layout.addWidget(self._missing_label)
        
        # Strategy selector
        strategy_layout = QHBoxLayout()
        strategy_layout.addWidget(QLabel("Обработка:"))
        self._missing_combo = QComboBox()
        self._missing_combo.addItem("Удалить строки с пропусками", "drop")
        self._missing_combo.addItem("Заполнить средним", "mean")
        self._missing_combo.addItem("Заполнить медианой", "median")
        self._missing_combo.addItem("Заполнить модой", "mode")
        strategy_layout.addWidget(self._missing_combo)
        
        fix_missing_btn = QPushButton("Исправить")
        fix_missing_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF8A65;
                color: #3E2723;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #FFAB91;
            }
        """)
        fix_missing_btn.clicked.connect(self._handle_missing)
        strategy_layout.addWidget(fix_missing_btn)
        strategy_layout.addStretch()
        missing_layout.addLayout(strategy_layout)
        
        layout.addWidget(self._missing_group)
        
        # Configuration
        self._config_group = QGroupBox("Конфигурация")
        self._config_group.setVisible(False)
        config_layout = QVBoxLayout(self._config_group)
        
        # Target column
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("Целевая переменная (y):"))
        self._target_combo = QComboBox()
        self._target_combo.setMinimumWidth(150)
        self._target_combo.currentTextChanged.connect(self._on_target_changed)
        target_layout.addWidget(self._target_combo)
        target_layout.addStretch()
        config_layout.addLayout(target_layout)
        
        # Feature columns
        config_layout.addWidget(QLabel("Признаки (X):"))
        self._features_list = QListWidget()
        self._features_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self._features_list.setMaximumHeight(120)
        self._features_list.itemSelectionChanged.connect(self._on_features_changed)
        config_layout.addWidget(self._features_list)
        
        btn_layout = QHBoxLayout()
        select_all_btn = QPushButton("Выбрать все")
        select_all_btn.clicked.connect(self._select_all_features)
        select_none_btn = QPushButton("Снять все")
        select_none_btn.clicked.connect(self._deselect_all_features)
        btn_layout.addWidget(select_all_btn)
        btn_layout.addWidget(select_none_btn)
        btn_layout.addStretch()
        config_layout.addLayout(btn_layout)
        
        # Split settings 
        split_layout = QHBoxLayout()
        split_layout.addWidget(QLabel("Тестовая выборка:"))
        self._test_size_spin = QDoubleSpinBox()
        self._test_size_spin.setRange(0.1, 0.5)
        self._test_size_spin.setSingleStep(0.05)
        self._test_size_spin.setValue(0.25)
        self._test_size_spin.setSuffix(" (25%)")
        self._test_size_spin.valueChanged.connect(
            lambda v: self._test_size_spin.setSuffix(f" ({int(v*100)}%)")
        )
        split_layout.addWidget(self._test_size_spin)
        
        split_layout.addWidget(QLabel("Random state:"))
        self._random_state_spin = QSpinBox()
        self._random_state_spin.setRange(0, 9999)
        self._random_state_spin.setValue(42)
        split_layout.addWidget(self._random_state_spin)
        split_layout.addStretch()
        config_layout.addLayout(split_layout)
        
        # Log transform option
        self._log_transform_check = QCheckBox("Применить log(y) преобразование")
        self._log_transform_check.setToolTip("Полезно для данных с  большим разбросом значений")
        config_layout.addWidget(self._log_transform_check)
        
        prepare_btn = QPushButton("Подготовить данные")
        prepare_btn.setStyleSheet("""
            QPushButton {
                background-color: #81C784;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #66BB6A;
            }
        """)
        prepare_btn.clicked.connect(self._prepare_data)
        config_layout.addWidget(prepare_btn)
        
        layout.addWidget(self._config_group)
        
        self._status_label = QLabel()
        self._status_label.setStyleSheet("""
            QLabel {
                background-color: #373737;
                color: #E0E0E0;
                padding: 10px;
                border-radius: 5px;
            }
        """)
        self._status_label.setVisible(False)
        layout.addWidget(self._status_label)
        
        layout.addStretch()
        
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
    
    def _load_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Открыть файл данных",
            "", "Data Files (*.xlsx *.xls *.csv)"
        )
        
        if not filepath:
            return
        
        try:
            data_info = self.data_manager.load_file(filepath)
            
            # Update UI
            filename = filepath.split('/')[-1]
            self._file_label.setText(f"{filename} ({data_info.shape[0]} строк, {data_info.shape[1]} столбцов)")
            self._file_label.setStyleSheet("color: #E0E0E0; font-weight: bold;")
            
            # Update preview table
            preview = self.data_manager.get_data_preview(10)
            self._preview_table.setRowCount(len(preview))
            self._preview_table.setColumnCount(len(preview.columns))
            self._preview_table.setHorizontalHeaderLabels(preview.columns.tolist())
            
            for i, row in preview.iterrows():
                for j, val in enumerate(row):
                    item = QTableWidgetItem(str(val))
                    self._preview_table.setItem(i, j, item)
            
            self._preview_table.resizeColumnsToContents()
            self._preview_group.setVisible(True)
            
            # Update configuration
            self._target_combo.clear()
            self._target_combo.addItems(data_info.columns)
            
            # Smart target selection: prefer 'y', 'target', 'class', or last column
            exclude_patterns = ['year', 'год', 'id', 'index', 'date', 'дата', 'time', 'timestamp', 'header_id']
            preferred_patterns = ['y', 'y1', 'target', 'label', 'class', 'класс', 'class_id', 'category']
            
            best_target_idx = len(data_info.columns) - 1  # Default: last column
            
            for i, col in enumerate(data_info.columns):
                col_lower = col.lower().strip()
                if col_lower in preferred_patterns or col_lower.startswith('y'):
                    best_target_idx = i
                    break
            
            if data_info.columns[best_target_idx].lower() in exclude_patterns:
                for i in range(len(data_info.columns) - 1, -1, -1):
                    if data_info.columns[i].lower() not in exclude_patterns:
                        best_target_idx = i
                        break
            
            self._target_combo.setCurrentIndex(best_target_idx)
            
            self._features_list.clear()
            for col in data_info.columns:
                item = QListWidgetItem(col)
                self._features_list.addItem(item)
                item.setSelected(True)
            
            self._config_group.setVisible(True)
            self._status_label.setVisible(False)
            
            # Check for missing values
            missing_count = self.data_manager.get_missing_count()
            if missing_count > 0:
                missing_by_col = self.data_manager.get_missing_by_column()
                cols_with_missing = [f"{col}: {cnt}" for col, cnt in missing_by_col.items() if cnt > 0]
                self._missing_label.setText(
                    f"Обнаружено {missing_count} пропущенных значений!\n"
                    f"Столбцы: {', '.join(cols_with_missing[:5])}"
                    + ("..." if len(cols_with_missing) > 5 else "")
                )
                self._missing_group.setVisible(True)
            else:
                self._missing_group.setVisible(False)
            
            self.data_loaded.emit(data_info)
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл:\n{str(e)}")
    
    def _on_target_changed(self, target: str):
        """Update features list when target changes"""
        for i in range(self._features_list.count()):
            item = self._features_list.item(i)
            if item.text() == target:
                item.setSelected(False)
    
    def _on_features_changed(self):
        """Handle feature selection change"""
        pass
    
    def _select_all_features(self):
        target = self._target_combo.currentText()
        for i in range(self._features_list.count()):
            item = self._features_list.item(i)
            if item.text() != target:
                item.setSelected(True)
    
    def _deselect_all_features(self):
        for i in range(self._features_list.count()):
            self._features_list.item(i).setSelected(False)
    
    def _prepare_data(self):
        target = self._target_combo.currentText()
        features = [self._features_list.item(i).text() 
                   for i in range(self._features_list.count())
                   if self._features_list.item(i).isSelected()]
        
        if not target:
            QMessageBox.warning(self, "Ошибка", "Выберите целевую переменную")
            return
        
        if not features:
            QMessageBox.warning(self, "Ошибка", "Выберите хотя бы один признак")
            return
        
        if target in features:
            features.remove(target)
        
        try:
            self.data_manager.set_target_and_features(target, features)
            X_train, X_test, y_train, y_test = self.data_manager.prepare_data(
                test_size=self._test_size_spin.value(),
                random_state=self._random_state_spin.value(),
                apply_log_transform=self._log_transform_check.isChecked()
            )
            
            # Detailed status with statistics
            y_min, y_max = y_train.min(), y_train.max()
            y_mean = y_train.mean()
            
            status_text = (
                f"Данные готовы\n"
                f"Train: {X_train.shape[0]} строк, Test: {X_test.shape[0]} строк\n"
                f"Целевая '{target}': [{y_min:.2f} ... {y_max:.2f}], mean={y_mean:.2f}\n"
                f"Признаков: {len(features)}"
            )
            
            self._status_label.setText(status_text)
            self._status_label.setVisible(True)
            
            # Print debug info to terminal
            print("\n" + "="*50)
            print("ПОДГОТОВКА ДАННЫХ - ОТЛАДКА")
            print("="*50)
            print(f"Целевая переменная: {target}")
            print(f"Признаки ({len(features)}): {features[:5]}{'...' if len(features) > 5 else ''}")
            print(f"X_train shape: {X_train.shape}")
            print(f"X_test shape: {X_test.shape}")
            print(f"y_train: min={y_train.min():.4f}, max={y_train.max():.4f}, mean={y_train.mean():.4f}")
            print(f"y_test: min={y_test.min():.4f}, max={y_test.max():.4f}, mean={y_test.mean():.4f}")
            print("="*50 + "\n")
            
            self.data_configured.emit()
            
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка подготовки данных:\n{str(e)}")
    
    def get_test_size(self) -> float:
        return self._test_size_spin.value()
    
    def get_random_state(self) -> int:
        return self._random_state_spin.value()
    
    def is_log_transform(self) -> bool:
        return self._log_transform_check.isChecked()
    
    def _handle_missing(self):
        """Handle missing values using selected strategy"""
        strategy = self._missing_combo.currentData()
        
        try:
            affected = self.data_manager.handle_missing_values(strategy)
            
            # Update file label with new row count
            data_info = self.data_manager.data_info
            filename = data_info.filename.split('/')[-1]
            self._file_label.setText(
                f"✓ {filename} ({data_info.shape[0]} строк, {data_info.shape[1]} столбцов)"
            )
            
            # Update preview table
            preview = self.data_manager.get_data_preview(10)
            self._preview_table.setRowCount(len(preview))
            for i, row in preview.iterrows():
                for j, val in enumerate(row):
                    item = QTableWidgetItem(str(val))
                    self._preview_table.setItem(i, j, item)
            
            missing_count = self.data_manager.get_missing_count()
            if missing_count == 0:
                self._missing_group.setVisible(False)
                QMessageBox.information(
                    self, "Успех", 
                    f"Пропущенные значения обработаны!\n"
                    f"Затронуто строк/ячеек: {affected}"
                )
            else:
                self._missing_label.setText(
                    f"Осталось {missing_count} пропущенных значений.\n"
                    "Попробуйте другую стратегию."
                )
                
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка обработки: {str(e)}")
