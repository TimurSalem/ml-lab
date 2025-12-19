"""
Visualization Panel widget for ML Lab
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTabWidget, QComboBox, QSplitter, QScrollArea, QFrame,
    QFileDialog, QMessageBox
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QFont

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class CustomNavigationToolbar(NavigationToolbar):
    """Extended NavigationToolbar with aspect ratio controls in Figure Options"""
    
    def edit_parameters(self):
        """Override to add aspect ratio options to Figure Options dialog"""
        from matplotlib.backends.qt_compat import QtWidgets
        
        axes = self.canvas.figure.get_axes()
        if not axes:
            QtWidgets.QMessageBox.warning(
                self.canvas.parent(), "Ошибка", "Нет осей для настройки")
            return
        
        # Get first axes for simplicity
        ax = axes[0]
        
        # Create custom dialog
        dialog = QtWidgets.QDialog(self.canvas.parent())
        dialog.setWindowTitle("Настройки графика")
        dialog.setMinimumWidth(400)
        dialog.setStyleSheet("""
            QDialog { background-color: #2D2D2D; color: #E0E0E0; }
            QLabel { color: #E0E0E0; }
            QLineEdit, QComboBox { 
                background-color: #424242; 
                color: #E0E0E0; 
                border: 1px solid #555; 
                padding: 5px;
                border-radius: 3px;
            }
            QGroupBox { 
                color: #E0E0E0; 
                border: 1px solid #555; 
                margin-top: 10px; 
                padding-top: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; }
            QPushButton {
                background-color: #424242;
                color: #E0E0E0;
                border: 1px solid #555;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #555; }
        """)
        
        layout = QtWidgets.QVBoxLayout(dialog)
        
        # ===== ASPECT RATIO GROUP =====
        aspect_group = QtWidgets.QGroupBox("Пропорции осей")
        aspect_layout = QtWidgets.QVBoxLayout(aspect_group)
        
        self._aspect_combo = QtWidgets.QComboBox()
        self._aspect_combo.addItems([
            "auto - Автоматически",
            "equal - Равные пропорции (1:1)",
        ])
        
        # Set current value
        current_aspect = ax.get_aspect()
        if current_aspect == 'equal':
            self._aspect_combo.setCurrentIndex(1)
        else:
            self._aspect_combo.setCurrentIndex(0)
        
        aspect_layout.addWidget(QtWidgets.QLabel("Режим пропорций:"))
        aspect_layout.addWidget(self._aspect_combo)
        
        # Quick aspect buttons
        btn_layout = QtWidgets.QHBoxLayout()
        
        auto_btn = QtWidgets.QPushButton("Авто")
        auto_btn.clicked.connect(lambda: self._aspect_combo.setCurrentIndex(0))
        btn_layout.addWidget(auto_btn)
        
        equal_btn = QtWidgets.QPushButton("1:1 (Equal)")
        equal_btn.setStyleSheet("QPushButton { background-color: #81C784; color: #1E1E1E; }")
        equal_btn.clicked.connect(lambda: self._aspect_combo.setCurrentIndex(1))
        btn_layout.addWidget(equal_btn)
        
        aspect_layout.addLayout(btn_layout)
        layout.addWidget(aspect_group)
        
        # AXIS LIMITS GROUP
        limits_group = QtWidgets.QGroupBox("Пределы осей")
        limits_layout = QtWidgets.QGridLayout(limits_group)
        
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        limits_layout.addWidget(QtWidgets.QLabel("X Min:"), 0, 0)
        self._xmin_edit = QtWidgets.QLineEdit(f"{xlim[0]:.4f}")
        limits_layout.addWidget(self._xmin_edit, 0, 1)
        
        limits_layout.addWidget(QtWidgets.QLabel("X Max:"), 0, 2)
        self._xmax_edit = QtWidgets.QLineEdit(f"{xlim[1]:.4f}")
        limits_layout.addWidget(self._xmax_edit, 0, 3)
        
        limits_layout.addWidget(QtWidgets.QLabel("Y Min:"), 1, 0)
        self._ymin_edit = QtWidgets.QLineEdit(f"{ylim[0]:.4f}")
        limits_layout.addWidget(self._ymin_edit, 1, 1)
        
        limits_layout.addWidget(QtWidgets.QLabel("Y Max:"), 1, 2)
        self._ymax_edit = QtWidgets.QLineEdit(f"{ylim[1]:.4f}")
        limits_layout.addWidget(self._ymax_edit, 1, 3)
        
        layout.addWidget(limits_group)
        
        # ===== TITLE GROUP =====
        title_group = QtWidgets.QGroupBox("Заголовок")
        title_layout = QtWidgets.QVBoxLayout(title_group)
        
        self._title_edit = QtWidgets.QLineEdit(ax.get_title())
        title_layout.addWidget(self._title_edit)
        
        layout.addWidget(title_group)
        
        # ===== BUTTONS =====
        button_layout = QtWidgets.QHBoxLayout()
        
        apply_btn = QtWidgets.QPushButton("Применить")
        apply_btn.setStyleSheet("QPushButton { background-color: #81C784; color: #1E1E1E; }")
        apply_btn.clicked.connect(lambda: self._apply_settings(ax, dialog))
        button_layout.addWidget(apply_btn)
        
        cancel_btn = QtWidgets.QPushButton("Отмена")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        dialog.exec()
    
    def _apply_settings(self, ax, dialog):
        """Apply settings from dialog to axes"""
        try:
            # Apply aspect ratio
            aspect_idx = self._aspect_combo.currentIndex()
            if aspect_idx == 0:
                ax.set_aspect('auto')
            else:
                ax.set_aspect('equal', adjustable='box')
            
            # Apply axis limits
            xmin = float(self._xmin_edit.text())
            xmax = float(self._xmax_edit.text())
            ymin = float(self._ymin_edit.text())
            ymax = float(self._ymax_edit.text())
            
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            
            # Apply title
            ax.set_title(self._title_edit.text(), color='#E0E0E0')
            
            self.canvas.figure.tight_layout()
            self.canvas.draw()
            dialog.accept()
            
        except ValueError as e:
            from matplotlib.backends.qt_compat import QtWidgets
            QtWidgets.QMessageBox.warning(
                dialog, "Ошибка", f"Неверный формат числа: {e}"
            )


class PlotWidget(QWidget):
    """Widget that contains a matplotlib figure with toolbar"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        
        # Default empty figure
        self._current_figure = None
        self._canvas = None
        self._toolbar = None
        
        # Create initial empty figure
        self._create_empty_figure()
    
    def _create_empty_figure(self):
        """Create an empty placeholder figure"""
        fig = Figure(figsize=(8, 6), dpi=100, facecolor='#1E1E1E')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#1E1E1E')
        ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', 
                fontsize=14, color='#666666', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        self._set_canvas(fig)
    
    def _set_canvas(self, fig: Figure):
        """Set the canvas to display the given figure"""
        if self._canvas:
            self._layout.removeWidget(self._canvas)
            self._canvas.deleteLater()
        if self._toolbar:
            self._layout.removeWidget(self._toolbar)
            self._toolbar.deleteLater()
        
        # Stre figure reference
        self._current_figure = fig
        
        # Create new canvas
        self._canvas = FigureCanvas(fig)
        self._toolbar = CustomNavigationToolbar(self._canvas, self)
        
        # Add custom buttons to toolbar
        self._add_custom_toolbar_buttons()
        
        # Add to layout
        self._layout.addWidget(self._toolbar)
        self._layout.addWidget(self._canvas)
        
        self._canvas.draw()
    
    def _add_custom_toolbar_buttons(self):
        """Add custom buttons to the navigation toolbar"""
        from PyQt6.QtWidgets import QToolButton
        from PyQt6.QtGui import QIcon
        
        # Add separator
        self._toolbar.addSeparator()
        
        # Equal Aspect Ratio button
        self._equal_aspect_btn = QToolButton()
        self._equal_aspect_btn.setText("1:1")
        self._equal_aspect_btn.setToolTip("Равные пропорции осей (для кластеризации)")
        self._equal_aspect_btn.setCheckable(True)
        self._equal_aspect_btn.setStyleSheet("""
            QToolButton {
                padding: 4px 8px;
                border: 1px solid #555;
                border-radius: 3px;
                background: #2D2D2D;
                color: #E0E0E0;
            }
            QToolButton:checked {
                background: #81C784;
                color: #1E1E1E;
                font-weight: bold;
            }
            QToolButton:hover {
                border-color: #FFB74D;
            }
        """)
        self._equal_aspect_btn.clicked.connect(self._toggle_equal_aspect)
        self._toolbar.addWidget(self._equal_aspect_btn)
        
        self._auto_scale_btn = QToolButton()
        self._auto_scale_btn.setText("Авто")
        self._auto_scale_btn.setToolTip("Автоматический масштаб осей")
        self._auto_scale_btn.setStyleSheet("""
            QToolButton {
                padding: 4px 8px;
                border: 1px solid #555;
                border-radius: 3px;
                background: #2D2D2D;
                color: #E0E0E0;
            }
            QToolButton:hover {
                border-color: #FFB74D;
            }
        """)
        self._auto_scale_btn.clicked.connect(self._auto_scale)
        self._toolbar.addWidget(self._auto_scale_btn)
    
    def _toggle_equal_aspect(self):
        """Toggle equal aspect ratio for all axes"""
        if self._current_figure:
            for ax in self._current_figure.axes:
                if self._equal_aspect_btn.isChecked():
                    ax.set_aspect('equal', adjustable='box')
                else:
                    ax.set_aspect('auto')
            self._current_figure.tight_layout()
            self._canvas.draw()
    
    def _auto_scale(self):
        """Reset to automatic scaling"""
        if self._current_figure:
            for ax in self._current_figure.axes:
                ax.set_aspect('auto')
                ax.autoscale(enable=True)
            self._equal_aspect_btn.setChecked(False)
            self._current_figure.tight_layout()
            self._canvas.draw()
    
    def clear(self):
        """Clear the figure"""
        self._create_empty_figure()
    
    def set_figure(self, fig: Figure):
        """Replace current figure with the provided figure"""
        if fig is None:
            self._create_empty_figure()
            return
        
        # Set dark background for all axes
        fig.patch.set_facecolor('#1E1E1E')
        for ax in fig.axes:
            ax.set_facecolor('#1E1E1E')
            # Style axis labels and ticks
            ax.tick_params(colors='#E0E0E0')
            ax.xaxis.label.set_color('#E0E0E0')
            ax.yaxis.label.set_color('#E0E0E0')
            ax.title.set_color('#E0E0E0')
            for spine in ax.spines.values():
                spine.set_color('#424242')
            # Style legend if present
            legend = ax.get_legend()
            if legend:
                legend.get_frame().set_facecolor('#2D2D2D')
                legend.get_frame().set_edgecolor('#424242')
                for text in legend.get_texts():
                    text.set_color('#E0E0E0')
        
        fig.tight_layout()
        self._set_canvas(fig)
    
    def save(self, filepath: str):
        """Save the figure to file"""
        if self._current_figure:
            self._current_figure.savefig(filepath, dpi=150, bbox_inches='tight', 
                                         facecolor='#1E1E1E', edgecolor='none')


class VisualizationPanel(QWidget):
    """Widget for displaying various visualizations"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._plots = {}
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        header = QHBoxLayout()
        
        title = QLabel("Визуализация")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #E0E0E0;")
        header.addWidget(title)
        
        header.addStretch()
        
        # Model export button
        self._export_model_btn = QPushButton("Сохранить модель")
        self._export_model_btn.setStyleSheet("""
            QPushButton {
                background-color: #424242;
                color: #E0E0E0;
                border: 1px solid #555555;
                padding: 8px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #555555;
                border-color: #81C784;
            }
        """)
        header.addWidget(self._export_model_btn)
        
        # Model import button
        self._import_model_btn = QPushButton("Загрузить модель")
        self._import_model_btn.setStyleSheet("""
            QPushButton {
                background-color: #424242;
                color: #E0E0E0;
                border: 1px solid #555555;
                padding: 8px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #555555;
                border-color: #FFB74D;
            }
        """)
        header.addWidget(self._import_model_btn)
        
        layout.addLayout(header)
        
        # Tab widget for different plots
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #424242;
                border-radius: 4px;
                background: #1E1E1E;
            }
            QTabBar::tab {
                padding: 8px 12px;
                margin-right: 2px;
                background: #2D2D2D;
                border: 1px solid #424242;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-size: 11px;
                color: #BDBDBD;
            }
            QTabBar::tab:selected {
                background: #1E1E1E;
                border-bottom: 2px solid #555555;
                color: #E0E0E0;
            }
        """)
        layout.addWidget(self._tabs)
        
        # Create default tabs
        self._create_default_tabs()
    
    def _create_default_tabs(self):
        """Create default visualization tabs"""
        tab_configs = [
            ("predictions", "Прогноз vs Факт"),
            ("feature_importance", "Важность признаков"),
            ("learning_curve", "Кривая обучения"),
            ("residuals", "Остатки"),
            ("comparison", "Сравнение"),
        ]
        
        for key, label in tab_configs:
            plot_widget = PlotWidget()
            self._plots[key] = plot_widget
            self._tabs.addTab(plot_widget, label)
    
    def update_plot(self, key: str, figure: Figure):
        """Update a specific plot"""
        if key in self._plots:
            self._plots[key].set_figure(figure)
    
    def add_custom_plot(self, key: str, label: str, figure: Figure):
        """Add a new custom plot tab"""
        if key not in self._plots:
            plot_widget = PlotWidget()
            self._plots[key] = plot_widget
            self._tabs.addTab(plot_widget, label)
        
        self.update_plot(key, figure)
    
    def clear_all(self):
        """Clear all plots"""
        for plot in self._plots.values():
            plot.clear()
    
    def _export_current(self):
        """Export current plot to file"""
        current_idx = self._tabs.currentIndex()
        current_key = list(self._plots.keys())[current_idx]
        current_plot = self._plots[current_key]
        
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Сохранить график",
            f"{current_key}.png",
            "PNG Images (*.png);;SVG Images (*.svg);;PDF Files (*.pdf)"
        )
        
        if filepath:
            try:
                current_plot.save(filepath)
                QMessageBox.information(self, "Успех", f"График сохранён: {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить: {str(e)}")
    
    def get_plot(self, key: str) -> PlotWidget:
        """Get a plot widget by key"""
        return self._plots.get(key)
