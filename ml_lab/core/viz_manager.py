import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from typing import Optional, List, Tuple
from ..models.base import BaseModel, TrainingResult, ModelCategory

class VizManager:
    
    plt.style.use('dark_background')
    
    @staticmethod
    def create_figure(figsize: Tuple[int, int] = (10, 6)) -> Tuple[Figure, any]:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    
    @staticmethod
    def _apply_dark_style(fig, ax):
        fig.patch.set_facecolor('#1E1E1E')
        if isinstance(ax, np.ndarray):
            for a in ax.flat:
                VizManager._style_axis(a)
        else:
            VizManager._style_axis(ax)
    
    @staticmethod
    def _style_axis(ax):
        ax.set_facecolor('#1E1E1E')
        ax.tick_params(colors='#E0E0E0', labelsize=9)
        ax.xaxis.label.set_color('#E0E0E0')
        ax.yaxis.label.set_color('#E0E0E0')
        ax.title.set_color('#E0E0E0')
        ax.title.set_fontsize(12)
        for spine in ax.spines.values():
            spine.set_color('#424242')
        ax.grid(True, alpha=0.2, color='#666666')
    
    @staticmethod
    def plot_feature_importance(model: BaseModel, 
                                  figsize: Tuple[int, int] = (10, 6)) -> Figure:
        if model._last_result is None or model._last_result.feature_importances is None:
            return None
        
        importances = model._last_result.feature_importances
        
        if len(importances) == 0:
            return None
        
        feature_names = model._feature_names
        if feature_names is None or len(feature_names) != len(importances):
            feature_names = [f"Признак {i+1}" for i in range(len(importances))]
        
        fig, ax = plt.subplots(figsize=figsize)
        VizManager._apply_dark_style(fig, ax)
        
        indices = np.argsort(importances)
        sorted_importances = importances[indices]
        sorted_names = [feature_names[i] for i in indices]
        
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(indices)))
        
        bars = ax.barh(range(len(indices)), sorted_importances, 
                       color=colors, edgecolor='white', linewidth=0.5)
        
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Важность признака', fontsize=11)
        ax.set_title('Важность признаков модели', fontsize=13, fontweight='bold')
        
        for bar, val in zip(bars, sorted_importances):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', ha='left', fontsize=8, color='#E0E0E0')
        
        fig.tight_layout()
        return fig
    
    @staticmethod
    def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                                    title: str = "Прогноз vs Реальные значения",
                                    figsize: Tuple[int, int] = (10, 6)) -> Figure:
        fig, ax = plt.subplots(figsize=figsize)
        VizManager._apply_dark_style(fig, ax)
        
        scatter = ax.scatter(y_true, y_pred, alpha=0.6, c='#4FC3F7', 
                            edgecolor='white', s=50, linewidth=0.5)
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        margin = (max_val - min_val) * 0.05
        ax.plot([min_val - margin, max_val + margin], 
                [min_val - margin, max_val + margin], 
                'r--', lw=2, label='Идеальный прогноз', alpha=0.8)
        
        ax.set_xlabel('Реальные значения', fontsize=11)
        ax.set_ylabel('Прогнозные значения', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', facecolor='#2D2D2D', edgecolor='#424242',
                 labelcolor='#E0E0E0')
        
        ax.set_xlim(min_val - margin, max_val + margin)
        ax.set_ylim(min_val - margin, max_val + margin)
        
        fig.tight_layout()
        return fig
    
    @staticmethod
    def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                       figsize: Tuple[int, int] = (12, 5)) -> Figure:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        VizManager._apply_dark_style(fig, axes)
        
        residuals = y_true - y_pred
        
        axes[0].scatter(y_pred, residuals, alpha=0.6, c='#4FC3F7', 
                       edgecolor='white', s=40, linewidth=0.5)
        axes[0].axhline(y=0, color='#FF5252', linestyle='--', lw=2, alpha=0.8)
        axes[0].set_xlabel('Прогнозные значения', fontsize=10)
        axes[0].set_ylabel('Остатки', fontsize=10)
        axes[0].set_title('Остатки vs Прогноз', fontsize=12, fontweight='bold')
        
        n_bins = min(30, max(10, len(residuals) // 5))
        axes[1].hist(residuals, bins=n_bins, alpha=0.7, color='#4FC3F7', 
                    edgecolor='white', linewidth=0.5)
        axes[1].axvline(x=0, color='#FF5252', linestyle='--', lw=2, alpha=0.8)
        axes[1].set_xlabel('Остатки', fontsize=10)
        axes[1].set_ylabel('Частота', fontsize=10)
        axes[1].set_title('Распределение остатков', fontsize=12, fontweight='bold')
        
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        stats_text = f'μ = {mean_res:.2f}\nσ = {std_res:.2f}'
        axes[1].text(0.95, 0.95, stats_text, transform=axes[1].transAxes,
                    fontsize=9, color='#E0E0E0', va='top', ha='right',
                    bbox=dict(boxstyle='round', facecolor='#2D2D2D', 
                             edgecolor='#424242', alpha=0.8))
        
        fig.tight_layout()
        return fig
    
    @staticmethod
    def plot_learning_curve(train_scores: List[float], test_scores: List[float],
                            x_values: List[int], x_label: str = "Параметр",
                            figsize: Tuple[int, int] = (10, 6)) -> Figure:
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(x_values, train_scores, 'o-', color='steelblue', 
                label='Правильность на обучающем наборе', linewidth=2)
        ax.plot(x_values, test_scores, 'o-', color='darkorange',
                label='Правильность на тестовом наборе', linewidth=2)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel('Правильность (R² / Accuracy)')
        ax.set_title('Кривая обучения')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_time_series_comparison(actual: np.ndarray, 
                                     predictions: dict,  # name -> values
                                     x_labels: Optional[List] = None,
                                     title: str = "Сравнение прогнозов",
                                     figsize: Tuple[int, int] = (14, 7)) -> Figure:
        fig, ax = plt.subplots(figsize=figsize)
        
        x = range(len(actual))
        
        ax.plot(x, actual, 'o-', label='Реальные данные', linewidth=2, color='black')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))
        for (name, values), color in zip(predictions.items(), colors):
            ax.plot(x, values, 'o-', label=name, linewidth=1.5, alpha=0.8, color=color)
        
        if x_labels is not None:
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
        
        ax.set_xlabel('Индекс')
        ax.set_ylabel('Значение')
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_scores_comparison(scores: dict,
                                title: str = "Сравнение метрик",
                                figsize: Tuple[int, int] = (8, 5)) -> Figure:
        fig, ax = plt.subplots(figsize=figsize)
        VizManager._apply_dark_style(fig, ax)
        
        names = list(scores.keys())
        values = list(scores.values())
        
        colors = ['#4FC3F7', '#81C784', '#FFB74D', '#F48FB1', '#B39DDB'][:len(names)]
        
        bars = ax.bar(names, values, color=colors, edgecolor='white', linewidth=1.5)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            label_y = height + 0.02 * max(values) if height >= 0 else height - 0.05 * max(values)
            ax.annotate(f'{val:.4f}',
                       xy=(bar.get_x() + bar.get_width() / 2, label_y),
                       ha='center', va='bottom' if height >= 0 else 'top', 
                       fontsize=11, fontweight='bold', color='#E0E0E0')
        
        ax.set_ylabel('Значение метрики', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        
        min_val = min(values) if min(values) < 0 else 0
        max_val = max(values)
        margin = (max_val - min_val) * 0.15
        ax.set_ylim(min_val - margin if min_val < 0 else 0, max_val + margin)
        
        fig.tight_layout()
        return fig
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, labels: List[str] = None,
                               figsize: Tuple[int, int] = (8, 6)) -> Figure:
        fig, ax = plt.subplots(figsize=figsize)
        VizManager._apply_dark_style(fig, ax)
        
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_tick_params(color='#E0E0E0')
        cbar.outline.set_edgecolor('#424242')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#E0E0E0')
        
        if labels is None:
            labels = [str(i) for i in range(len(cm))]
        
        ax.set(xticks=np.arange(len(labels)),
               yticks=np.arange(len(labels)),
               xticklabels=labels, yticklabels=labels,
               ylabel='Истинный класс',
               xlabel='Предсказанный класс',
               title='Матрица ошибок (Confusion Matrix)')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        thresh = cm.max() / 2.
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center", fontsize=10, fontweight='bold',
                       color="white" if cm[i, j] > thresh else "#333333")
        
        fig.tight_layout()
        return fig
    
    @staticmethod
    def plot_cluster_scatter(X: np.ndarray, labels: np.ndarray,
                              centers: np.ndarray = None,
                              figsize: Tuple[int, int] = (10, 8)) -> Figure:
        fig, ax = plt.subplots(figsize=figsize)
        VizManager._apply_dark_style(fig, ax)
        
        if X.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
            x_label = f"PC1 ({pca.explained_variance_ratio_[0]:.1%})"
            y_label = f"PC2 ({pca.explained_variance_ratio_[1]:.1%})"
            if centers is not None and centers.shape[1] > 2:
                centers = pca.transform(centers)
        else:
            X_2d = X
            x_label = "Признак 1"
            y_label = "Признак 2"
        
        unique_labels = np.unique(labels)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[color], 
                      alpha=0.6, edgecolor='white', s=50, linewidth=0.5,
                      label=f'Кластер {label}')
        
        if centers is not None:
            ax.scatter(centers[:, 0], centers[:, 1], c='#FF5252', marker='X',
                      s=200, edgecolor='white', linewidth=2, label='Центроиды')
        
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title('Визуализация кластеров', fontsize=13, fontweight='bold')
        ax.legend(loc='best', facecolor='#2D2D2D', edgecolor='#424242', 
                 labelcolor='#E0E0E0')
        
        fig.tight_layout()
        return fig
    
    @staticmethod
    def plot_dendrogram(X: np.ndarray, method: str = 'complete',
                         figsize: Tuple[int, int] = (12, 8)) -> Figure:
        from scipy.cluster.hierarchy import linkage, dendrogram
        
        fig, ax = plt.subplots(figsize=figsize)
        
        mergings = linkage(X, method=method)
        dendrogram(mergings, ax=ax, leaf_rotation=90, leaf_font_size=8)
        
        ax.set_xlabel('Индекс образца')
        ax.set_ylabel('Расстояние')
        ax.set_title('Дендрограмма иерархической кластеризации')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_silhouette_scores(n_clusters_range: List[int], scores: List[float],
                                figsize: Tuple[int, int] = (10, 6)) -> Figure:
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(n_clusters_range, scores, 'o-', color='steelblue', linewidth=2)
        ax.set_xlabel('Количество кластеров')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Оценка качества кластеризации')
        ax.grid(True, alpha=0.3)
        
        best_idx = np.argmax(scores)
        ax.scatter([n_clusters_range[best_idx]], [scores[best_idx]], 
                  color='red', s=100, zorder=5, label=f'Лучшее: {n_clusters_range[best_idx]} кластеров')
        ax.legend()
        
        plt.tight_layout()
        return fig
