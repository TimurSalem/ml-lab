# ML Lab

Интерактивный лабораторный интерфейс для обучения и анализа моделей машинного обучения.

## Установка

```bash
pip install -r requirements.txt
```

## Запуск

```bash
python3 -m ml_lab
```

## Возможности

### Алгоритмы машинного обучения

**Регрессия:**
- Linear Regression
- Ridge (L2 регуляризация)
- Lasso (L1 регуляризация)
- Random Forest Regressor
- Gradient Boosting Regressor
- K-Neighbors Regressor

**Классификация:**
- Logistic Regression
- K-Neighbors Classifier

**Кластеризация:**
- KMeans
- DBSCAN
- Agglomerative Clustering

### Функционал

- Загрузка данных из CSV/Excel файлов
- Обработка пропущенных значений (удаление, среднее, медиана, мода)
- Масштабирование данных (StandardScaler, MinMaxScaler, RobustScaler)
- Кросс-валидация с настраиваемым количеством фолдов
- Автоподбор параметров (GridSearchCV)
- Визуализация результатов:
  - Прогноз vs Факт
  - Важность признаков
  - Остатки модели
  - Confusion Matrix
  - Кластеры
- Экспорт результатов в CSV/Excel
- Сохранение и загрузка обученных моделей

## Структура проекта

```
ml_lab/
├── __init__.py
├── __main__.py
├── main.py
├── config.py              # Конфигурация приложения
├── core/
│   ├── data_manager.py    # Управление данными
│   ├── training_manager.py # Управление обучением
│   ├── model_manager.py   # Реестр моделей
│   └── viz_manager.py     # Визуализация
├── models/
│   ├── base.py            # Базовый класс модели
│   ├── regressors/        # Модели регрессии
│   ├── classifiers/       # Модели классификации
│   └── clustering/        # Модели кластеризации
├── ui/
│   ├── main_window.py     # Главное окно
│   └── widgets/           # UI компоненты
└── utils/
    └── scalers.py         # Утилиты масштабирования
```

## Технологии

- Python 3.10+
- PyQt6 — графический интерфейс
- scikit-learn — алгоритмы машинного обучения
- pandas — работа с данными
- matplotlib — визуализация
- numpy — вычисления

## Автор

Тимур Салем

## Лицензия

MIT License
