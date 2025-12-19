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

## Фоточки

<img width="735" height="836" alt="Снимок экрана 2025-12-18 в 18 27 03" src="https://github.com/user-attachments/assets/a7afd64d-55d4-40a9-918f-69acb19c9d36" />

<img width="701" height="753" alt="Снимок экрана 2025-12-18 в 16 25 20" src="https://github.com/user-attachments/assets/e122d69e-39fb-4ffc-b252-453d7db8096f" />

<img width="1440" height="900" alt="Снимок экрана 2025-12-18 в 16 28 10" src="https://github.com/user-attachments/assets/a39756e1-07eb-4b20-985c-160b4eed5977" />

<img width="1440" height="900" alt="Снимок экрана 2025-12-18 в 16 17 22" src="https://github.com/user-attachments/assets/77940912-2cd4-494f-ba91-c57e9d980127" />

<img width="1440" height="900" alt="Снимок экрана 2025-12-18 в 16 17 31" src="https://github.com/user-attachments/assets/19bb7ecd-98ba-46fc-871e-53ecfc0b4266" />

<img width="735" height="784" alt="Снимок экрана 2025-12-19 в 13 21 29" src="https://github.com/user-attachments/assets/30b4759b-222d-46c8-b978-5c8e4c9c0234" />





## Автор

Тимур Салем

## Лицензия

MIT License
