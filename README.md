# MTS 26 Projects

Сборник учебных и соревновательных проектов по машинному обучению и анализу данных. Каждый проект находится в отдельной папке со своим README, ноутбуком и описанием запуска.

## Проекты

| Проект                                                                                                     | Задача | Тип | Ключевой результат |
|------------------------------------------------------------------------------------------------------------|---|---|---|
| [ChurnRadar](./Churn_Radar)                                                                                | Бинарная классификация оттока клиентов телеком-оператора | Итоговая работа по модулю ML | Test F1 = 0.650, инференс одной записи менее 0.1 мс |
| [Analysis of Pricing Factors for Lending Club Loans](./Analysis_of_Pricing_Factors_for_Lending_Club_Loans) | Регрессионный анализ процентной ставки по P2P-кредитам | Итоговая работа по статистике и эконометрике | OLS с робастными ошибками HC3, R² = 0.440 |
| [NYC Availability Forecast](./NYC_Availability_Forecast)                                                   | Прогноз числа дней доступности объявления об аренде в Нью-Йорке | ML-соревнование | Пятое место, RMSE около 93 дней |

---

## ChurnRadar

Сверхбыстрый скоринг оттока клиентов телеком-оператора на LightGBM с упором на production-инференс и мониторинг дрифта без истинных меток.

- **Модель:** LightGBM, 5-fold `StratifiedKFold`, гиперпараметры подобраны многокритериальным Optuna (F1 против сложности модели).
- **Признаки:** флаги пропусков, парсинг строки `pack`, восстановление счётчиков после обфускации, производные показатели, `CatBoostEncoder`.
- **Инференс:** модель скомпилирована через `lleaves`, пайплайн на чистом Python, один поток CPU, функция `get_score(dict_input)` укладывается в 0.1 мс.
- **Мониторинг:** класс `InputDriftMonitor` (PSI, KS, доля положительных предсказаний, unseen rate), накладные расходы около 6 мкс на событие.
- **Метрики:** OOF ROC-AUC 0.9031, OOF F1 0.6521, Test F1 0.6501.

Стек: Python, LightGBM, lleaves, Optuna, category_encoders, scikit-learn, pandas, NumPy, SciPy, Matplotlib.

Подробнее: [ChurnRadar/README.md](./ChurnRadar/README.md)

---

## Analysis of Pricing Factors for Lending Club Loans

Исследовательский анализ данных платформы Lending Club и построение регрессионной модели для изучения факторов, связанных с процентной ставкой (`int_rate`).

- **Этапы:** подготовка данных и feature engineering, EDA, проверка статистических гипотез, анализ мультиколлинеарности (VIF), сравнение моделей (Dummy, линейная регрессия, Lasso, Elastic Net, Ridge с RFE).
- **Итоговая модель:** OLS-регрессия с робастными стандартными ошибками HC3, так как выявлена гетероскедастичность.
- **Особенность:** `grade` и `sub_grade` исключены как прямые детерминанты ставки.
- **Результат:** R² = 0.440, коэффициенты интерпретируются как условные статистические связи, а не причинные эффекты.
- **Данные:** [Lending Club Loan Data (Kaggle)](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv), файл `datasets/loan.csv` нужно подготовить самостоятельно.

Стек: Python, pandas, NumPy, Matplotlib, Seaborn, Plotly, SciPy, statsmodels, scikit-learn, phik.

Подробнее: [Analysis_of_Pricing_Factors_for_Lending_Club_Loans/README.md](./Analysis_of_Pricing_Factors_for_Lending_Club_Loans/README.md)

---

## NYC Availability Forecast

Прогноз количества дней (от 0 до 365), в течение которых объявление о краткосрочной аренде в Нью-Йорке доступно для бронирования. Проект выполнен в рамках ML-соревнования, автор занял пятое место.

- **Признаки:** ID-признаки и статистики по повторяющемуся `_id`, географические признаки, агрегаты по хостам и районам, текстовые признаки (TF-IDF с SVD, эмбеддинги SentenceTransformer).
- **Кодирование и отбор:** `CatBoostEncoder` в OOF-режиме, мягкий отбор признаков по SHAP.
- **Модель:** двухуровневый stacking. Первый уровень: CatBoost, LightGBM, XGBoost. Второй уровень: Ridge.
- **Валидация:** 5-fold `StratifiedKFold` по бинам целевой переменной, метрика MSE, предсказания ограничены диапазоном [0, 365].
- **Результат:** RMSE около 93 дней.
- **Данные:** соревнование приватное, данные и ссылка на него не распространяются.

Стек: Python, pandas, NumPy, scikit-learn, CatBoost, LightGBM, XGBoost, Optuna, SHAP, SentenceTransformers, PyTorch, UMAP, Folium.

Подробнее: [NYC_Availability_Forecast/README.md](./NYC_Availability_Forecast/README.md)

---

## Структура репозитория

```text
.
├── README.md
├── ChurnRadar/
│   ├── churn_radar.ipynb
│   ├── churn_radar.html
│   └── README.md
├── Analysis_of_Pricing_Factors_for_Lending_Club_Loans/
│   ├── analysis_of_pricing_factors_for_lending_club_loans.ipynb
│   ├── analysis_of_pricing_factors_for_lending_club_loans.html
│   ├── datasets/
│   │   └── loan.csv
│   └── README.md
└── NYC_Availability_Forecast/
    ├── nyc-availability-forecast.ipynb
    ├── nyc-availability-forecast.html
    ├── leaderboard.png
    └── README.md
```

## Как запустить

1. Клонируйте репозиторий:

   ```bash
   git clone https://github.com/LevBulanov/mts_26_projects.git
   cd mts_26_projects
   ```

2. Перейдите в папку нужного проекта и следуйте инструкциям из её README.

3. Для каждого проекта рекомендуется отдельное виртуальное окружение:

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Linux / macOS
   .venv\Scripts\activate         # Windows
   ```

4. Установите зависимости, перечисленные в README проекта, и запустите ноутбук:

   ```bash
   jupyter notebook
   ```

## Данные

Файлы с данными в репозиторий не входят:

- ChurnRadar: `train.csv` и `test_full.csv` кладутся рядом с ноутбуком.
- Lending Club: `datasets/loan.csv` скачивается с Kaggle по ссылке выше.
- NYC Availability Forecast: данные приватного соревнования недоступны.

## Общие замечания

- Результаты могут слегка отличаться в зависимости от версий библиотек и окружения.
- Подбор гиперпараметров с помощью Optuna в ноутбуках закомментирован или отключён из-за длительного времени работы, найденные параметры зашиты в код.
- Часть ноутбуков содержит вычислительно затратные этапы (эмбеддинги, обучение ансамблей, компиляция модели).
