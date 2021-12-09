## Auto ML
Библиотека для автоматизации процесса обучения моделей для бинарной классификации.

### Описание
Работа библиотеки основана на выборе модели имеющей лучшую обобщающую способность на основе заданной метрики.

Используемые модели с возможными гиперпараметрами:

    basic_models = {
        KNeighborsClassifier: {
            'n_neighbors': [3, 5, 11],
            'weights': ['uniform', 'distance'],
        },
        LogisticRegression: {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        },
        RandomForestClassifier: {
            'n_estimators': [10, 100, 1000],
            'max_features': ['sqrt', 'log2'],
        },
    }

Доступные метрики оценки качества:

    custom_metrics = {
        'accuracy': accuracy_score,
        'f1': f1_score,
        'precision': precision_score,
        'recall': recall_score,
        'roc_auc': roc_auc_score,
    }

### Примеры использования

Для запуска обучения необходимо ввести следующую команду:

    git clone ...
    cd auto_ml/auto_ml
    python auto_ml.py [OPTIONS] TARGET_FEATURES_PATH PREDICTORS_PATH {accuracy|f1|precision|recall|roc_auc} OUTPUT_PATH

По окончанию обучения будет выведен результат:
    
    2021-12-09 17:26:36,337 :: INFO :: Best model: RandomForestClassifier(max_features='sqrt')
    2021-12-09 17:26:36,337 :: INFO :: Best accuracy score: 0.56

Результат автоматического обучения(наилучшая модель) сохраняется в файл .pickle в директорию OUTPUT_PATH.
