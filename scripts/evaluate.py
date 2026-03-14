# scripts/evaluate.py

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
import joblib
import json
import yaml
import os

# оценка качества модели
def evaluate_model():
    # 1. Прочитайте файл с гиперпараметрами params.yaml
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)

    # 2. Загрузите результат прошлого шага: fitted_model.pkl
    # А также данные для оценки (те же, что использовались для обучения)
    if not os.path.exists('models/fitted_model.pkl'):
        raise FileNotFoundError("Модель не найдена. Сначала запустите этап fit_model.")
    
    pipeline = joblib.load('models/fitted_model.pkl')
    data = pd.read_csv('data/initial_data.csv')

    # реализуйте основную логику шага с использованием прочтённых гиперпараметров
    cv_strategy = StratifiedKFold(n_splits=5)
    cv_res = cross_validate(
        pipeline,
        data,
        data['target'],
        cv=cv_strategy,
        n_jobs=-1,
        scoring=['f1', 'roc_auc']
    )
    
    for key, value in cv_res.items():
        cv_res[key] = round(value.mean(), 3) 

    # сохраните результата кросс-валидации в cv_res.json
    os.makedirs('cv_results', exist_ok=True)
    with open('cv_results/cv_res.json', 'w') as f:
        json.dump(cv_res, f, indent=4)

if __name__ == '__main__':
    evaluate_model()