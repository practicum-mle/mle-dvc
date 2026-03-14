# scripts/fit.py

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from category_encoders import CatBoostEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from catboost import CatBoostClassifier
import yaml
import os
import joblib

def fit_model():
    # 1. Загрузка параметров
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)
    
    index_col = params['index_col']
    target_col = params['target_col']

    # 2. Загрузка данных
    data = pd.read_csv('data/initial_data.csv')

    # 3. Подготовка данных (РЕКОМЕНДАЦИЯ 1: Убираем утечку)
    # Отделяем признаки от таргета и убираем ID
    X = data.drop([target_col, index_col], axis=1)
    y = data[target_col]

    # Определяем типы колонок на основе X
    cat_features = X.select_dtypes(include='object')
    potential_binary_features = cat_features.nunique() == 2

    binary_cat_cols = potential_binary_features[potential_binary_features].index.tolist()
    other_cat_cols = potential_binary_features[~potential_binary_features].index.tolist()
    num_features = X.select_dtypes(['float', 'int']).columns.tolist()

    preprocessor = ColumnTransformer(
        [
            ('binary', OneHotEncoder(drop=params['one_hot_drop']), binary_cat_cols),
            ('cat', CatBoostEncoder(return_df=False), other_cat_cols),
            ('num', StandardScaler(), num_features)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )

    # Используем параметр из yaml (РЕКОМЕНДАЦИЯ 4)
    model = CatBoostClassifier(
        auto_class_weights=params['auto_class_weights'],
        verbose=False
    )

    pipeline = Pipeline(
        [
            ('preprocessor', preprocessor),
            ('model', model)
        ]
    )
    
    # Обучаем только на признаках X
    pipeline.fit(X, y) 

    # 4. Сохранение
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/fitted_model.pkl')
    print("Success: Model trained without leakage and saved.")

if __name__ == '__main__':
    fit_model()