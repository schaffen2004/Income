import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import lightgbm as lgb

# General preprocessing function
def preprocess_data(df, features, target=None):
    X = df[features].copy()
    y = df[target] if target else None
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ]), categorical_cols)
    ])
    
    X_processed = preprocessor.fit_transform(X)
    
    num_features = numerical_cols
    cat_encoder = preprocessor.named_transformers_['cat']['encoder']
    cat_features = cat_encoder.get_feature_names_out(categorical_cols)
    feature_names = num_features + cat_features.tolist()
    
    return pd.DataFrame(X_processed, columns=feature_names), y, preprocessor

# Regression Models Function
def run_regression_models(X_train, X_test, y_train, y_test):
    models = {
        'LightGBM': lgb.LGBMRegressor(random_state=42)
   
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred)
        }
        results[name] = model
        print(results)
    
    return results, model

# Problem 3: Regression (Predict Hours Per Week)
def problem_3():
    features = ['age', 'education', 'education-num', 'occupation']
    data = pd.read_csv('adult_combined.csv')
    X, y, preprocessor = preprocess_data(data, features, 'hours-per-week')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Problem 3 - Hours Per Week Regression:")
    results, trained_models = run_regression_models(X_train, X_test, y_train, y_test)
    return results, trained_models, preprocessor

# Function to predict new input for regression
def predict_new_regression_data(new_data, features, trained_model, preprocessor):
    new_df = pd.DataFrame([new_data], columns=features)
    new_X = preprocessor.transform(new_df)
    prediction = trained_model.predict(new_X)
    return f"{round(prediction[0],2)}h"
