import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
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

# Classification Models Function
def run_classification_models(X_train, X_test, y_train, y_test):
    models = {
        'LightGBM': lgb.LGBMClassifier(random_state=42),
    }
    
    results = {}
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred)
        }
        trained_models[name] = model
        print(f"{name} - Accuracy: {results[name]['Accuracy']:.4f}, F1 Score: {results[name]['F1 Score']:.4f}")
    
    return results, trained_models

# Problem 1: Classification (Predict Income)
def problem_1():
    features = ['age', 'workclass', 'education', 'education-num', 'marital-status', 'occupation', 
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
    data = pd.read_csv('adult_combined.csv')
    X, y, preprocessor = preprocess_data(data, features, 'income')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print("Problem 1 - Income Classification:")
    results, trained_models = run_classification_models(X_train, X_test, y_train, y_test)
    return results, trained_models, preprocessor

# Function to predict new input
def predict_new_data(new_data, features, trained_model, preprocessor):
    new_df = pd.DataFrame([new_data], columns=features)
    new_X = preprocessor.transform(new_df)
    prediction = trained_model.predict(new_X)
    return prediction[0]
