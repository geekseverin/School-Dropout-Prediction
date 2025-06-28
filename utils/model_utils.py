import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import joblib
from mlxtend.frequent_patterns import apriori, association_rules

def train_models(X, y):
    """Entraîner plusieurs modèles de classification"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {}
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    models['RandomForest'] = {
        'model': rf,
        'accuracy': accuracy_score(y_test, rf_pred),
        'predictions': rf_pred
    }
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    models['XGBoost'] = {
        'model': xgb_model,
        'accuracy': accuracy_score(y_test, xgb_pred),
        'predictions': xgb_pred
    }
    
    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42, max_depth=10)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    models['DecisionTree'] = {
        'model': dt,
        'accuracy': accuracy_score(y_test, dt_pred),
        'predictions': dt_pred
    }
    
    return models, X_test, y_test

def select_best_model(models):
    """Sélectionner le meilleur modèle basé sur l'accuracy"""
    best_model_name = max(models.keys(), key=lambda k: models[k]['accuracy'])
    best_model = models[best_model_name]['model']
    return best_model, best_model_name

def save_model(model, scaler, label_encoders, filepath):
    """Sauvegarder le modèle et les transformateurs"""
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoders': label_encoders
    }
    joblib.dump(model_data, filepath)

def load_model(filepath):
    """Charger le modèle et les transformateurs"""
    return joblib.load(filepath)

def extract_association_rules(df):
    """Extraire les règles d'association avec Apriori"""
    # Créer des variables binaires pour l'analyse
    binary_df = pd.DataFrame()
    
    # Discrétiser les variables continues
    binary_df['low_grades'] = (df['average_grade'] < df['average_grade'].quantile(0.33)).astype(int)
    binary_df['high_absenteeism'] = (df['absenteeism_rate'] > df['absenteeism_rate'].quantile(0.67)).astype(int)
    binary_df['low_engagement'] = (df['moodle_hours'] < df['moodle_hours'].quantile(0.33)).astype(int)
    binary_df['low_satisfaction'] = (df['satisfaction_score'] < df['satisfaction_score'].quantile(0.33)).astype(int)
    binary_df['dropout'] = (df['dropout'] == 'Yes').astype(int)
    
    # Appliquer l'algorithme Apriori
    try:
        frequent_itemsets = apriori(binary_df, min_support=0.1, use_colnames=True)
        if len(frequent_itemsets) > 0:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
            return rules
        else:
            return pd.DataFrame()
    except:
        return pd.DataFrame()

def get_feature_importance(model, feature_names):
    """Obtenir l'importance des features"""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        return importance_df
    return None