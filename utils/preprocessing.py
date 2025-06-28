import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_clean_data(file_path):
    """Charger et nettoyer les données"""
    df = pd.read_csv(file_path)
    
    # Vérifier les valeurs manquantes
    print(f"Valeurs manquantes par colonne:\n{df.isnull().sum()}")
    
    # Remplacer les valeurs manquantes si nécessaire
    df = df.fillna(df.median(numeric_only=True))
    
    return df

def encode_categorical_features(df):
    """Encoder les variables catégorielles"""
    le_dict = {}
    categorical_cols = ['gender', 'region', 'parent_education', 'dropout']
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col])
            le_dict[col] = le
    
    return df, le_dict

def prepare_features(df):
    """Préparer les features pour le modèle"""
    # Colonnes à utiliser pour la prédiction (sans dropout)
    feature_cols = ['age', 'gender_encoded', 'region_encoded', 'parent_education_encoded',
                   'average_grade', 'absenteeism_rate', 'assignments_submitted',
                   'moodle_hours', 'forum_posts', 'satisfaction_score']
    
    X = df[feature_cols]
    y = df['dropout_encoded'] if 'dropout_encoded' in df.columns else None
    
    # Normaliser les features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    
    return X_scaled, y, scaler

def get_feature_names():
    """Retourner les noms des features"""
    return ['age', 'gender_encoded', 'region_encoded', 'parent_education_encoded',
            'average_grade', 'absenteeism_rate', 'assignments_submitted',
            'moodle_hours', 'forum_posts', 'satisfaction_score']