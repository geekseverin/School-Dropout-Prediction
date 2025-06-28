import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def fill_categorical_missing_values(df, categorical_cols):
    """
    Combler les valeurs manquantes des variables catégorielles 
    par la valeur la plus fréquente (mode)
    """
    df_filled = df.copy()
    
    for col in categorical_cols:
        if col in df_filled.columns:
            # Vérifier s'il y a des valeurs manquantes
            missing_count = df_filled[col].isnull().sum()
            if missing_count > 0:
                # Trouver la valeur la plus fréquente (mode)
                most_frequent = df_filled[col].mode()[0]
                
                # Remplacer les valeurs manquantes
                df_filled[col].fillna(most_frequent, inplace=True)
                
                print(f"Colonne '{col}': {missing_count} valeurs manquantes remplacées par '{most_frequent}'")
    
    return df_filled

def load_and_clean_data(file_path):
    """Charger et nettoyer les données"""
    df = pd.read_csv(file_path)
    
    # Vérifier les valeurs manquantes avant nettoyage
    print("=== AVANT NETTOYAGE ===")
    print(f"Valeurs manquantes par colonne:\n{df.isnull().sum()}")
    
    # Définir les colonnes catégorielles
    categorical_cols = ['gender', 'region', 'parent_education', 'dropout']
    
    # Combler les valeurs manquantes des variables catégorielles
    print("\n=== TRAITEMENT DES VARIABLES CATÉGORIELLES ===")
    df = fill_categorical_missing_values(df, categorical_cols)
    
    # Combler les valeurs manquantes des variables numériques avec la médiane
    numeric_missing = df.select_dtypes(include=[np.number]).isnull().sum()
    if numeric_missing.sum() > 0:
        print(f"\n=== TRAITEMENT DES VARIABLES NUMÉRIQUES ===")
        print(f"Valeurs manquantes numériques avant: \n{numeric_missing[numeric_missing > 0]}")
        df = df.fillna(df.median(numeric_only=True))
        print("Variables numériques: valeurs manquantes remplacées par la médiane")
    
    # Vérifier les valeurs manquantes après nettoyage
    print(f"\n=== APRÈS NETTOYAGE ===")
    print(f"Valeurs manquantes par colonne:\n{df.isnull().sum()}")
    
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
            
            # Afficher les mappings pour information
            unique_values = df[col].unique()
            encoded_values = le.transform(unique_values)
            mapping = dict(zip(unique_values, encoded_values))
            print(f"Mapping pour '{col}': {mapping}")
    
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

