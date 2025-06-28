import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from utils.preprocessing import load_and_clean_data, encode_categorical_features, prepare_features
from utils.model_utils import train_models, select_best_model, save_model, extract_association_rules
from utils.clustering import perform_kmeans_clustering, analyze_clusters

def run_full_pipeline():
    """Exécuter le pipeline complet d'entraînement"""
    
    print("🚀 Démarrage du pipeline d'entraînement...")
    
    # 1. Charger et préparer les données
    print("📊 Chargement des données...")
    df = load_and_clean_data('data/student_data.csv')
    
    # 2. Encoder les variables catégorielles
    print("🔄 Encodage des variables...")
    df_encoded, label_encoders = encode_categorical_features(df)
    
    # 3. Préparer les features
    print("⚙️ Préparation des features...")
    X, y, scaler = prepare_features(df_encoded)
    
    # 4. Entraîner les modèles
    print("🤖 Entraînement des modèles...")
    models, X_test, y_test = train_models(X, y)
    
    # 5. Sélectionner le meilleur modèle
    print("🏆 Sélection du meilleur modèle...")
    best_model, best_model_name = select_best_model(models)
    print(f"Meilleur modèle: {best_model_name}")
    print(f"Accuracy: {models[best_model_name]['accuracy']:.3f}")
    
    # 6. Effectuer le clustering
    print("🎯 Clustering des étudiants...")
    clusters, kmeans_model, silhouette_score = perform_kmeans_clustering(X)
    cluster_analysis, df_clustered = analyze_clusters(df, clusters)
    
    print(f"Score de silhouette: {silhouette_score:.3f}")
    
    # 7. Extraire les règles d'association
    print("📝 Extraction des règles d'association...")
    association_rules = extract_association_rules(df)
    
    # 8. Sauvegarder le modèle principal
    print("💾 Sauvegarde du modèle...")
    save_model(best_model, scaler, label_encoders, 'models/model.pkl')
    
    # 9. Sauvegarder le modèle de clustering
    import joblib
    clustering_data = {
        'kmeans_model': kmeans_model,
        'cluster_analysis': cluster_analysis,
        'association_rules': association_rules
    }
    joblib.dump(clustering_data, 'models/clustering_model.pkl')
    
    # 10. Sauvegarder les données clustérisées
    df_clustered.to_csv('data/student_data_clustered.csv', index=False)
    
    print("✅ Pipeline terminé avec succès!")
    print(f"📁 Modèle sauvegardé dans: models/model.pkl")
    print(f"📁 Clustering sauvegardé dans: models/clustering_model.pkl")
    
    # Afficher un résumé
    print("\n📈 RÉSUMÉ DE L'ANALYSE:")
    print(f"- Nombre d'étudiants: {len(df)}")
    print(f"- Taux d'abandon global: {(df['dropout'] == 'Yes').mean():.1%}")
    print(f"- Nombre de clusters: {len(cluster_analysis)}")
    print(f"- Précision du modèle: {models[best_model_name]['accuracy']:.1%}")
    
    return True

if __name__ == "__main__":
    # Créer les dossiers nécessaires
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    run_full_pipeline()