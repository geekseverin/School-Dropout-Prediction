import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

def perform_kmeans_clustering(X, n_clusters=4, sample_size=2000):
    """Effectuer le clustering K-Means avec gestion mémoire pour silhouette"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)

    # Gestion mémoire : silhouette_score sur un échantillon si X est trop grand
    if len(X) > sample_size:
        X_sample, clusters_sample = resample(X, clusters, n_samples=sample_size, random_state=42)
        silhouette_avg = silhouette_score(X_sample, clusters_sample)
    else:
        silhouette_avg = silhouette_score(X, clusters)

    return clusters, kmeans, silhouette_avg

def analyze_clusters(df, clusters):
    """Analyser les caractéristiques de chaque cluster"""
    df_clustered = df.copy()
    df_clustered['cluster'] = clusters

    cluster_analysis = {}

    for cluster_id in range(len(np.unique(clusters))):
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]

        analysis = {
            'size': len(cluster_data),
            'dropout_rate': cluster_data['dropout'].value_counts(normalize=True).get('Yes', 0),
            'avg_grade': cluster_data['average_grade'].mean(),
            'avg_absenteeism': cluster_data['absenteeism_rate'].mean(),
            'avg_satisfaction': cluster_data['satisfaction_score'].mean(),
            'avg_moodle_hours': cluster_data['moodle_hours'].mean(),
            'most_common_region': cluster_data['region'].mode().iloc[0] if not cluster_data['region'].mode().empty else 'Unknown'
        }

        cluster_analysis[f'Cluster {cluster_id}'] = analysis

    return cluster_analysis, df_clustered

def get_cluster_profiles():
    """Définir les profils types des clusters"""
    profiles = {
        0: {
            'name': 'Étudiants à Risque Élevé',
            'description': 'Notes faibles, absentéisme élevé, faible engagement',
            'recommendations': [
                'Suivi académique renforcé',
                'Tutorat personnalisé',
                'Améliorer l\'assiduité'
            ]
        },
        1: {
            'name': 'Étudiants Moyens',
            'description': 'Performance modérée, engagement variable',
            'recommendations': [
                'Encourager la participation',
                'Améliorer les méthodes d\'étude',
                'Suivi régulier'
            ]
        },
        2: {
            'name': 'Étudiants Performants',
            'description': 'Bonnes notes, engagement élevé, faible risque',
            'recommendations': [
                'Maintenir l\'excellence',
                'Opportunités d\'enrichissement',
                'Mentorat d\'autres étudiants'
            ]
        },
        3: {
            'name': 'Étudiants Désengagés',
            'description': 'Faible participation, satisfaction médiocre',
            'recommendations': [
                'Remotiver l\'engagement',
                'Améliorer l\'expérience étudiante',
                'Conseil d\'orientation'
            ]
        }
    }
    return profiles