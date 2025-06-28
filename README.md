
# 🎓 Système de Prévention de l'Abandon Scolaire

Ce projet utilise des techniques de **Data Mining** pour prédire et prévenir l'abandon scolaire dans les universités. Il combine l'apprentissage automatique, le clustering et les règles d'association pour identifier les étudiants à risque et proposer des recommandations personnalisées.

## 📋 Fonctionnalités

### 🔍 Analyse Exploratoire
- Visualisations interactives des données étudiantes
- Statistiques globales sur l'abandon scolaire
- Matrices de corrélation et graphiques de distribution

### 🎯 Clustering des Étudiants
- Identification de 4 profils d'étudiants via K-Means
- Analyse détaillée de chaque cluster
- Recommandations spécifiques par profil

### 🔮 Prédiction Individuelle
- Interface interactive pour évaluer le risque d'un étudiant
- Prédiction en temps réel avec probabilités
- Génération de recommandations personnalisées

### 📊 Règles d'Association
- Extraction de patterns avec l'algorithme Apriori
- Identification des facteurs corrélés à l'abandon
- Règles d'aide à la décision

### 📥 Génération de Rapports
- Rapports PDF détaillés pour chaque étudiant
- Export CSV des données et recommandations
- Documentation complète des résultats

## 🚀 Installation et Utilisation

### Prérequis
```bash
Python 3.8+
pip install -r requirements.txt
```

### Structure du Projet
```
├── data/
│   └── student_data.csv                 # Données des étudiants
├── models/
│   ├── model.pkl                        # Modèle de prédiction entraîné
│   └── clustering_model.pkl             # Modèle de clustering
├── utils/
│   ├── preprocessing.py                 # Préparation des données
│   ├── model_utils.py                   # Utilitaires ML
│   ├── clustering.py                    # Fonctions de clustering
│   └── recommender.py                   # Système de recommandations
├── app/
│   ├── orchestration.py                 # Pipeline d'entraînement
│   └── main.py                          # Application Streamlit
├── requirements.txt                     # Dépendances
└── README.md
```

### Étapes d'Exécution

1. **Placer les données**
```bash
# Copier votre fichier student_data.csv dans le dossier data/
cp student_data.csv data/
```

2. **Entraîner les modèles**
```bash
python app/orchestration.py
```

3. **Lancer l'application**
```bash
streamlit run app/main.py
```

4. **Accéder au dashboard**
```
http://localhost:8501
```

## 🔧 Technologies Utilisées

### Machine Learning
- **Scikit-learn** : Classification (Random Forest, XGBoost)
- **K-Means** : Clustering des profils étudiants
- **Apriori** : Extraction de règles d'association

### Visualisation
- **Streamlit** : Interface web interactive
- **Plotly** : Graphiques interactifs
- **Seaborn/Matplotlib** : Visualisations statistiques

### Génération de Rapports
- **ReportLab** : Génération de PDF
- **Pandas** : Export CSV

## 📊 Données d'Entrée

Le système utilise les variables suivantes :

| Variable | Type | Description |
|----------|------|-------------|
| `age` | Numérique | Âge de l'étudiant |
| `gender` | Catégorielle | Genre (Male/Female) |
| `region` | Catégorielle | Région d'origine |
| `parent_education` | Catégorielle | Niveau d'éducation des parents |
| `average_grade` | Numérique | Note moyenne (0-20) |
| `absenteeism_rate` | Numérique | Taux d'absentéisme (%) |
| `assignments_submitted` | Numérique | Pourcentage de devoirs rendus |
| `moodle_hours` | Numérique | Heures passées sur la plateforme |
| `forum_posts` | Numérique | Nombre de posts sur le forum |
| `satisfaction_score` | Numérique | Score de satisfaction (1-10) |
| `dropout` | Catégorielle | Statut d'abandon (Yes/No) |

## 🎯 Profils d'Étudiants Identifiés

### 🔴 Cluster 0 : Étudiants à Risque Élevé
- Notes faibles, absentéisme élevé
- **Recommandations** : Suivi renforcé, tutorat personnalisé

### 🟡 Cluster 1 : Étudiants Moyens
- Performance modérée, engagement variable
- **Recommandations** : Encouragement, amélioration des méthodes

### 🟢 Cluster 2 : Étudiants Performants
- Bonnes notes, engagement élevé
- **Recommandations** : Maintien de l'excellence, mentorat

### 🟠 Cluster 3 : Étudiants Désengagés
- Faible participation, satisfaction médiocre
- **Recommandations** : Remotivation, conseil d'orientation

## 📈 Métriques de Performance

Le système évalue les modèles avec :
- **Accuracy** : Précision globale de la prédiction
- **Silhouette Score** : Qualité du clustering
- **Confidence/Support** : Fiabilité des règles d'association

## 🛠️ Personnalisation

### Ajouter de Nouvelles Variables
1. Modifier `utils/preprocessing.py`
2. Mettre à jour `get_feature_names()`
3. Réentraîner avec `orchestration.py`

### Modifier les Clusters
1. Ajuster `n_clusters` dans `clustering.py`
2. Mettre à jour `get_cluster_profiles()`

### Personnaliser les Recommandations
1. Modifier `generate_recommendations()` dans `recommender.py`
2. Ajouter de nouvelles règles métier


## 🤝 Contribution

Pour contribuer au projet :
1. Fork le repository
2. Créer une branche feature
3. Commit les modifications
4. Ouvrir une Pull Request

## 📧 Contact

**Email  :** gpidjakare@gmail.com  
**Telephone :** +22870356451


---

*Développé avec ❤️ pour la prévention de l'abandon scolaire*
