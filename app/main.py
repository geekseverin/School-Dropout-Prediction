import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.preprocessing import LabelEncoder
from utils.model_utils import load_model
from utils.recommender import generate_recommendations, generate_student_report, create_csv_report
from utils.clustering import get_cluster_profiles

# 📌 Fonction utilitaire pour encodage sécurisé
def safe_label_encode(le, values):
    """Encodeur qui gère les labels inconnus"""
    values = values.fillna('UNKNOWN').astype(str)
    unseen = set(values.unique()) - set(le.classes_)
    if unseen:
        le.classes_ = np.concatenate([le.classes_, list(unseen)])
    return le.transform(values)

# Configuration de la page
st.set_page_config(
    page_title="Prévention Abandon Scolaire",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .cluster-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/student_data.csv')
        df_clustered = pd.read_csv('data/student_data_clustered.csv')
        return df, df_clustered
    except FileNotFoundError:
        st.error("Fichiers de données non trouvés. Veuillez exécuter orchestration.py d'abord.")
        return None, None

@st.cache_resource
def load_models():
    try:
        model_data = load_model('models/model.pkl')
        clustering_data = joblib.load('models/clustering_model.pkl')
        return model_data, clustering_data
    except FileNotFoundError:
        st.error("Modèles non trouvés. Veuillez exécuter orchestration.py d'abord.")
        return None, None

def show_exploratory_analysis(df, df_clustered):
    st.header("📊 Analyse Exploratoire des Données")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(df)}</h3>
            <p>Total Étudiants</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        dropout_rate = (df['dropout'] == 'Yes').mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>{dropout_rate:.1%}</h3>
            <p>Taux d'Abandon</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        avg_grade = df['average_grade'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>{avg_grade:.1f}</h3>
            <p>Note Moyenne</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        avg_satisfaction = df['satisfaction_score'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>{avg_satisfaction:.1f}</h3>
            <p>Satisfaction Moyenne</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribution des Notes par Statut")
        fig = px.box(df, x='dropout', y='average_grade', color='dropout', title="Notes vs Abandon")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Absentéisme vs Abandon")
        fig = px.scatter(df, x='absenteeism_rate', y='average_grade',
                         color='dropout', title="Absentéisme vs Notes")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Matrice de Corrélation")
    numeric_cols = ['age', 'average_grade', 'absenteeism_rate', 'assignments_submitted',
                    'moodle_hours', 'forum_posts', 'satisfaction_score']
    corr_matrix = df[numeric_cols].corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Corrélations entre Variables")
    st.plotly_chart(fig, use_container_width=True)

def show_clustering_analysis(df_clustered, clustering_data):
    st.header("🎯 Analyse des Profils d'Étudiants")
    cluster_analysis = clustering_data['cluster_analysis']

    st.subheader("Vue d'ensemble des Clusters")
    cluster_summary = []
    for cluster_name, analysis in cluster_analysis.items():
        cluster_summary.append({
            'Cluster': cluster_name,
            'Taille': analysis['size'],
            'Taux d\'abandon': f"{analysis['dropout_rate']:.1%}",
            'Note moyenne': f"{analysis['avg_grade']:.1f}",
            'Absentéisme': f"{analysis['avg_absenteeism']:.1f}%",
            'Satisfaction': f"{analysis['avg_satisfaction']:.1f}"
        })

    summary_df = pd.DataFrame(cluster_summary)
    st.dataframe(summary_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribution des Clusters")
        cluster_counts = df_clustered['cluster'].value_counts().sort_index()
        fig = px.pie(values=cluster_counts.values, 
                     names=[f'Cluster {i}' for i in cluster_counts.index],
                     title="Répartition des Étudiants par Cluster")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Taux d'Abandon par Cluster")
        dropout_by_cluster = df_clustered.groupby('cluster')['dropout'].apply(
            lambda x: (x == 'Yes').mean())
        fig = px.bar(x=dropout_by_cluster.index, y=dropout_by_cluster.values,
                     title="Taux d'Abandon par Cluster",
                     labels={'x': 'Cluster', 'y': 'Taux d\'abandon'})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Profils Détaillés des Clusters")
    profiles = get_cluster_profiles()
    for i in range(4):
        if i in profiles:
            with st.expander(f"📋 {profiles[i]['name']} (Cluster {i})"):
                st.write(f"**Description:** {profiles[i]['description']}")
                st.write("**Recommandations :**")
                for rec in profiles[i]['recommendations']:
                    st.write(f"• {rec}")

def show_prediction_interface(model_data, clustering_data):
    st.header("🔮 Simulation de Prédiction Individuelle")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Informations de l'Étudiant")
        age = st.slider("Âge", 18, 40, 25)
        gender = st.selectbox("Genre", ["Male", "Female"])
        region = st.selectbox("Région", ["Lome", "Notse", "Tsevie"])
        parent_education = st.selectbox("Éducation des Parents", ["None", "Primary", "Secondary", "Higher"])
        average_grade = st.slider("Note Moyenne", 0.0, 20.0, 12.0, 0.1)
        absenteeism_rate = st.slider("Taux d'Absentéisme (%)", 0.0, 50.0, 15.0, 0.1)
        assignments_submitted = st.slider("Devoirs Rendus (%)", 0.0, 100.0, 80.0, 0.1)
        moodle_hours = st.slider("Heures sur Moodle", 0.0, 50.0, 10.0, 0.1)
        forum_posts = st.slider("Posts Forum", 0, 20, 3)
        satisfaction_score = st.slider("Score de Satisfaction", 1.0, 10.0, 7.0, 0.1)

    with col2:
        st.subheader("Résultats de la Prédiction")

        if st.button("🚀 Prédire le Risque", type="primary"):
            student_data = {
                'age': age,
                'gender': gender,
                'region': region,
                'parent_education': parent_education,
                'average_grade': average_grade,
                'absenteeism_rate': absenteeism_rate,
                'assignments_submitted': assignments_submitted,
                'moodle_hours': moodle_hours,
                'forum_posts': forum_posts,
                'satisfaction_score': satisfaction_score
            }

            model = model_data['model']
            scaler = model_data['scaler']
            label_encoders = model_data['label_encoders']

            # Créer un DataFrame temporaire avec les bonnes colonnes
            temp_df = pd.DataFrame([student_data])

            # Encoder les variables catégorielles
            for col, le in label_encoders.items():
                if col in temp_df.columns and col != 'dropout':
                    temp_df[col + '_encoded'] = safe_label_encode(le, temp_df[col])

            # Préparer les features avec les bons noms de colonnes
            feature_cols = ['age', 'gender_encoded', 'region_encoded', 'parent_education_encoded',
                            'average_grade', 'absenteeism_rate', 'assignments_submitted',
                            'moodle_hours', 'forum_posts', 'satisfaction_score']
            
            # CORRECTION : Créer un DataFrame avec les noms de colonnes pour éviter l'erreur
            X_pred = temp_df[feature_cols].copy()
            X_pred_scaled = scaler.transform(X_pred)
            
            # Créer un DataFrame avec les mêmes noms de colonnes pour la prédiction
            X_pred_scaled_df = pd.DataFrame(X_pred_scaled, columns=feature_cols)

            # Prédictions avec les DataFrames ayant les bons noms de colonnes
            try:
                dropout_probability = model.predict_proba(X_pred_scaled_df)[0][1]
                prediction = model.predict(X_pred_scaled_df)[0]
            except Exception as e:
                st.error(f"Erreur lors de la prédiction du modèle principal: {e}")
                # Fallback avec array numpy
                dropout_probability = model.predict_proba(X_pred_scaled)[0][1]
                prediction = model.predict(X_pred_scaled)[0]

            # Prédiction du cluster
            kmeans_model = clustering_data['kmeans_model']
            try:
                cluster = kmeans_model.predict(X_pred_scaled_df)[0]
            except Exception as e:
                st.warning(f"Utilisation de l'array numpy pour le clustering: {e}")
                cluster = kmeans_model.predict(X_pred_scaled)[0]

            risk_level = "🔴 ÉLEVÉ" if dropout_probability > 0.7 else "🟡 MODÉRÉ" if dropout_probability > 0.5 else "🟢 FAIBLE"

            st.markdown(f"""
            ### 📊 Résultats de l'Analyse
            **Probabilité d'abandon:** {dropout_probability:.1%}  
            **Niveau de risque:** {risk_level}  
            **Cluster identifié:** Cluster {cluster}
            """)

            recommendations = generate_recommendations(student_data, cluster, dropout_probability)

            st.markdown("### 💡 Recommandations Personnalisées")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")

            st.markdown("### 📥 Télécharger le Rapport")
            
            # Générer automatiquement les rapports après la prédiction
            try:
                # Génération du PDF
                pdf_buffer = generate_student_report(student_data, cluster, dropout_probability, recommendations)
                
                # Génération du CSV
                csv_df = create_csv_report(student_data, cluster, dropout_probability, recommendations)
                csv_string = csv_df.to_csv(index=False)
                
                col_pdf, col_csv = st.columns(2)
                
                with col_pdf:
                    st.download_button(
                        label="📄 Télécharger le rapport PDF",
                        data=pdf_buffer.getvalue(),
                        file_name=f"rapport_etudiant_{age}ans.pdf",
                        mime="application/pdf",
                        key="download_pdf"
                    )

                with col_csv:
                    st.download_button(
                        label="📊 Télécharger le rapport CSV",
                        data=csv_string,
                        file_name=f"rapport_etudiant_{age}ans.csv",
                        mime="text/csv",
                        key="download_csv"
                    )
                    
                st.success("✅ Rapports générés avec succès! Vous pouvez maintenant les télécharger.")
                
            except Exception as e:
                st.error(f"❌ Erreur lors de la génération des rapports: {e}")
                st.info("💡 Vérifiez que les fonctions `generate_student_report` et `create_csv_report` sont correctement implémentées dans le module `utils.recommender`.")

def show_association_rules(clustering_data):
    st.header("🔗 Règles d'Association")
    rules = clustering_data.get('association_rules', pd.DataFrame())
    if not rules.empty:
        st.subheader("Règles Découvertes")
        important_rules = rules[rules['confidence'] > 0.7].head(10)
        for idx, rule in important_rules.iterrows():
            antecedents = ', '.join(list(rule['antecedents']))
            consequents = ', '.join(list(rule['consequents']))
            st.markdown(f"""
            **Règle {idx + 1}:**  
            Si `{antecedents}` ➜ alors `{consequents}`  
            - **Confiance:** {rule['confidence']:.2%}  
            - **Support:** {rule['support']:.2%}  
            - **Lift:** {rule['lift']:.2f}
            """)
            st.markdown("---")
    else:
        st.info("Aucune règle d'association significative trouvée.")

def main():
    st.markdown('<h1 class="main-header">🎓 Système de Prévention de l\'Abandon Scolaire</h1>', unsafe_allow_html=True)
    
    # Chargement des données et modèles
    df, df_clustered = load_data()
    model_data, clustering_data = load_models()
    
    if df is None or model_data is None:
        st.stop()

    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox("Choisir une section:",
                                ["📊 Analyse Exploratoire", "🎯 Analyse des Clusters",
                                 "🔮 Prédiction Individuelle", "🔗 Règles d'Association"])

    if page == "📊 Analyse Exploratoire":
        show_exploratory_analysis(df, df_clustered)
    elif page == "🎯 Analyse des Clusters":
        show_clustering_analysis(df_clustered, clustering_data)
    elif page == "🔮 Prédiction Individuelle":
        show_prediction_interface(model_data, clustering_data)
    elif page == "🔗 Règles d'Association":
        show_association_rules(clustering_data)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Statistiques du Modèle")
    if df is not None:
        st.sidebar.metric("Total Étudiants", len(df))
        st.sidebar.metric("Taux d'Abandon", f"{(df['dropout'] == 'Yes').mean():.1%}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("*Développé avec ❤️ par l'équipe Data Science*")

if __name__ == "__main__":
    main()