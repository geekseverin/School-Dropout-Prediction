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

# Ajout d'imports pour les règles d'association
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
except ImportError:
    st.warning("mlxtend n'est pas installé. Installez-le avec: pip install mlxtend")
    apriori = None
    association_rules = None

def interpret_rule(antecedents, consequents):
    """
    Interprète une règle d'association en langage naturel
    """
    # Dictionnaire de traduction des termes techniques
    translations = {
        'Excellentes_Notes': 'avoir d\'excellentes notes',
        'Notes_Faibles': 'avoir des notes faibles',
        'Fort_Absentéisme': 'être très absent',
        'Faible_Absentéisme': 'être assidu',
        'Très_Satisfait': 'être très satisfait',
        'Peu_Satisfait': 'être peu satisfait',
        'Utilise_Beaucoup_Moodle': 'utiliser beaucoup Moodle',
        'Utilise_Peu_Moodle': 'utiliser peu Moodle',
        'Beaucoup_Devoirs_Rendus': 'rendre beaucoup de devoirs',
        'Peu_Devoirs_Rendus': 'rendre peu de devoirs',
        'Très_Actif_Forum': 'être très actif sur le forum',
        'Peu_Actif_Forum': 'être peu actif sur le forum',
        'Risque_Abandon': 'risquer d\'abandonner',
        'Étudiant_Persistant': 'persister dans ses études',
        'Étudiant_Masculin': 'être un étudiant masculin',
        'Étudiante_Féminine': 'être une étudiante féminine',
        'Parents_Éducation_Supérieure': 'avoir des parents avec éducation supérieure',
        'Parents_Sans_Éducation': 'avoir des parents sans éducation formelle',
        'Région_Lome': 'venir de Lomé',
        'Région_Notse': 'venir de Notsé',
        'Région_Tsevie': 'venir de Tsévié'
    }
    
    # Traduction des antécédents
    ant_translated = []
    for ant in antecedents:
        if ant in translations:
            ant_translated.append(translations[ant])
        else:
            ant_translated.append(ant.replace('_', ' ').lower())
    
    # Traduction des conséquents
    cons_translated = []
    for cons in consequents:
        if cons in translations:
            cons_translated.append(translations[cons])
        else:
            cons_translated.append(cons.replace('_', ' ').lower())
    
    # Construction de la phrase
    if len(ant_translated) == 1:
        condition = ant_translated[0]
    elif len(ant_translated) == 2:
        condition = f"{ant_translated[0]} ET {ant_translated[1]}"
    else:
        condition = f"{', '.join(ant_translated[:-1])} ET {ant_translated[-1]}"
    
    if len(cons_translated) == 1:
        result = cons_translated[0]
    else:
        result = f"{', '.join(cons_translated[:-1])} ET {cons_translated[-1]}"
    
    return f"Les étudiants qui tendent à {condition} ont tendance à {result}"

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
    .rule-card {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
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

def generate_association_rules_from_data(df):
    """
    Génère les règles d'association à partir des données avec des labels compréhensibles
    """
    if apriori is None or association_rules is None:
        return pd.DataFrame()
    
    try:
        # Création des variables binaires avec des noms plus explicites
        df_binary = pd.DataFrame()
        
        # Conversion des variables continues en catégories compréhensibles
        df_binary['Excellentes_Notes'] = (df['average_grade'] >= df['average_grade'].quantile(0.75)).astype(int)
        df_binary['Notes_Faibles'] = (df['average_grade'] <= df['average_grade'].quantile(0.25)).astype(int)
        df_binary['Fort_Absentéisme'] = (df['absenteeism_rate'] >= df['absenteeism_rate'].quantile(0.75)).astype(int)
        df_binary['Faible_Absentéisme'] = (df['absenteeism_rate'] <= df['absenteeism_rate'].quantile(0.25)).astype(int)
        df_binary['Très_Satisfait'] = (df['satisfaction_score'] >= df['satisfaction_score'].quantile(0.75)).astype(int)
        df_binary['Peu_Satisfait'] = (df['satisfaction_score'] <= df['satisfaction_score'].quantile(0.25)).astype(int)
        df_binary['Utilise_Beaucoup_Moodle'] = (df['moodle_hours'] >= df['moodle_hours'].quantile(0.75)).astype(int)
        df_binary['Utilise_Peu_Moodle'] = (df['moodle_hours'] <= df['moodle_hours'].quantile(0.25)).astype(int)
        df_binary['Beaucoup_Devoirs_Rendus'] = (df['assignments_submitted'] >= df['assignments_submitted'].quantile(0.75)).astype(int)
        df_binary['Peu_Devoirs_Rendus'] = (df['assignments_submitted'] <= df['assignments_submitted'].quantile(0.25)).astype(int)
        df_binary['Très_Actif_Forum'] = (df['forum_posts'] >= df['forum_posts'].quantile(0.75)).astype(int)
        df_binary['Peu_Actif_Forum'] = (df['forum_posts'] <= df['forum_posts'].quantile(0.25)).astype(int)
        df_binary['Risque_Abandon'] = (df['dropout'] == 'Yes').astype(int)
        df_binary['Étudiant_Persistant'] = (df['dropout'] == 'No').astype(int)
        
        # Variables catégorielles avec noms explicites
        if 'gender' in df.columns:
            df_binary['Étudiant_Masculin'] = (df['gender'] == 'Male').astype(int)
            df_binary['Étudiante_Féminine'] = (df['gender'] == 'Female').astype(int)
        
        if 'region' in df.columns:
            for region in df['region'].unique():
                df_binary[f'Région_{region}'] = (df['region'] == region).astype(int)
        
        if 'parent_education' in df.columns:
            df_binary['Parents_Éducation_Supérieure'] = (df['parent_education'] == 'Higher').astype(int)
            df_binary['Parents_Sans_Éducation'] = (df['parent_education'] == 'None').astype(int)
        
        # Génération des règles d'association
        frequent_itemsets = apriori(df_binary, min_support=0.05, use_colnames=True)
        
        if len(frequent_itemsets) > 0:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4)
            return rules
        else:
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Erreur lors de la génération des règles d'association: {e}")
        return pd.DataFrame()

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

    # 🎯 Section des histogrammes
    st.subheader("📈 Distributions des Variables")
    
    # Première ligne d'histogrammes
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribution des Notes par Statut")
        fig = px.box(df, x='dropout', y='average_grade', color='dropout', 
                     title="Notes vs Abandon", color_discrete_map={'Yes': '#ff4444', 'No': '#32CD32'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Distribution des Âges")
        fig = px.histogram(df, x='age', color='dropout', barmode='group',
                          title="Répartition des Âges par Statut",
                          color_discrete_map={'Yes': '#ff4444', 'No': '#32CD32'})
        st.plotly_chart(fig, use_container_width=True)

    # Deuxième ligne d'histogrammes
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Taux d'Absentéisme")
        fig = px.histogram(df, x='absenteeism_rate', color='dropout', barmode='group',
                          title="Distribution du Taux d'Absentéisme",
                          color_discrete_map={'Yes': '#ff4444', 'No': '#32CD32'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Devoirs Rendus")
        fig = px.histogram(df, x='assignments_submitted', color='dropout', barmode='group',
                          title="Distribution des Devoirs Rendus (%)",
                          color_discrete_map={'Yes': '#ff4444', 'No': '#32CD32'})
        st.plotly_chart(fig, use_container_width=True)

    # Troisième ligne d'histogrammes
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Utilisation de Moodle")
        fig = px.histogram(df, x='moodle_hours', color='dropout', barmode='group',
                          title="Heures d'Utilisation de Moodle",
                          color_discrete_map={'Yes': '#ff4444', 'No': '#32CD32'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Score de Satisfaction")
        fig = px.histogram(df, x='satisfaction_score', color='dropout', barmode='group',
                          title="Distribution du Score de Satisfaction",
                          color_discrete_map={'Yes': '#ff4444', 'No': '#32CD32'})
        st.plotly_chart(fig, use_container_width=True)

    # 📊 Variables catégorielles
    st.subheader("🔍 Analyse des Variables Catégorielles")
    
    col1, col2 = st.columns(2)
    with col1:
        if 'gender' in df.columns:
            st.subheader("Répartition par Genre")
            gender_dropout = df.groupby(['gender', 'dropout']).size().reset_index(name='count')
            fig = px.bar(gender_dropout, x='gender', y='count', color='dropout',
                        title="Abandon par Genre", barmode='group',
                        color_discrete_map={'Yes': '#ff4444', 'No': '#32CD32'})
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'region' in df.columns:
            st.subheader("Répartition par Région")
            region_dropout = df.groupby(['region', 'dropout']).size().reset_index(name='count')
            fig = px.bar(region_dropout, x='region', y='count', color='dropout',
                        title="Abandon par Région", barmode='group',
                        color_discrete_map={'Yes': '#ff4444', 'No': '#32CD32'})
            st.plotly_chart(fig, use_container_width=True)

    # 🎓 Éducation des parents
    if 'parent_education' in df.columns:
        st.subheader("Impact de l'Éducation des Parents")
        parent_ed_dropout = df.groupby(['parent_education', 'dropout']).size().reset_index(name='count')
        fig = px.bar(parent_ed_dropout, x='parent_education', y='count', color='dropout',
                    title="Abandon selon l'Éducation des Parents", barmode='group',
                    color_discrete_map={'Yes': '#ff4444', 'No': '#32CD32'})
        st.plotly_chart(fig, use_container_width=True)

    # 📈 Analyse des relations
    st.subheader("🔗 Relations entre Variables")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Absentéisme vs Notes")
        fig = px.scatter(df, x='absenteeism_rate', y='average_grade',
                         color='dropout', title="Relation Absentéisme-Notes",
                         color_discrete_map={'Yes': '#ff4444', 'No': '#32CD32'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Satisfaction vs Notes")
        fig = px.scatter(df, x='satisfaction_score', y='average_grade',
                         color='dropout', title="Relation Satisfaction-Notes",
                         color_discrete_map={'Yes': '#ff4444', 'No': '#32CD32'})
        st.plotly_chart(fig, use_container_width=True)

    # 📊 Matrice de corrélation
    st.subheader("🔍 Matrice de Corrélation")
    numeric_cols = ['age', 'average_grade', 'absenteeism_rate', 'assignments_submitted',
                    'moodle_hours', 'forum_posts', 'satisfaction_score']
    corr_matrix = df[numeric_cols].corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                    title="Corrélations entre Variables",
                    color_continuous_scale='RdBu_r')
    st.plotly_chart(fig, use_container_width=True)

    # 📈 Statistiques descriptives
    st.subheader("📋 Statistiques Descriptives")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Étudiants à Risque d'Abandon**")
        dropout_stats = df[df['dropout'] == 'Yes'][numeric_cols].describe()
        st.dataframe(dropout_stats.round(2))
    
    with col2:
        st.write("**Étudiants Persistants**")
        persistent_stats = df[df['dropout'] == 'No'][numeric_cols].describe()
        st.dataframe(persistent_stats.round(2))

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
        region = st.selectbox("Région", ["Lome", "Notse", "Tsevie","Kpalime", "Sokode", "Dapaong", "Atakpame", "Kara"])
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

def show_association_rules(clustering_data, df=None):
    st.header("🔗 Règles d'Association")
    
    # Debug: Afficher les clés disponibles dans clustering_data
    #st.subheader("🔍 Diagnostic des Données")
    #with st.expander("Voir les clés disponibles dans clustering_data"):
     #   st.write("Clés disponibles:", list(clustering_data.keys()))
    
    # Tentative de récupération des règles depuis le modèle sauvegardé
    rules = clustering_data.get('association_rules', pd.DataFrame())
    
    # Si pas de règles sauvegardées, les générer à partir des données
    if rules.empty and df is not None:
        st.info("🔄 Génération des règles d'association à partir des données...")
        rules = generate_association_rules_from_data(df)
    
    if not rules.empty:
        st.subheader("📋 Règles Découvertes")
        
        # Filtres pour les règles
        col1, col2, col3 = st.columns(3)
        with col1:
            min_confidence = st.slider("Confiance minimale", 0.0, 1.0, 0.5, 0.05)
        with col2:
            min_support = st.slider("Support minimal", 0.0, 1.0, 0.1, 0.05)
        with col3:
            min_lift = st.slider("Lift minimal", 0.0, 5.0, 1.0, 0.1)
        
        # Filtrer les règles
        filtered_rules = rules[
            (rules['confidence'] >= min_confidence) & 
            (rules['support'] >= min_support) & 
            (rules['lift'] >= min_lift)
        ].head(20)
        
        if not filtered_rules.empty:
            st.success(f"🎯 {len(filtered_rules)} règles trouvées avec les critères sélectionnés")
            
            # Interprétation des règles en langage naturel
            st.subheader("💡 Insights Découverts")
            
            # Top 5 des règles les plus importantes
            top_rules = filtered_rules.nlargest(5, 'confidence')
            
            for idx, rule in top_rules.iterrows():
                try:
                    # Gestion des antécédents et conséquents
                    if hasattr(rule['antecedents'], '__iter__') and not isinstance(rule['antecedents'], str):
                        antecedents = list(rule['antecedents'])
                    else:
                        antecedents = [str(rule['antecedents'])]
                    
                    if hasattr(rule['consequents'], '__iter__') and not isinstance(rule['consequents'], str):
                        consequents = list(rule['consequents'])
                    else:
                        consequents = [str(rule['consequents'])]
                    
                    # Interprétation en langage naturel
                    interpretation = interpret_rule(antecedents, consequents)
                    
                    # Déterminer la couleur selon la confiance
                    if rule['confidence'] >= 0.8:
                        icon = "🔴"
                        level = "CRITIQUE"
                        border_color = "#dc3545"
                    elif rule['confidence'] >= 0.6:
                        icon = "🟡"
                        level = "IMPORTANT"
                        border_color = "#ffc107"
                    else:
                        icon = "🟢"
                        level = "NOTABLE"
                        border_color = "#28a745"
                    
                    st.markdown(f"""
                    <div style="border-left: 4px solid {border_color}; background: #f8f9fa; padding: 1.5rem; margin: 1rem 0; border-radius: 5px;">
                        <h4>{icon} Insight {level}</h4>
                        <p style="font-size: 1.1em; color: #333;"><strong>{interpretation}</strong></p>
                        <div style="display: flex; gap: 20px; margin-top: 15px; font-size: 0.9em;">
                            <span style="background: #e9ecef; padding: 5px 10px; border-radius: 15px;">
                                <strong>Fiabilité:</strong> {rule['confidence']:.1%}
                            </span>
                            <span style="background: #e9ecef; padding: 5px 10px; border-radius: 15px;">
                                <strong>Fréquence:</strong> {rule['support']:.1%}
                            </span>
                            <span style="background: #e9ecef; padding: 5px 10px; border-radius: 15px;">
                                <strong>Force:</strong> {rule['lift']:.1f}x
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Erreur lors de l'interprétation de la règle {idx}: {e}")
                    continue
            
            # Graphique des règles
            st.subheader("📊 Visualisation des Règles")
            
            # Scatter plot Confiance vs Support
            fig = px.scatter(
                filtered_rules, 
                x='support', 
                y='confidence',
                size='lift',
                color='lift',
                title="Règles d'Association: Confiance vs Support",
                labels={'support': 'Support', 'confidence': 'Confiance', 'lift': 'Lift'},
                hover_data=['lift']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau des règles
            st.subheader("📋 Tableau des Règles")
            display_df = filtered_rules[['support', 'confidence', 'lift']].copy()
            display_df['antecedents'] = filtered_rules['antecedents'].astype(str)
            display_df['consequents'] = filtered_rules['consequents'].astype(str)
            display_df = display_df[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
            st.dataframe(display_df, use_container_width=True)
            
        else:
            st.warning("⚠️ Aucune règle ne correspond aux critères sélectionnés. Essayez de réduire les seuils.")
    else:
        st.info("ℹ️ Aucune règle d'association trouvée.")
        st.markdown("""
        **Causes possibles :**
        - Les règles n'ont pas été générées lors de l'entraînement
        - Les données ne contiennent pas suffisamment de patterns
        - Les seuils de support/confiance sont trop élevés
        
        **Solutions :**
        1. Vérifiez que mlxtend est installé: `pip install mlxtend`
        2. Réexécutez le script d'entraînement avec génération des règles
        3. Utilisez le bouton ci-dessous pour générer les règles maintenant
        """)
        
        if df is not None and st.button("🔄 Générer les règles d'association maintenant"):
            with st.spinner("Génération en cours..."):
                rules = generate_association_rules_from_data(df)
                if not rules.empty:
                    st.success("✅ Règles générées avec succès!")
                    st.rerun()
                else:
                    st.error("❌ Impossible de générer les règles d'association")

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
        show_association_rules(clustering_data, df)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Statistiques du Modèle")
    if df is not None:
        st.sidebar.metric("Total Étudiants", len(df))
        st.sidebar.metric("Taux d'Abandon", f"{(df['dropout'] == 'Yes').mean():.1%}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("*Développé avec ❤️ par Gnabana Séverin PIDJAKARE*")
    st.sidebar.markdown("*Email : gpidjakare@gmail.com*")
    st.sidebar.markdown("*Tel : +22870356451*")

if __name__ == "__main__":
    main()