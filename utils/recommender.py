import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import io

def generate_recommendations(student_data, cluster, dropout_probability):
    """Générer des recommandations personnalisées"""
    recommendations = []
    
    # Recommandations basées sur le cluster
    cluster_recs = {
        0: [
            "📚 Suivi académique renforcé recommandé",
            "👨‍🏫 Tutorat personnalisé nécessaire", 
            "📅 Améliorer l'assiduité aux cours"
        ],
        1: [
            "🎯 Encourager la participation active",
            "📖 Améliorer les méthodes d'étude",
            "🔍 Suivi régulier des progrès"
        ],
        2: [
            "⭐ Maintenir l'excellence académique",
            "🚀 Opportunités d'enrichissement disponibles",
            "🤝 Possibilité de mentorat d'autres étudiants"
        ],
        3: [
            "💪 Remotiver l'engagement étudiant",
            "🌟 Améliorer l'expérience étudiante",
            "🧭 Conseil d'orientation recommandé"
        ]
    }
    
    # Ajouter les recommandations du cluster
    if cluster in cluster_recs:
        recommendations.extend(cluster_recs[cluster])
    
    # Recommandations basées sur les données individuelles
    if student_data['average_grade'] < 12:
        recommendations.append("📈 Amélioration des notes prioritaire")
    
    if student_data['absenteeism_rate'] > 20:
        recommendations.append("⏰ Réduire l'absentéisme")
    
    if student_data['moodle_hours'] < 5:
        recommendations.append("💻 Augmenter l'utilisation de la plateforme")
    
    if student_data['forum_posts'] < 2:
        recommendations.append("💬 Participer davantage aux discussions")
    
    if student_data['satisfaction_score'] < 6:
        recommendations.append("😊 Améliorer la satisfaction étudiante")
    
    # Recommandations basées sur le risque
    if dropout_probability > 0.7:
        recommendations.append("🚨 INTERVENTION URGENTE NÉCESSAIRE")
    elif dropout_probability > 0.5:
        recommendations.append("⚠️ Suivi rapproché recommandé")
    
    return recommendations

def generate_student_report(student_data, cluster, dropout_probability, recommendations):
    """Générer un rapport PDF pour l'étudiant"""
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Titre
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        textColor=colors.darkblue,
        alignment=1  # Center
    )
    
    story.append(Paragraph("RAPPORT D'ANALYSE - RISQUE D'ABANDON SCOLAIRE", title_style))
    story.append(Spacer(1, 20))
    
    # Informations de l'étudiant
    story.append(Paragraph("INFORMATIONS DE L'ÉTUDIANT", styles['Heading2']))
    
    student_info = [
        ['Âge', f"{student_data['age']} ans"],
        ['Genre', student_data['gender']],
        ['Région', student_data['region']],
        ['Éducation des parents', student_data['parent_education']],
        ['Note moyenne', f"{student_data['average_grade']:.2f}/20"],
        ['Taux d\'absentéisme', f"{student_data['absenteeism_rate']:.1f}%"],
        ['Devoirs rendus', f"{student_data['assignments_submitted']:.1f}%"],
        ['Heures sur Moodle', f"{student_data['moodle_hours']:.1f}h"],
        ['Posts forum', f"{student_data['forum_posts']}"],
        ['Score satisfaction', f"{student_data['satisfaction_score']:.1f}/10"]
    ]
    
    table = Table(student_info, colWidths=[3*72, 2*72])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(table)
    story.append(Spacer(1, 20))
    
    # Résultats de l'analyse
    story.append(Paragraph("RÉSULTATS DE L'ANALYSE", styles['Heading2']))
    
    # Risque d'abandon
    risk_color = colors.red if dropout_probability > 0.7 else colors.orange if dropout_probability > 0.5 else colors.green
    risk_text = f"<font color='{risk_color}'>Probabilité d'abandon: {dropout_probability:.1%}</font>"
    story.append(Paragraph(risk_text, styles['Normal']))
    story.append(Spacer(1, 10))
    
    # Profil de l'étudiant
    cluster_names = {
        0: "Étudiant à Risque Élevé",
        1: "Étudiant Moyen", 
        2: "Étudiant Performant",
        3: "Étudiant Désengagé"
    }
    
    cluster_name = cluster_names.get(cluster, f"Cluster {cluster}")
    story.append(Paragraph(f"<b>Profil identifié:</b> {cluster_name}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Recommandations
    story.append(Paragraph("RECOMMANDATIONS PERSONNALISÉES", styles['Heading2']))
    
    for i, rec in enumerate(recommendations, 1):
        story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
        story.append(Spacer(1, 5))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def create_csv_report(student_data, cluster, dropout_probability, recommendations):
    """Créer un rapport CSV"""
    report_data = {
        'Metric': ['Age', 'Gender', 'Region', 'Parent_Education', 'Average_Grade',
                  'Absenteeism_Rate', 'Assignments_Submitted', 'Moodle_Hours',
                  'Forum_Posts', 'Satisfaction_Score', 'Cluster', 'Dropout_Probability'],
        'Value': [
            student_data['age'],
            student_data['gender'], 
            student_data['region'],
            student_data['parent_education'],
            student_data['average_grade'],
            student_data['absenteeism_rate'],
            student_data['assignments_submitted'],
            student_data['moodle_hours'],
            student_data['forum_posts'],
            student_data['satisfaction_score'],
            cluster,
            f"{dropout_probability:.3f}"
        ]
    }
    
    df = pd.DataFrame(report_data)
    
    # Ajouter les recommandations
    rec_df = pd.DataFrame({
        'Metric': [f'Recommendation_{i+1}' for i in range(len(recommendations))],
        'Value': recommendations
    })
    
    final_df = pd.concat([df, rec_df], ignore_index=True)
    return final_df