# 🎓 School Dropout Prevention System

This project uses **Data Mining** techniques to predict and prevent school dropout in universities. It combines machine learning, clustering, and association rules to identify at-risk students and provide personalized recommendations.

## 📋 Features

### 🔍 Exploratory Analysis
- Interactive visualizations of student data
- Global statistics on school dropout
- Correlation matrices and distribution plots

### 🎯 Student Clustering
- Identification of 4 student profiles using K-Means
- Detailed analysis of each cluster
- Specific recommendations per profile

### 🔮 Individual Prediction
- Interactive interface to assess a student's risk
- Real-time prediction with probabilities
- Generation of personalized recommendations

### 📊 Association Rules
- Pattern extraction using Apriori algorithm
- Identification of factors correlated with dropout
- Decision-making support rules

### 📥 Report Generation
- Detailed PDF reports for each student
- CSV export of data and recommendations
- Complete documentation of results

## 🚀 Installation and Usage

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Project Structure
```
├── data/
│   └── student_data.csv                 # Student data
├── models/
│   ├── model.pkl                        # Trained prediction model
│   └── clustering_model.pkl             # Clustering model
├── utils/
│   ├── preprocessing.py                 # Data preparation
│   ├── model_utils.py                   # ML utilities
│   ├── clustering.py                    # Clustering functions
│   └── recommender.py                   # Recommendation system
├── app/
│   ├── orchestration.py                 # Training pipeline
│   └── main.py                          # Streamlit application
├── requirements.txt                     # Dependencies
└── README.md
```

### Execution Steps

1. **Place the data**
```bash
# Copy your student_data.csv file to the data/ folder
cp student_data.csv data/
```

2. **Train the models**
```bash
python app/orchestration.py
```

3. **Launch the application**
```bash
streamlit run app/main.py
```

4. **Access the dashboard**
```
http://localhost:8501
```

## 🔧 Technologies Used

### Machine Learning
- **Scikit-learn**: Classification (Random Forest, XGBoost)
- **K-Means**: Student profile clustering
- **Apriori**: Association rules extraction

### Visualization
- **Streamlit**: Interactive web interface
- **Plotly**: Interactive charts
- **Seaborn/Matplotlib**: Statistical visualizations

### Report Generation
- **ReportLab**: PDF generation
- **Pandas**: CSV export

## 📊 Input Data

The system uses the following variables:

| Variable | Type | Description |
|----------|------|-------------|
| `age` | Numeric | Student's age |
| `gender` | Categorical | Gender (Male/Female) |
| `region` | Categorical | Region of origin |
| `parent_education` | Categorical | Parents' education level |
| `average_grade` | Numeric | Average grade (0-20) |
| `absenteeism_rate` | Numeric | Absenteeism rate (%) |
| `assignments_submitted` | Numeric | Percentage of assignments submitted |
| `moodle_hours` | Numeric | Hours spent on platform |
| `forum_posts` | Numeric | Number of forum posts |
| `satisfaction_score` | Numeric | Satisfaction score (1-10) |
| `dropout` | Categorical | Dropout status (Yes/No) |

## 🎯 Identified Student Profiles

### 🔴 Cluster 0: High-Risk Students
- Low grades, high absenteeism
- **Recommendations**: Enhanced monitoring, personalized tutoring

### 🟡 Cluster 1: Average Students
- Moderate performance, variable engagement
- **Recommendations**: Encouragement, method improvement

### 🟢 Cluster 2: High-Performing Students
- Good grades, high engagement
- **Recommendations**: Excellence maintenance, mentoring

### 🟠 Cluster 3: Disengaged Students
- Low participation, mediocre satisfaction
- **Recommendations**: Remotivation, career counseling

## 📈 Performance Metrics

The system evaluates models with:
- **Accuracy**: Overall prediction accuracy
- **Silhouette Score**: Clustering quality
- **Confidence/Support**: Association rules reliability

## 🛠️ Customization

### Adding New Variables
1. Modify `utils/preprocessing.py`
2. Update `get_feature_names()`
3. Retrain with `orchestration.py`

### Modifying Clusters
1. Adjust `n_clusters` in `clustering.py`
2. Update `get_cluster_profiles()`

### Customizing Recommendations
1. Modify `generate_recommendations()` in `recommender.py`
2. Add new business rules

## 🤝 Contributing

To contribute to the project:
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Open a Pull Request

## 📧 Contact

**Email:** gpidjakare@gmail.com  
**Phone:** +22870356451

---

*Developed with ❤️ for school dropout prevention*
