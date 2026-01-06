# Heart Disease Prediction Dashboard

A professional, production-quality Streamlit dashboard for heart disease risk assessment and prediction using machine learning. Built for presentation at medical and data science conferences.

## 🎯 Overview

This dashboard provides an interactive platform for:
- **Data Exploration**: Comprehensive exploratory data analysis with interactive visualizations
- **Model Training & Evaluation**: Multiple ML models with detailed performance metrics
- **Risk Assessment**: Real-time patient risk prediction using trained models
- **Clinical Insights**: Automated insights based on data patterns and model findings
- **Data Export**: Download filtered data and predictions for further analysis

## 📊 Features

### Data Handling
- Automatic data loading and preprocessing from `heart_disease.csv`
- Intelligent missing value handling (imputation with median/mode)
- Risk categorization based on clinical factors
- Complete data dictionary with clinical descriptions

### Visualizations (15+ Interactive Charts)
- **Funnel Chart**: Patient progression through risk stages
- **Correlation Heatmap**: Feature relationships and dependencies
- **ROC & Precision-Recall Curves**: Model performance evaluation
- **Confusion Matrix**: Classification accuracy visualization
- **3D Scatter Plot**: Multi-dimensional feature analysis
- **Feature Importance Charts**: Model interpretability
- **Categorical Analysis**: Stacked bar charts for categorical variables
- **Distribution Plots**: Histograms and KDE plots with target overlays
- **Box Plots & Pairplots**: Statistical distributions and relationships

### Machine Learning Models
- **Logistic Regression**: Linear model with interpretable coefficients
- **Random Forest**: Ensemble method with feature importance
- Train/test split with proper scaling and validation
- Comprehensive metrics: Accuracy, Precision, Recall, Specificity, F1, AUC

### Interactive Features
- **Sidebar Filters**: Age, sex, chest pain type, heart rate, cholesterol, blood pressure
- **Theme Toggle**: Light/dark mode switching
- **Model Selection**: Switch between trained models
- **Interactive Prediction Form**: Real-time patient risk assessment
- **Customizable Visualizations**: Select features for dynamic analysis
- **Data Export**: Download CSV files with filtered data and predictions

## 🏗️ Project Structure

```
Heart_Disease_Prediction/
├── app.py                      # Main Streamlit application
├── heart_disease.csv           # Dataset (local)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .streamlit/
│   └── config.toml            # Streamlit configuration
└── utils/
    ├── data_loader.py         # Data loading and preprocessing
    ├── model.py               # ML models and training
    └── visuals.py             # Chart generation functions
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Local Installation

1. **Clone or download the project**
```bash
cd "Heart Disease"
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Ensure heart_disease.csv is in the project root**
The dataset should be located at the same level as `app.py`

### Running Locally

```bash
streamlit run app.py
```

The dashboard will open in your default browser at `http://localhost:8501`

## 📋 Dataset Information

### Columns (14 features + 1 target)
- **Age**: Patient age in years
- **Sex**: 0=Female, 1=Male
- **Chest pain type**: 1=Typical angina, 2=Atypical, 3=Non-anginal, 4=Asymptomatic
- **BP**: Resting blood pressure (mmHg)
- **Cholesterol**: Serum cholesterol (mg/dL)
- **FBS over 120**: Fasting blood sugar > 120 mg/dL (0=No, 1=Yes)
- **EKG results**: Resting EKG (0=Normal, 1=ST-T abnormality, 2=LV hypertrophy)
- **Max HR**: Maximum heart rate achieved
- **Exercise angina**: Exercise-induced angina (0=No, 1=Yes)
- **ST depression**: ST segment depression by exercise
- **Slope of ST**: Slope of ST segment (1=Upsloping, 2=Flat, 3=Downsloping)
- **Number of vessels fluro**: Major vessels with blockage (0-3)
- **Thallium**: Thallium test result (3=Normal, 6=Fixed defect, 7=Reversible defect)
- **Heart Disease**: Target (Presence/Absence)

### Data Statistics
- **Total Records**: 270 patients
- **Features**: 14 predictive features
- **Target Variable**: Heart disease presence/absence
- **Data Completeness**: 100% (after preprocessing)

## 🎓 Dashboard Tabs

### 1. Overview Tab
- Dataset summary and statistics
- Data completeness check
- Risk distribution analysis
- Patient progression funnel
- Data dictionary with clinical descriptions

### 2. Exploratory Analysis Tab
- Feature correlation analysis
- Distribution analysis by disease status
- 3D feature relationships
- Categorical variable analysis
- Pairplot visualization

### 3. Model & Metrics Tab
- Top KPI metrics display
- Confusion matrix heatmap
- ROC and Precision-Recall curves
- Classification report (detailed metrics)
- Feature importance analysis
- Model coefficients (Logistic Regression)

### 4. Interactive Prediction Tab
- Patient information form
- Real-time risk assessment
- Risk level classification (Low/Moderate/High)
- Population comparison metrics
- Risk factors identification
- Probability scores

### 5. Insights & Takeaways Tab
- Dataset insights and patterns
- Model performance summary
- Top predictive features
- Clinical findings and recommendations
- Export options (data, predictions, reports)

## 🔧 Configuration

### Streamlit Settings (.streamlit/config.toml)
```toml
[theme]
primaryColor = "#3b82f6"        # Blue (medical theme)
backgroundColor = "#f5f7fa"     # Light background
secondaryBackgroundColor = "#e0e7ff"  # Light blue
textColor = "#1e3a8a"           # Dark blue text
```

## 📦 Dependencies

See `requirements.txt` for complete list:

- **streamlit** (1.28.1): Web app framework
- **pandas** (2.0.3): Data manipulation
- **numpy** (1.24.3): Numerical computing
- **scikit-learn** (1.3.0): Machine learning
- **plotly** (5.17.0): Interactive visualizations
- **seaborn** (0.12.2): Statistical plotting
- **matplotlib** (3.7.2): Plotting library
- **scipy** (1.11.2): Scientific computing

## 🌐 Deployment

### Streamlit Community Cloud

1. **Push to GitHub**
   - Create a GitHub repository
   - Push all files except `.gitignore` and large data files

2. **Connect to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app" and select your GitHub repo
   - Choose main branch and `app.py` as the main file
   - Click "Deploy"

3. **Add Secrets (if needed)**
   - Go to app settings
   - Add any API keys or credentials in the "Secrets" section

### Environment Variables
Create a `.env` file for local development (optional):
```
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=false
```

## 🎨 Design & Styling

### Professional Features
- Healthcare color palette (blues and greens)
- Clean, modern UI with card-based layout
- Responsive design for mobile and desktop
- Custom CSS for enhanced visual hierarchy
- Consistent typography and spacing
- Logo placeholder for hospital/institute branding

### Accessibility
- High contrast colors for readability
- Clear button labels and instructions
- Comprehensive hover tooltips
- Mobile-responsive layout

## 📊 Model Details

### Logistic Regression
- Binary classification model
- Features are standardized (mean=0, std=1)
- Interpretable coefficients show feature impact
- Fast inference, good baseline

### Random Forest
- Ensemble of 100 decision trees
- No feature scaling required
- Captures non-linear relationships
- Provides feature importance scores

### Performance Metrics
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True + False positives)
- **Recall/Sensitivity**: True positives / (True positives + False negatives)
- **Specificity**: True negatives / (True negatives + False positives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve

## 💡 Usage Examples

### Exploratory Analysis
1. Navigate to the "Exploratory Analysis" tab
2. Use sidebar filters to focus on specific patient groups
3. Examine correlation heatmap to identify feature relationships
4. Select features from dropdowns to deep-dive into distributions

### Making Predictions
1. Go to "Interactive Prediction" tab
2. Fill in patient information using form widgets
3. Click "Generate Prediction" button
4. Review risk assessment and comparison metrics
5. Identify key risk factors for clinical decision-making

### Exporting Results
1. Navigate to "Insights & Takeaways" tab
2. Use download buttons to export:
   - Filtered dataset (CSV)
   - Model predictions (CSV)
   - Dashboard report (TXT)

## 🔐 Data Privacy & Security

- No data is stored on external servers (local deployment)
- CSV file remains on your machine
- Models are trained locally in-memory
- No user data is collected or transmitted
- Suitable for HIPAA-compliant environments (with proper setup)

## 📞 Support & Contact

For questions or issues:
- Review the Data Dictionary in the Overview tab
- Check model descriptions in the Model & Metrics tab
- Examine clinical recommendations in Insights tab
- Email: research@hospital.edu

## 📄 License & Citation

This dashboard is provided for educational and research purposes. When presenting at conferences or publishing, please cite:

```
Heart Disease Prediction Dashboard (2024)
A Professional Streamlit Application for Clinical Risk Assessment
Data Science Conference Edition
```

## 🎯 Conference Presentation Tips

### Recommended Demo Flow
1. **Start with Overview**: Show dataset and prevalence statistics
2. **Highlight Visualizations**: Present key patterns in Exploratory Analysis
3. **Demonstrate Models**: Compare metrics and performance in Model tab
4. **Live Prediction**: Show interactive prediction with a typical/edge-case patient
5. **Share Insights**: Conclude with clinical findings and takeaways

### Key Talking Points
- Multiple integrated visualizations tell a complete story
- Interactive filtering enables targeted analysis
- Two complementary models provide balanced approach
- Real-time prediction supports clinical decision-making
- Exportable insights enable integration with clinical workflows

### Time Estimates
- Full dashboard walkthrough: 10-15 minutes
- Focus on key features: 5-7 minutes
- Interactive demo only: 3-5 minutes

## 📝 Version History

**Version 1.0** (2024)
- Initial release
- 5 main tabs with 15+ visualizations
- 2 ML models with complete evaluation
- Interactive prediction system
- Professional styling and documentation
- Conference-ready design

## ✅ Checklist for Production Deployment

- [ ] Test with real patient data
- [ ] Verify model performance metrics
- [ ] Check data privacy compliance
- [ ] Validate all visualizations render correctly
- [ ] Test on different screen sizes
- [ ] Prepare presentation scripts
- [ ] Set up error logging
- [ ] Document any custom modifications
- [ ] Create user training materials
- [ ] Establish model retraining schedule

## 🚀 Future Enhancements

Potential improvements for future versions:
- Model retraining pipeline
- Real patient data integration
- Advanced statistical testing
- Automated report generation
- Patient cohort comparison tools
- Risk stratification algorithms
- Integration with EHR systems
- Mobile app version
- API endpoint for external predictions

---

**Built for excellence in clinical data science** ❤️
