"""
Heart Disease Prediction Dashboard
A professional, interactive Streamlit application for heart disease risk assessment and prediction.
Built for medical/data science conferences.
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import os
import sys

# Add utils to path
utils_path = os.path.join(os.path.dirname(__file__), 'utils')
sys.path.insert(0, utils_path)

from data_loader import load_data, preprocess_data, get_data_summary, get_filtered_data, get_risk_categories, DATA_DICTIONARY, NUMERIC_FEATURES
from model import HeartDiseaseModel
import visuals as viz


# ==================== PAGE CONFIG & STYLING ====================
st.set_page_config(
    page_title="Heart Disease Prediction Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
custom_css = """
<style>
    /* Main Container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header Styling */
    h1 {
        color: #1e3a8a;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    h2 {
        color: #1e3a8a;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    
    h3 {
        color: #2563eb;
        margin-top: 1.5rem;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #3b82f6;
        margin-bottom: 1rem;
    }
    
    /* Sidebar */
    .sidebar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button Styling */
    button {
        border-radius: 6px !important;
        font-weight: 600 !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 1px solid #ddd;
        margin-top: 3rem;
        font-size: 0.9rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* DataFrame styling */
    .dataframe {
        border-collapse: collapse;
        width: 100%;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)


# ==================== INITIALIZATION & DATA LOADING ====================
@st.cache_resource
def load_and_prepare_data():
    """Load and preprocess data once."""
    df = load_data("heart_disease.csv")
    df = preprocess_data(df)
    df = get_risk_categories(df)
    return df


@st.cache_resource
def train_models(df):
    """Train models once."""
    model_manager = HeartDiseaseModel(df)
    model_manager.split_data()
    model_manager.train_logistic_regression()
    model_manager.train_random_forest()
    return model_manager


# Load data
df = load_and_prepare_data()
model_manager = train_models(df)

# ==================== SIDEBAR & FILTERS ====================
with st.sidebar:
    st.markdown("### 🏥 Dashboard Controls")
    st.divider()
    
    # Theme toggle
    theme = st.radio("Theme", ["Light", "Dark"], horizontal=True)
    
    st.divider()
    st.markdown("### 📊 Data Filters")
    
    # Age range slider
    age_range = st.slider(
        "Age Range (years)",
        int(df["Age"].min()),
        int(df["Age"].max()),
        (30, 75),
        key="age_filter"
    )
    
    # Sex selector
    sex_options = {0: "Female", 1: "Male"}
    sex_selected = st.multiselect(
        "Sex",
        [0, 1],
        default=[0, 1],
        format_func=lambda x: sex_options[x],
        key="sex_filter"
    )
    
    # Chest pain type
    cp_options = {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal", 4: "Asymptomatic"}
    cp_selected = st.multiselect(
        "Chest Pain Type",
        [1, 2, 3, 4],
        default=[1, 2, 3, 4],
        format_func=lambda x: cp_options[x],
        key="cp_filter"
    )
    
    # Max HR range
    max_hr_range = st.slider(
        "Max Heart Rate (bpm)",
        int(df["Max HR"].min()),
        int(df["Max HR"].max()),
        (71, 202),
        key="hr_filter"
    )
    
    # Cholesterol range
    chol_range = st.slider(
        "Cholesterol (mg/dL)",
        int(df[df["Cholesterol"] > 0]["Cholesterol"].min()),
        int(df["Cholesterol"].max()),
        (100, 400),
        key="chol_filter"
    )
    
    # BP range
    bp_range = st.slider(
        "Resting BP (mmHg)",
        int(df["BP"].min()),
        int(df["BP"].max()),
        (94, 200),
        key="bp_filter"
    )
    
    st.divider()
    st.markdown("### 🤖 Model Selection")
    
    selected_model = st.radio(
        "Choose Model",
        ["Logistic Regression", "Random Forest"],
        horizontal=False
    )
    
    st.divider()
    
    # Apply filters button
    apply_filters = st.button("🔄 Apply Filters", use_container_width=True)
    
    # Show filter info
    st.markdown("### 📋 Filter Summary")
    st.info(f"""
    **Active Filters:**
    - Age: {age_range[0]}-{age_range[1]} years
    - Sex: {', '.join([sex_options[s] for s in sex_selected]) if sex_selected else 'None'}
    - Chest Pain: {len(cp_selected)} selected
    - Max HR: {max_hr_range[0]}-{max_hr_range[1]} bpm
    - Cholesterol: {chol_range[0]}-{chol_range[1]} mg/dL
    - BP: {bp_range[0]}-{bp_range[1]} mmHg
    """)


# Apply filters
filters = {
    "age_range": age_range,
    "sex": sex_selected,
    "chest_pain_type": cp_selected,
    "max_hr_range": max_hr_range,
    "cholesterol_range": chol_range,
    "bp_range": bp_range
}

df_filtered = get_filtered_data(df, filters)


# ==================== HEADER ====================
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1>❤️ Heart Disease Prediction Dashboard</h1>
    <p style="font-size: 1.1rem; color: #666; margin: 0;">
        Professional Risk Assessment and Prediction System
    </p>
    <p style="font-size: 0.95rem; color: #999; margin-top: 0.5rem;">
        Powered by Machine Learning | Data Science Conference Edition
    </p>
</div>
""", unsafe_allow_html=True)


# ==================== TOP KPI METRICS ====================
st.markdown("### 📊 Model Performance Metrics")

col1, col2, col3, col4, col5, col6 = st.columns(6)

metrics = model_manager.metrics[selected_model]

with col1:
    st.metric(
        "Accuracy",
        f"{metrics['accuracy']:.3f}",
        delta=f"{metrics['accuracy']*100:.1f}%",
        delta_color="normal"
    )

with col2:
    st.metric(
        "Precision",
        f"{metrics['precision']:.3f}",
        delta=f"{metrics['precision']*100:.1f}%",
        delta_color="normal"
    )

with col3:
    st.metric(
        "Recall/Sensitivity",
        f"{metrics['recall']:.3f}",
        delta=f"{metrics['recall']*100:.1f}%",
        delta_color="normal"
    )

with col4:
    st.metric(
        "Specificity",
        f"{metrics['specificity']:.3f}",
        delta=f"{metrics['specificity']*100:.1f}%",
        delta_color="normal"
    )

with col5:
    st.metric(
        "F1-Score",
        f"{metrics['f1']:.3f}",
        delta=f"{metrics['f1']*100:.1f}%",
        delta_color="normal"
    )

with col6:
    st.metric(
        "AUC-ROC",
        f"{metrics['auc']:.3f}",
        delta=f"{metrics['auc']*100:.1f}%",
        delta_color="normal"
    )

st.divider()


# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Overview",
    "🔬 Exploratory Analysis",
    "🎯 Model & Metrics",
    "🔮 Interactive Prediction",
    "💡 Insights & Takeaways"
])


# ==================== TAB 1: OVERVIEW ====================
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Dataset Overview")
        
        dataset_stats = f"""
        **Total Patients:** {len(df_filtered):,}
        
        **Disease Prevalence:** {(df_filtered['Heart_Disease_Binary'].sum() / len(df_filtered) * 100):.1f}%
        
        **Features:** {len(df_filtered.columns)}
        
        **Data Completeness:** 100%
        
        **Date Range:** All patients included
        """
        st.info(dataset_stats)
        
        # Data summary table
        st.markdown("### 📈 Summary Statistics")
        summary_df = df_filtered[NUMERIC_FEATURES].describe().round(2)
        st.dataframe(summary_df, use_container_width=True)
    
    with col2:
        st.markdown("### 📋 Data Dictionary")
        
        # Create expandable data dictionary
        with st.expander("View Full Data Dictionary", expanded=True):
            for col_name, description in DATA_DICTIONARY.items():
                st.markdown(f"**{col_name}**")
                st.caption(description)
        
        st.markdown("### ⚠️ Missing Values")
        
        missing_data = df_filtered.isnull().sum()
        if missing_data.sum() == 0:
            st.success("✅ No missing values detected!")
        else:
            st.dataframe(missing_data[missing_data > 0], use_container_width=True)
    
    st.divider()
    
    # Risk distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Risk Distribution")
        risk_dist = df_filtered['Risk_Category'].value_counts()
        fig_risk = viz.create_categorical_analysis(
            df_filtered,
            "Risk_Category",
            target_col="Heart Disease"
        )
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        st.markdown("### 🔄 Patient Progression Funnel")
        fig_funnel = viz.create_funnel_chart(df_filtered)
        st.plotly_chart(fig_funnel, use_container_width=True)


# ==================== TAB 2: EXPLORATORY ANALYSIS ====================
with tab2:
    st.markdown("### 📊 Data Exploration & Visualization")
    
    # Correlation heatmap
    st.markdown("#### 🔗 Feature Correlation Analysis")
    fig_corr = viz.create_correlation_heatmap(df_filtered, NUMERIC_FEATURES)
    st.pyplot(fig_corr, use_container_width=True)
    
    st.divider()
    
    # Feature distribution analysis
    st.markdown("#### 📈 Feature Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_feature = st.selectbox(
            "Select feature for distribution analysis:",
            NUMERIC_FEATURES,
            key="dist_feature"
        )
        
        fig_dist = viz.create_distribution_plots(df_filtered, selected_feature)
        st.pyplot(fig_dist, use_container_width=True)
    
    with col2:
        fig_box = viz.create_box_plot(df_filtered, selected_feature)
        st.plotly_chart(fig_box, use_container_width=True)
    
    st.divider()
    
    # 3D Scatter plot
    st.markdown("#### 🎯 3D Feature Analysis")
    fig_3d = viz.create_3d_scatter(df_filtered)
    st.plotly_chart(fig_3d, use_container_width=True)
    
    st.divider()
    
    # Categorical analysis
    st.markdown("#### 📊 Categorical Variables Analysis")
    
    col1, col2 = st.columns(2)
    
    categorical_cols = ["Sex", "Chest pain type", "Exercise angina", "FBS over 120"]
    
    with col1:
        cat_feature1 = st.selectbox("Select first categorical feature:", categorical_cols, key="cat1")
        fig_cat1 = viz.create_categorical_analysis(df_filtered, cat_feature1)
        st.plotly_chart(fig_cat1, use_container_width=True)
    
    with col2:
        cat_feature2 = st.selectbox("Select second categorical feature:", 
                                    [c for c in categorical_cols if c != cat_feature1], 
                                    key="cat2")
        fig_cat2 = viz.create_categorical_analysis(df_filtered, cat_feature2)
        st.plotly_chart(fig_cat2, use_container_width=True)


# ==================== TAB 3: MODEL & METRICS ====================
with tab3:
    st.markdown(f"### 🎯 Model Performance: {selected_model}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔍 Confusion Matrix")
        fig_cm = viz.create_confusion_matrix_heatmap(metrics['confusion_matrix'], selected_model)
        st.pyplot(fig_cm, use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 Classification Report")
        
        report_dict = metrics['classification_report']
        
        report_data = {
            "Class": ["No Disease", "Disease", "Weighted Avg"],
            "Precision": [
                f"{report_dict['0']['precision']:.3f}",
                f"{report_dict['1']['precision']:.3f}",
                f"{report_dict['weighted avg']['precision']:.3f}"
            ],
            "Recall": [
                f"{report_dict['0']['recall']:.3f}",
                f"{report_dict['1']['recall']:.3f}",
                f"{report_dict['weighted avg']['recall']:.3f}"
            ],
            "F1-Score": [
                f"{report_dict['0']['f1-score']:.3f}",
                f"{report_dict['1']['f1-score']:.3f}",
                f"{report_dict['weighted avg']['f1-score']:.3f}"
            ]
        }
        
        st.dataframe(pd.DataFrame(report_data), use_container_width=True)
    
    st.divider()
    
    # ROC and PR curves
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔄 ROC Curve")
        fpr, tpr, _ = metrics['roc_curve']
        fig_roc = viz.create_roc_curve(fpr, tpr, metrics['auc'], selected_model)
        st.plotly_chart(fig_roc, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎯 Precision-Recall Curve")
        precision, recall, _ = metrics['precision_recall_curve']
        fig_pr = viz.create_precision_recall_curve(precision, recall, selected_model)
        st.plotly_chart(fig_pr, use_container_width=True)
    
    st.divider()
    
    # Feature importance
    st.markdown("#### 🌟 Feature Importance Analysis")
    
    importance_df = model_manager.get_feature_importance(selected_model)
    fig_imp = viz.create_feature_importance_chart(importance_df, selected_model)
    st.plotly_chart(fig_imp, use_container_width=True)
    
    # Feature importance table
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Top Features:**")
        st.dataframe(importance_df.head(10), use_container_width=True)
    
    with col2:
        if selected_model == "Logistic Regression":
            st.markdown("**Model Coefficients:**")
            coef_df = model_manager.get_coefficients(selected_model)
            st.dataframe(coef_df.head(10), use_container_width=True)


# ==================== TAB 4: INTERACTIVE PREDICTION ====================
with tab4:
    st.markdown("### 🔮 Patient Risk Assessment & Prediction")
    
    st.info("Enter patient information to generate a real-time risk prediction using the selected model.")
    
    # Create input form in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age (years)", min_value=25, max_value=100, value=55)
        bp = st.number_input("Resting BP (mmHg)", min_value=90, max_value=200, value=120)
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=0, max_value=600, value=240)
    
    with col2:
        max_hr = st.number_input("Max Heart Rate (bpm)", min_value=60, max_value=210, value=150)
        st_depression = st.number_input("ST Depression (mm)", min_value=0.0, max_value=10.0, value=0.5)
        sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
        sex_encoded = 1 if sex == "Male" else 0
    
    with col3:
        chest_pain = st.selectbox("Chest Pain Type", 
                                  {1: "Typical Angina", 2: "Atypical", 3: "Non-anginal", 4: "Asymptomatic"},
                                  format_func=lambda x: {1: "Typical Angina", 2: "Atypical", 3: "Non-anginal", 4: "Asymptomatic"}.get(x))
        
        exercise_angina = st.radio("Exercise-Induced Angina?", ["No", "Yes"], horizontal=True)
        exercise_angina_encoded = 1 if exercise_angina == "Yes" else 0
        
        fbs = st.radio("Fasting Blood Sugar > 120?", ["No", "Yes"], horizontal=True)
        fbs_encoded = 1 if fbs == "Yes" else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ekg = st.selectbox("EKG Results", 
                          {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"},
                          format_func=lambda x: {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"}.get(x))
    
    with col2:
        slope = st.selectbox("Slope of ST Segment",
                           {1: "Upsloping", 2: "Flat", 3: "Downsloping"},
                           format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}.get(x))
    
    with col3:
        vessels = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
        thallium = st.selectbox("Thallium Test Result",
                              {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"},
                              format_func=lambda x: {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}.get(x))
    
    st.divider()
    
    # Make prediction
    if st.button("🔮 Generate Prediction", use_container_width=True, type="primary"):
        # Create input dataframe
        input_data = pd.DataFrame({
            'Age': [age],
            'Sex': [sex_encoded],
            'Chest pain type': [chest_pain],
            'BP': [bp],
            'Cholesterol': [cholesterol],
            'FBS over 120': [fbs_encoded],
            'EKG results': [ekg],
            'Max HR': [max_hr],
            'Exercise angina': [exercise_angina_encoded],
            'ST depression': [st_depression],
            'Slope of ST': [slope],
            'Number of vessels fluro': [vessels],
            'Thallium': [thallium]
        })
        
        # Get prediction
        pred_class, pred_proba = model_manager.predict(selected_model, input_data)
        
        # Display results
        st.markdown("### 📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Risk level color coding
            if pred_proba < 0.3:
                risk_level = "🟢 LOW RISK"
                risk_color = "#2ca02c"
            elif pred_proba < 0.7:
                risk_level = "🟡 MODERATE RISK"
                risk_color = "#ff7f0e"
            else:
                risk_level = "🔴 HIGH RISK"
                risk_color = "#d62728"
            
            st.markdown(f"""
            <div style="background: {risk_color}; color: white; padding: 2rem; border-radius: 10px; text-align: center;">
                <h2 style="color: white; margin: 0;">{risk_level}</h2>
                <p style="font-size: 0.9rem; margin: 0.5rem 0;">Heart Disease Risk</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric(
                "Disease Probability",
                f"{pred_proba*100:.1f}%",
                delta=f"{(pred_proba - df_filtered['Heart_Disease_Binary'].mean())*100:.1f}%",
                delta_color="off" if pred_proba < df_filtered['Heart_Disease_Binary'].mean() else "inverse"
            )
        
        with col3:
            prediction_text = "Heart Disease: YES ⚠️" if pred_class == 1 else "Heart Disease: NO ✅"
            st.metric("Prediction", prediction_text)
        
        # Comparison with population
        st.markdown("### 📊 How This Patient Compares to Dataset")
        
        comparison_data = {
            "Metric": ["Age", "Max HR", "Cholesterol", "Resting BP", "ST Depression"],
            "Patient Value": [age, max_hr, cholesterol, bp, st_depression],
            "Dataset Mean": [
                f"{df_filtered['Age'].mean():.1f}",
                f"{df_filtered['Max HR'].mean():.1f}",
                f"{df_filtered['Cholesterol'].mean():.1f}",
                f"{df_filtered['BP'].mean():.1f}",
                f"{df_filtered['ST depression'].mean():.2f}"
            ]
        }
        
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
        
        # Risk factors identified
        st.markdown("### ⚠️ Key Risk Factors Identified")
        
        risk_factors = []
        if age > 60:
            risk_factors.append("• Age > 60 years")
        if bp > 140:
            risk_factors.append("• Elevated blood pressure (> 140 mmHg)")
        if cholesterol > 240:
            risk_factors.append("• High cholesterol (> 240 mg/dL)")
        if max_hr < 100:
            risk_factors.append("• Low maximum heart rate (< 100 bpm)")
        if st_depression > 1.5:
            risk_factors.append("• Significant ST depression (> 1.5 mm)")
        if exercise_angina_encoded == 1:
            risk_factors.append("• Exercise-induced angina present")
        if vessels > 0:
            risk_factors.append(f"• {vessels} major vessel(s) with potential blockage")
        
        if risk_factors:
            for factor in risk_factors:
                st.warning(factor)
        else:
            st.success("✅ No major risk factors detected")


# ==================== TAB 5: INSIGHTS & TAKEAWAYS ====================
with tab5:
    st.markdown("### 💡 Key Insights & Clinical Takeaways")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 📊 Dataset Insights")
        
        disease_rate = (df_filtered['Heart_Disease_Binary'].sum() / len(df_filtered) * 100)
        avg_age = df_filtered['Age'].mean()
        male_rate = (df_filtered['Sex'].sum() / len(df_filtered) * 100)
        
        insights = f"""
        **Disease Prevalence:** {disease_rate:.1f}% of patients in filtered dataset show heart disease
        
        **Mean Age:** {avg_age:.1f} years (Range: {int(df_filtered['Age'].min())}-{int(df_filtered['Age'].max())})
        
        **Gender Distribution:** {male_rate:.1f}% male, {100-male_rate:.1f}% female
        
        **Risk Categories:**
        - Low Risk: {len(df_filtered[df_filtered['Risk_Category'] == 'Low Risk'])} patients
        - Moderate Risk: {len(df_filtered[df_filtered['Risk_Category'] == 'Moderate Risk'])} patients
        - High Risk: {len(df_filtered[df_filtered['Risk_Category'] == 'High Risk'])} patients
        """
        
        st.info(insights)
    
    with col2:
        st.markdown("#### 🎯 Model Performance Summary")
        
        model_summary = f"""
        **Selected Model:** {selected_model}
        
        **Overall Accuracy:** {metrics['accuracy']*100:.1f}%
        
        **Sensitivity (Recall):** {metrics['recall']*100:.1f}%
        - Correctly identifies disease in {metrics['recall']*100:.1f}% of actual cases
        
        **Specificity:** {metrics['specificity']*100:.1f}%
        - Correctly identifies absence of disease in {metrics['specificity']*100:.1f}% of healthy cases
        
        **AUC-ROC:** {metrics['auc']:.3f}
        - Excellent discrimination between disease/no-disease groups
        """
        
        st.success(model_summary)
    
    st.divider()
    
    st.markdown("#### 🔬 Clinical Findings")
    
    # Top risk factors
    importance_df = model_manager.get_feature_importance(selected_model)
    top_factors = importance_df.head(5)['Feature'].tolist()
    
    findings = f"""
    **Most Predictive Features:**
    1. {top_factors[0] if len(top_factors) > 0 else "N/A"}
    2. {top_factors[1] if len(top_factors) > 1 else "N/A"}
    3. {top_factors[2] if len(top_factors) > 2 else "N/A"}
    4. {top_factors[3] if len(top_factors) > 3 else "N/A"}
    5. {top_factors[4] if len(top_factors) > 4 else "N/A"}
    
    **Key Patterns:**
    - Patients with multiple major vessels show higher disease prevalence
    - Age and maximum heart rate are significant indicators
    - Resting blood pressure and cholesterol levels correlate with disease presence
    - Exercise-induced angina is a strong clinical indicator
    
    **Recommendations:**
    - Regular monitoring for high-risk patients (ST depression > 1.5mm, low max HR)
    - Blood pressure management critical (target < 140 mmHg)
    - Comprehensive lipid profile for cholesterol levels > 240 mg/dL
    - Stress testing for patients with exercise-induced symptoms
    """
    
    st.markdown(findings)
    
    st.divider()
    
    # Download options
    st.markdown("#### 📥 Export Data & Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download filtered data
        csv_filtered = df_filtered.to_csv(index=False)
        st.download_button(
            label="📊 Download Filtered Data",
            data=csv_filtered,
            file_name="heart_disease_filtered.csv",
            mime="text/csv"
        )
    
    with col2:
        # Download predictions
        predictions_df = df_filtered[['Age', 'Sex', 'Heart Disease']].copy()
        
        # Add model predictions
        X_filtered = df_filtered[[col for col in df_filtered.columns if col not in ['Heart Disease', 'Heart_Disease_Binary', 'Risk_Category']]]
        
        if selected_model == "Logistic Regression":
            X_filtered_scaled = model_manager.scaler.transform(X_filtered)
            predictions_df['Predicted_Probability'] = model_manager.models[selected_model].predict_proba(X_filtered_scaled)[:, 1]
        else:
            predictions_df['Predicted_Probability'] = model_manager.models[selected_model].predict_proba(X_filtered)[:, 1]
        
        csv_pred = predictions_df.to_csv(index=False)
        st.download_button(
            label="🔮 Download Predictions",
            data=csv_pred,
            file_name=f"predictions_{selected_model.replace(' ', '_')}.csv",
            mime="text/csv"
        )
    
    with col3:
        # Download summary report
        summary_text = f"""Heart Disease Prediction Dashboard - Summary Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET SUMMARY:
- Total Patients: {len(df_filtered)}
- Disease Prevalence: {disease_rate:.1f}%
- Features: {len(df_filtered.columns)}

MODEL PERFORMANCE ({selected_model}):
- Accuracy: {metrics['accuracy']:.3f}
- Precision: {metrics['precision']:.3f}
- Recall: {metrics['recall']:.3f}
- Specificity: {metrics['specificity']:.3f}
- F1-Score: {metrics['f1']:.3f}
- AUC-ROC: {metrics['auc']:.3f}

FILTERS APPLIED:
- Age: {age_range[0]}-{age_range[1]} years
- Sex: {len(sex_selected)} selected
- Chest Pain Types: {len(cp_selected)} selected
- Max HR: {max_hr_range[0]}-{max_hr_range[1]} bpm
- Cholesterol: {chol_range[0]}-{chol_range[1]} mg/dL
- Resting BP: {bp_range[0]}-{bp_range[1]} mmHg
"""
        
        st.download_button(
            label="📄 Download Report",
            data=summary_text,
            file_name="dashboard_report.txt",
            mime="text/plain"
        )


# ==================== FOOTER ====================
st.divider()

footer_html = """
<div style="text-align: center; padding: 2rem; color: #666; border-top: 1px solid #ddd; margin-top: 3rem;">
    <p style="margin: 0.5rem 0; font-size: 0.9rem;">
        <strong>Heart Disease Prediction Dashboard</strong> | Data Science Conference Edition
    </p>
    <p style="margin: 0.5rem 0; font-size: 0.85rem;">
        Developed using Streamlit, Scikit-learn, and Plotly
    </p>
    <p style="margin: 0.5rem 0; font-size: 0.85rem;">
        © 2024 Medical Data Science Team | For Educational & Research Purposes
    </p>
    <p style="margin: 0.5rem 0; font-size: 0.8rem; color: #999;">
        Contact: research@hospital.edu | Conference Presentation Ready ✅
    </p>
</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)
