# Quick Start Guide - Heart Disease Prediction Dashboard

## ⚡ 30-Second Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
streamlit run app.py
```

### 3. Open in Browser
The app will automatically open at: `http://localhost:8501`

---

## 🎯 First Steps

### What You'll See
1. **Header**: Professional dashboard title with metric cards (Accuracy, Precision, Recall, etc.)
2. **Sidebar**: Filters for age, sex, heart rate, cholesterol, BP, and model selection
3. **Five Tabs**: Overview, Analysis, Models, Prediction, Insights

### Quick Demo (3 minutes)

#### 1. Explore Data (1 min)
- Go to **Overview** tab
- See dataset statistics and risk distribution
- Check the funnel chart showing patient progression

#### 2. View Models (1 min)
- Go to **Model & Metrics** tab
- Compare model performance with metric cards at top
- Explore confusion matrix, ROC curve, and feature importance

#### 3. Make a Prediction (1 min)
- Go to **Interactive Prediction** tab
- Fill in patient information
- Click "Generate Prediction" to see risk assessment
- View how patient compares to dataset averages

---

## 🎛️ Using Sidebar Filters

**Left Panel Controls:**
- **Age Range**: Adjust slider to filter by age
- **Sex**: Select Male, Female, or both
- **Chest Pain Type**: Choose 1-4 types
- **Max Heart Rate**: Set range (bpm)
- **Cholesterol**: Set range (mg/dL)
- **Resting BP**: Set range (mmHg)
- **Model Selection**: Switch between Logistic Regression / Random Forest
- **Apply Filters**: Click button to update visualizations

All charts dynamically update based on your filters!

---

## 📊 Tab Guide

### Overview Tab
- **Dataset Summary**: Total patients, disease rate, feature count
- **Summary Statistics**: Mean, std, min, max for numeric features
- **Data Dictionary**: Hover to see field descriptions
- **Risk Distribution**: See low/moderate/high risk breakdown
- **Funnel Chart**: Track patient progression through risk stages

### Exploratory Analysis Tab
- **Correlation Heatmap**: See which features are related
- **Distribution Plots**: Histogram showing feature distributions by disease status
- **Box Plots**: Statistical comparison by disease presence
- **3D Scatter**: Multi-dimensional feature exploration (Age × Max HR × Cholesterol)
- **Categorical Analysis**: Bar charts for sex, chest pain, exercise angina, etc.

### Model & Metrics Tab
- **Top Metrics**: 6 key performance indicators in colorful cards
- **Confusion Matrix**: True/False positives and negatives heatmap
- **Classification Report**: Detailed precision, recall, F1 scores
- **ROC Curve**: Model's ability to discriminate disease vs no disease
- **Precision-Recall**: Trade-off between precision and sensitivity
- **Feature Importance**: Which features matter most in the model
- **Feature Coefficients**: (Logistic Regression only) Impact direction

### Interactive Prediction Tab
- **Patient Input Form**: Enter age, BP, cholesterol, etc.
- **Risk Assessment**: Instant prediction with color-coded risk level
- **Probability Score**: 0-100% chance of disease
- **Population Comparison**: How patient compares to dataset averages
- **Risk Factors**: List of identified clinical risk factors
- **Download Results**: Export patient data with predictions

### Insights & Takeaways Tab
- **Dataset Insights**: Key patterns and statistics
- **Model Summary**: Performance overview
- **Clinical Findings**: Top risk factors and recommendations
- **Export Options**: Download filtered data, predictions, or reports

---

## 🔮 Making Predictions

### Example Patient: 55-year-old Male

1. **Fill in form:**
   - Age: 55
   - Sex: Male
   - Resting BP: 135 mmHg
   - Cholesterol: 250 mg/dL
   - Max HR: 140 bpm
   - Chest Pain: Typical Angina
   - Exercise Angina: Yes
   - Other fields: Fill as appropriate

2. **Click "Generate Prediction"**

3. **Interpretation:**
   - **🟢 GREEN (0-30%)**: Low risk - routine monitoring
   - **🟡 YELLOW (30-70%)**: Moderate risk - closer evaluation recommended
   - **🔴 RED (70-100%)**: High risk - immediate clinical attention needed

---

## 💾 Exporting Data

### In Insights Tab:
- **📊 Download Filtered Data**: Get CSV of filtered patients
- **🔮 Download Predictions**: Get predictions for filtered dataset
- **📄 Download Report**: Get summary statistics and model parameters

### File Naming:
- `heart_disease_filtered.csv` - Your filtered dataset
- `predictions_[model_name].csv` - Predictions with probabilities
- `dashboard_report.txt` - Summary statistics

---

## 🐛 Troubleshooting

### App Won't Start
```bash
# Make sure you're in the right directory
cd "Heart Disease"

# Try installing again
pip install --upgrade streamlit pandas scikit-learn plotly

# Then run
streamlit run app.py
```

### Data Not Loading
- Check `heart_disease.csv` is in the same folder as `app.py`
- File should have exactly 14 columns and ~270 rows
- Column names must match exactly

### Models Taking Too Long
- First run trains the models (30-60 seconds)
- Streamlit caches them, so subsequent runs are instant
- If still slow, reduce number of Random Forest trees (edit `utils/model.py`)

### Charts Not Displaying
- Try refreshing the page (R key in Streamlit)
- Check that Plotly installed: `pip install plotly`
- Clear Streamlit cache: `streamlit cache clear`

---

## 🎓 Key Concepts

### Sensitivity vs Specificity
- **Sensitivity (Recall)**: "Of actual disease cases, how many did we catch?"
- **Specificity**: "Of healthy people, how many did we correctly identify?"
- **Goal**: High both! But there's always a trade-off.

### AUC-ROC Score
- **0.5**: Random guessing
- **0.7-0.8**: Good discrimination
- **0.8-0.9**: Very good
- **0.9+**: Excellent

### Feature Importance
- **Random Forest**: Higher = more important for decision making
- **Logistic Regression**: Coefficient size shows strength, sign shows direction

---

## 🚀 Advanced Tips

### Keyboard Shortcuts (in Streamlit)
- `R`: Rerun app
- `C`: Clear cache
- `i`: View app info

### Filtering for Analysis
1. Use sidebar to filter to specific patient subgroup
2. Compare model predictions across subgroups
3. Export filtered data for further statistical analysis

### Deployment
- Share with colleagues: Deploy to Streamlit Cloud (see README.md)
- Local network: `streamlit run app.py --server.address 0.0.0.0`
- Then access from other computers: `http://your-ip:8501`

---

## ✅ Verification Checklist

After installation, verify everything works:

- [ ] App starts without errors: `streamlit run app.py`
- [ ] All 5 tabs visible at top
- [ ] Sidebar filters responsive to changes
- [ ] Charts render smoothly
- [ ] Prediction form works end-to-end
- [ ] Download buttons work
- [ ] No error messages in terminal

---

## 📞 Need Help?

1. **Data Questions**: Check Overview → Data Dictionary
2. **Model Questions**: Check Model & Metrics → Classification Report
3. **Prediction Issues**: Check Interactive Prediction → Risk Factors
4. **Technical Issues**: See Troubleshooting section above

---

## 🎉 Ready to Go!

You now have a **conference-ready heart disease prediction dashboard**. 

**Next Steps:**
1. Explore the data
2. Understand the models
3. Try predictions
4. Export insights
5. Prepare your presentation!

**Good luck with your conference presentation!** 🏥❤️
