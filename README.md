# 🩸 Diabetes Prediction

## 📌 Project Overview
This project demonstrates how to **predict diabetes risk** based on various health metrics with a special focus on **feature scaling techniques**, **feature selection methods**, and **comparative model performance**.

- 🧹 **Data Preprocessing** – techniques for handling categorical features and ensuring data quality
- 📊 **Feature Scaling Comparison** – evaluated three different scaling methods: Standard, Min-Max, and Robust scaling
- 🔍 **Feature Selection** – applied multiple feature selection methods including correlation analysis, variance threshold, and statistical tests
- 🤖 **Machine Learning** – trained classification models to predict diabetes with comprehensive evaluation metrics
- 🎯 **Goal** – develop an efficient and accurate diabetes prediction model while understanding the impact of feature selection and scaling

## 📊 Dataset
The Diabetes Prediction dataset contains patient health measurements with the following key features:
- Gender
- Age
- Hypertension and heart disease status
- Smoking history
- BMI (Body Mass Index)
- HbA1c level (glycated hemoglobin)
- Blood glucose level

Each patient record has an associated diabetes status (0: No diabetes, 1: Diabetes) which is the target variable for prediction.

## 🛠 Key Techniques Implemented

1. **Data Exploration & Visualization**
   - Distribution analysis of diabetes cases
   - Statistical summary of numerical features
   - Visualization of feature distributions by diabetes status
   - Correlation analysis between health metrics

2. **Data Preprocessing**
   - Gender encoding
   - One-hot encoding for categorical variables (smoking history)
   - Missing value detection and handling
   - Data cleaning to ensure quality inputs for modeling

3. **Feature Scaling Comparison**
   - StandardScaler (Z-score normalization)
   - MinMaxScaler (0-1 scaling)
   - RobustScaler (median and IQR based scaling)
   - Visual comparison of scaling impacts on feature distributions
   
4. **Feature Selection Methods**
   - Correlation Analysis: Measuring linear relationship between features and target
   - Variance Threshold: Removing low-variance features that provide little information
   - Statistical Tests (F-statistics): Identifying features with strong discriminatory power
   - Comparative analysis of selected features across methods

5. **Model Training & Evaluation**
   - Logistic Regression
   - Random Forest Classifier
   - Comprehensive metrics:
     - Accuracy
     - Precision, Recall, and F1-Score
     - Confusion matrices for error analysis
   - Model comparison to identify the best approach

## 📈 Results
The analysis revealed that Random Forest generally performed better for diabetes prediction, with an accuracy of 96.96% compared to Logistic Regression's 95.88%.

Key findings:
- The most predictive features for diabetes were blood glucose level and HbA1c level, consistently identified across all feature selection methods
- StandardScaler was chosen for the final model due to its suitability for the algorithms used
- Feature selection using SelectKBest with F-statistics provided the most relevant feature set
- Random Forest showed better performance particularly in the precision of diabetes prediction (93% vs 86%)
- Both models struggled somewhat with recall for the diabetes class, with Random Forest performing better (69% vs 61%)

## 📦 Requirements
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## 🚀 How to Use
1. Clone this repository
2. Ensure you have all required packages installed
3. Open the Jupyter notebook `Diabetes_Prediction_Condensed.ipynb`
4. Run the cells sequentially to see the analysis and results

## 📂 File Structure
- `Diabetes_Prediction_Condensed.ipynb` - The main Jupyter notebook containing all analysis and code
- `diabetes_prediction_dataset.csv` - The dataset file
- `README.md` - This file with project information

## 💡 Key Insights
- Feature scaling is critical for machine learning models, with each technique having distinct effects on the feature distributions
- The top 8 features selected by F-statistics were sufficient for high-accuracy prediction, demonstrating effective dimensionality reduction
- Blood glucose level and HbA1c level were the strongest predictors of diabetes, aligning with medical knowledge
- Random Forest outperformed Logistic Regression, likely due to its ability to capture non-linear relationships in health data
- The models showed high accuracy but somewhat lower recall for diabetes cases, suggesting potential for further optimization to reduce false negatives
- Class imbalance (fewer diabetes cases than non-diabetes) impacts model performance and should be considered in clinical applications
