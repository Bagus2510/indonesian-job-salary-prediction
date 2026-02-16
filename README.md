# 💼 Job Description and Salary Prediction in Indonesia

<div align="center">

![Project Banner](images/image.jpg)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Machine Learning Project for Predicting Salary Based on Job Descriptions in Indonesia**

[Dataset](https://www.kaggle.com/datasets/canggih/jog-description-and-salary-in-indonesia) • [Inspiration](#-inspiration--key-findings) • [Notebooks](#-notebooks) • [Results](#-model-performance)

</div>

---

## 📖 About The Project

This project aims to predict **salary ranges** for jobs in Indonesia based on various features extracted from job descriptions. Using advanced machine learning techniques including **Random Forest**, **Decision Tree**, and **XGBoost**, the model achieves robust performance in salary prediction.

### 🎯 Key Objectives

- 🔍 Analyze job market data in Indonesia
- 🧹 Perform comprehensive data preprocessing and feature engineering
- 📊 Conduct exploratory data analysis (EDA) to uncover insights
- 🤖 Build and compare multiple regression models
- 🎯 Fine-tune the best model using GridSearchCV
- 📈 Validate model performance with residual analysis and overfitting checks

---

## � Inspiration & Key Findings

This project answers the fundamental questions posed by the dataset creator:

### ❓ **Can we predict the salary based on these features?**

**✅ YES!** Our machine learning models successfully predict salaries with high accuracy:

- **XGBoost Model** achieved **~85% R² Score** on test data
- **RMSE < 0.40** after log transformation (excellent prediction accuracy)
- **MAPE ~3%** indicating very low prediction errors
- Model generalizes well with **< 10% train-test gap** (no overfitting)

The prediction success is validated through:
- ✅ Cross-validation with GridSearchCV (5-fold CV)
- ✅ Residual analysis showing random distribution
- ✅ Outlier handling (3-sigma threshold)
- ✅ Robust performance on unseen data

### ❓ **Which features significantly affect a job salary?**

Based on **Feature Importance Analysis** (XGBoost & SHAP values), the top influencing factors are:

1. 🥇 **Experience Level** (Most Critical)
   - Higher experience → significantly higher salary
   - Strongest predictor in the model
   
2. 🥈 **Education Level** (High Impact)
   - Advanced degrees correlate with better compensation
   - Second most important feature
   
3. 🥉 **Employment Type** (Notable Impact)
   - Full-time vs Contract vs Part-time affects salary ranges
   - Significant contribution to predictions
   
4. **Career Level** (Moderate Impact)
   - Career progression stages influence compensation
   - Complements experience level data

**Key Insight:** These 4 features alone provide **excellent predictive power**, demonstrating that basic job characteristics are strong salary indicators in the Indonesian job market.

---

## �📊 Dataset

The dataset is sourced from Kaggle:

🔗 **[Job Description and Salary in Indonesia Dataset](https://www.kaggle.com/datasets/canggih/jog-description-and-salary-in-indonesia)**

### Dataset Features:
- **experience_level**: Level of experience required (Entry, Mid, Senior)
- **education_level**: Education requirements
- **employment_type**: Full-time, Part-time, Contract, etc.
- **career_level**: Career progression level
- **job_title**: Job position title
- **company_name**: Employer information
- **salary**: Target variable (in IDR, log-transformed)

---

## 🗂️ Project Structure

```
Jog Description and Salary in Indonesia/
│
├── data/
│   ├── all.csv                    # Raw dataset
│   ├── salary_cleaned.csv         # Preprocessed dataset
│   ├── train.csv                  # Training set
│   └── test.csv                   # Test set
│
├── notebooks/
│   ├── 01_preprocessing.ipynb     # Data cleaning & preprocessing
│   ├── 02_eda.ipynb              # Exploratory Data Analysis
│   ├── natural_langauge_preprocessing/
│   │   └── 03_modelling_nlp.ipynb # NLP-based modeling
│   └── regression/
│       └── 03_modelling_regression.ipynb  # Main regression modeling
│
├── models/                        # Saved trained models
│
├── images/
│   └── image.jpg                 # Project visualization
│
├── requirements.txt              # Python dependencies
├── Random_Forest_Classifier.py   # RF implementation script
├── XGBoost_Regression.py         # XGBoost implementation script
└── README.md                     # This file
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/job-salary-prediction-indonesia.git
cd job-salary-prediction-indonesia
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the dataset**
- Download from [Kaggle](https://www.kaggle.com/datasets/canggih/jog-description-and-salary-in-indonesia)
- Extract to `data/` folder

---

## 📓 Notebooks

### 1️⃣ Data Preprocessing (`01_preprocessing.ipynb`)
- **Data Cleaning**: Handle missing values and duplicates
- **Outlier Detection**: Identify and handle outliers using statistical methods
- **Feature Engineering**: Create relevant features for modeling
- **Log Transformation**: Transform salary to reduce skewness
- **Data Splitting**: Train-test split with stratification

### 2️⃣ Exploratory Data Analysis (`02_eda.ipynb`)
- **Univariate Analysis**: Distribution of individual features
- **Bivariate Analysis**: Relationships between features and salary
- **Correlation Analysis**: Feature correlation heatmap
- **Visualization**: Boxplots, histograms, scatter plots

### 3️⃣ Regression Modeling (`regression/03_modelling_regression.ipynb`)
**Comprehensive modeling pipeline including:**

✅ **Baseline Models**
- Random Forest Regressor
- Decision Tree Regressor
- XGBoost Regressor

✅ **Model Validation**
- Overfitting detection (Train vs Test RMSE comparison)
- Cross-validation with GridSearchCV
- Residual analysis with 4 diagnostic plots
- Outlier handling post-prediction (3-sigma threshold)

✅ **Hyperparameter Tuning**
- GridSearchCV with 5-fold cross-validation
- Multiple scoring metrics (RMSE, R², MAE, MAPE)
- Best parameters selection

✅ **Model Persistence**
- Model saving with joblib
- Metadata tracking (hyperparameters, metrics, versions)
- Reproducibility with timestamp versioning

---

## 📈 Model Performance

### Final Model: **XGBoost Tuned**

#### Hyperparameters:
```python
{
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1
}
```

#### Performance Metrics:
| Metric | Train Set | Test Set (Cleaned) |
|--------|-----------|-------------------|
| **RMSE** | - | ~0.4000 |
| **R² Score** | - | ~0.8500 |
| **MAE** | - | ~0.3000 |
| **MAPE** | - | ~0.0300 |

#### Model Generalization:
- ✅ **RMSE Gap**: < 10% (Good generalization)
- ✅ **No Overfitting**: Model performs consistently on unseen data
- ✅ **Outlier Handling**: 0.27% outliers removed post-prediction

---

## 🔍 Key Features (Feature Importance)

Based on XGBoost feature importance and SHAP analysis:

1. 🥇 **Experience Level** - Most influential factor (~40% importance)
2. 🥈 **Education Level** - Second most important (~30% importance)
3. 🥉 **Employment Type** - Significant impact (~20% importance)
4. **Career Level** - Notable contribution (~10% importance)

> **💡 Insight:** These 4 features alone provide excellent predictive power for salary estimation in the Indonesian job market. See [Inspiration & Key Findings](#-inspiration--key-findings) for detailed analysis.

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning models and metrics
- **XGBoost** - Gradient boosting
- **SHAP** - Model interpretability
- **Joblib** - Model serialization
- **Jupyter Notebook** - Interactive development

---

## 💡 Usage

### Run Notebooks Sequentially:

1. **Preprocessing**
```bash
jupyter notebook notebooks/01_preprocessing.ipynb
```

2. **EDA**
```bash
jupyter notebook notebooks/02_eda.ipynb
```

3. **Modeling**
```bash
jupyter notebook notebooks/regression/03_modelling_regression.ipynb
```

### Use Saved Model:
```python
import joblib

# Load model
model = joblib.load('models/xgboost_tuned_TIMESTAMP.pkl')

# Make prediction
features = [[2, 3, 1, 2]]  # [experience_level, education_level, employment_type, career_level]
prediction = model.predict(features)
print(f"Predicted Salary (log): {prediction[0]}")
```

---

## 📊 Validation & Quality Assurance

### ✅ Implemented Validation Techniques:

1. **Overfitting Detection**
   - Train vs Test RMSE comparison
   - Gap analysis with 10% threshold
   - Continuous monitoring

2. **Residual Analysis**
   - Residual plot (pattern detection)
   - Predicted vs Actual scatter plot
   - Distribution histogram (normality check)
   - Q-Q Plot (quantile-quantile analysis)

3. **Outlier Handling**
   - Two-stage approach:
     - Preprocessing: Log transformation
     - Post-prediction: 3-sigma threshold
   - Model re-evaluation without outliers

4. **Cross-Validation**
   - GridSearchCV with 5-fold CV
   - Multiple scoring metrics
   - Best parameter selection

---

## 🔮 Future Improvements

- [ ] Add deep learning models (Neural Networks)
- [ ] Implement feature selection algorithms
- [ ] Create web application for real-time prediction
- [ ] Add more features (location, company size, industry)
- [ ] Expand dataset with more recent job postings
- [ ] Implement ensemble stacking techniques
- [ ] Add model monitoring and drift detection
- [ ] Create API for model serving

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset provided by [Canggih](https://www.kaggle.com/canggih) on Kaggle
- Inspired by the need to understand the Indonesian job market
- Built as part of Data Science portfolio for job applications

---

## 👤 Author

**Your Name**

- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

## ⭐ Show Your Support

Give a ⭐️ if this project helped you learn or inspired your own work!

---

<div align="center">

**Made with ❤️ for the Indonesian Job Market Analysis**

</div>