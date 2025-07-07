# 💳 Real-Time Credit Card Fraud Detection with SHAP Explainability

This project is a **Streamlit-based interactive web application** designed to detect fraudulent credit card transactions. It leverages a pre-trained **XGBoost classification model**, addresses class imbalance with **SMOTE**, and provides **SHAP explainability** to help understand model decisions.

---

## 🚀 Features

✅ Upload transaction CSV file for real-time fraud detection  
✅ Risk categorization: 🟢 Low Risk | 🟡 Medium Risk | 🔴 High Risk  
✅ Download detected fraudulent transactions as CSV  
✅ Interactive risk distribution bar chart  
✅ Confusion Matrix visualization (if ground truth labels are provided)  
✅ SHAP Beeswarm plot to visualize feature importance  

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** for web UI  
- **XGBoost** for fraud prediction  
- **SMOTE** for handling data imbalance  
- **SHAP** for model interpretability  
- **Matplotlib** & **Seaborn** for visualizations  
- **Pandas**, **Joblib**, **Scikit-learn** for data processing  

---

## 📁 Project Structure

```
CREDIT_CARD_FRAUD_DETECTION/
├── Dataset/
│   ├── clean_fraud_dataset.csv      # Cleaned dataset (optional for testing)
│   └── fraudTest.csv                # Test dataset 
├── Notebooks/
│   ├── data_preprocessing.ipynb     # Data cleaning and preparation
│   ├── demographic_analysis.ipynb   # Demographic insights
│   ├── model_explainability.ipynb   # SHAP explainability exploration
│   ├── model_training.ipynb         # Model training code
│   └── time_pattern_analysis.ipynb  # Time-based fraud pattern analysis
├── app.py                           # Streamlit web application
├── feature_list.pkl                 # List of features used by the model
├── trained_model.pkl                # Pre-trained XGBoost fraud detection model
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation
```

---

## 🖥️ How to Run the App

### 1. Clone the Repository

```bash
git clone https://github.com/mrunalgaikwad2364/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Install Required Packages

It's recommended to use a virtual environment:

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📊 Expected Workflow

1. **Upload** your transaction CSV file  
2. Click **Detect Fraud** to view risk levels and fraudulent transactions  
3. Explore:
   - Risk level bar chart  
   - Download fraudulent transactions report  
4. Switch to **Confusion Matrix** tab if your data contains ground truth (`is_fraud` column)  
5. Explore model behavior in the **SHAP Explanation** tab  

---

## 📂 Sample Input Format

Your CSV should contain columns like:

| amount | state | unix_time | category | gender | ... |  
|--------|-------|-----------|----------|--------|-----|  
| 125.50 | CA    | 1625259000 | shopping | M      | ... |  

Unnecessary columns like name, card number, etc. will be ignored during processing.

---

## 🧑‍💻 Developed For

CodSoft Internship - Machine Learning Intern  
Project: End-to-End Fraud Detection with Explainability  

---

## ⚡ Future Improvements

- Real-time API integration  
- Advanced visualizations (Geospatial fraud detection, time-series trends)  
- Auto-retraining pipeline  
- Enhanced UI/UX with animations  

---

## 📬 Contact

Feel free to connect:  
[LinkedIn](https://www.linkedin.com/in/mrunal-gaikwad-328273300) | [GitHub](https://github.com/mrunalgaikwad2364/credit-card-fraud-detection.git)  
