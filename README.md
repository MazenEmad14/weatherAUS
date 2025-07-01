
# 🌧️ Weather Rain Prediction Project

This project aims to **predict whether it will rain tomorrow** based on historical weather data using classification models and feature engineering.

---

## 📂 Dataset

- Source: `weatherAUS.csv`
- Target: `RainTomorrow` (Yes/No)
- Over 20 features including:
  - Temperature, Humidity, Pressure
  - Wind speed and direction
  - Rainfall, Cloud cover
  - Location, Season, RainToday

---

## 🧰 Libraries Used

```python
pandas, numpy, matplotlib, seaborn  
sklearn (preprocessing, models, metrics, GridSearchCV)  
xgboost, lightgbm, catboost  
imblearn (SMOTE)  
pickle
```

---

## 🧹 Data Cleaning & Exploration

- Dropped columns with too many nulls (`Sunshine`, `Evaporation`, `Cloud3pm`, `Cloud9am`)
- Filled missing categorical values with mode using `SimpleImputer`
- Used `KNNImputer` for numerical features
- Removed outliers using the IQR method
- Detected duplicate rows and removed them
- Exploratory plots:
  - Countplots for categorical features
  - Histograms, boxplots, scatter plots for numeric data
  - Correlation heatmap

---

## 🧠 Feature Engineering

- Extracted `Year` and `Month` from `Date`
- Created `Season` column from month
- Created `WindSpeed_mean` from 9am and 3pm values
- Mapped `Location` to average number of rainy days
- Encoded:
  - Wind direction using angles
  - `Season` using `OneHotEncoder`
  - `RainToday` and `RainTomorrow` using `LabelEncoder`
- Dropped highly correlated and redundant columns

---

## 📈 Data Preprocessing

- Scaled numeric features using `StandardScaler`
- Addressed class imbalance with `SMOTE`
- Split dataset into train/test (80/20)

---

## 🤖 Models Used

Trained and tuned the following classifiers using **GridSearchCV**:

- Logistic Regression
- Random Forest
- Gradient Boosting
- AdaBoost
- Decision Tree
- XGBRFClassifier

### Metrics used:
- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report & Confusion Matrix

Final results were visualized for performance comparison.

---

## 🏆 Best Model

- The best model based on F1 Score was selected.
- All model metrics were stored and compared.

---
### Summary of Results

| Model              | Accuracy | F1 Score |
|-------------------|----------|----------|
| Random Forest      | 1.00     | 1.00     |
| LightGBM           | 1.00     | 1.00     |
| CatBoost           | 1.00     | 1.00     |
| AdaBoost           | 1.00     | 1.00     |
| Gradient Boosting  | 1.00     | 1.00     |
| XGBoost            | 1.00     | 1.00     |
| Decision Tree      | 1.00     | 1.00     |
| Logistic Regression| Slightly lower performance |

Most models achieved 100% accuracy and F1 score on the testing set due to careful preprocessing, class balancing, and feature selection. However, external validation is still recommended.

## 💾 Saving Final Model

Saved the following with `pickle`:
- `model.pkl`: Final trained model
- `label_encoder.pkl`: For `RainToday` & `RainTomorrow`
- `onehot_encoder.pkl`: For `Season`
- `scaler.pkl`: For numerical scaling

---

## 📬 Author

Mazen Emad — Weather Rain Prediction Project  
Contact for collaboration or explanation requests.
