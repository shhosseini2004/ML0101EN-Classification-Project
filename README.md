# ML0101EN-Classification-Project  
This repository contains the **Classification Project** from the ML0101EN course series. It demonstrates the application of supervised learning techniques to predict the presence of **heart disease** based on patient health parameters.

---

## ❤️ Project Objective
The goal of this project is to build a classification model that predicts whether a patient is likely to have heart disease using clinical and physiological data.

---

## 🧠 Key Tasks
- Load and analyze the dataset (`heart.csv`)
- Standardize numeric features for model consistency
- Split data into training and test sets
- Train classification models:
  - **Logistic Regression**
  - (Optional extensions: Decision Tree, Random Forest, etc.)
- Evaluate model performance using:
  - Accuracy  
  - F1-Score  
  - Jaccard Index  
  - Log Loss  
  - Confusion Matrix visualization

---

## 📊 Dataset Description

| Feature | Meaning |
|----------|----------|
| age | Age of the patient |
| sex | Gender (1 = male, 0 = female) |
| cp | Chest pain type |
| trtbps | Resting blood pressure |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting electrocardiographic results |
| thalachh | Maximum heart rate achieved |
| exng | Exercise induced angina (1 = yes, 0 = no) |
| oldpeak | Depression induced by exercise relative to rest |
| slp | Slope of the peak exercise ST segment |
| caa | Number of major vessels (0–3) colored by fluoroscopy |
| thall | Thalassemia type |
| output | Target variable (1 = heart disease, 0 = no heart disease) |

---

## 🛠️ Technologies Used

| Library | Purpose |
|----------|----------|
| **Python** | Core environment |
| **NumPy** / **Pandas** | Data manipulation |
| **Scikit-learn** | Modeling and metrics |
| **Matplotlib / Seaborn** | Data visualization |
| **SciPy** | Optimization and numerical tools |

---

## 🧪 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/shhosseini2004/ML0101EN-Classification-Project.git
cd ML0101EN-Classification-Project
