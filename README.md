# Diabetes Prediction Project

This project uses machine learning models to predict whether a patient has diabetes based on medical data. The dataset used is the **Pima Indians Diabetes Database**.

---

## Features
The dataset contains the following features:
- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skinfold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age in years
- **Outcome**: 1 if diabetic, 0 otherwise

---

## How to Run the Project

1. Ensure you have Python installed (version 3.7 or higher).
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the dataset (`diabetes.csv`) in the `data` folder.
4. Run the script to train and evaluate the models:
   ```bash
   python src/pima_prediction.py
   ```

---

## File Structure
```
Project/
├── data/
│   └── diabetes.csv          # Dataset
├── models/
│   └── diabetes_model_rf_tuned.pkl  # Saved tuned Random Forest model
│   └── diabetes_rf_model.pkl        # Saved untuned Random Forest model
│   └── scaler.pkl            # Saved scaler
├── src/
│   └── pima_prediction.py    # Main script for training and evaluation
├── README.md                 # Project documentation
```

---

## Results
- **Random Forest Accuracy**: `XX%`
- **Tuned Random Forest Accuracy**: `XX%`
- **Logistic Regression Accuracy**: `XX%`

---

## About `pima_prediction.py`

The `pima_prediction.py` script is the main file for this project. It performs the following tasks:
1. Loads and preprocesses the dataset (`diabetes.csv`).
2. Trains three machine learning models:
   - Random Forest Classifier
   - Tuned Random Forest Classifier (with GridSearchCV)
   - Logistic Regression
3. Evaluates the models using accuracy, confusion matrix, and classification report.
4. Plots ROC curves for the models.
5. Saves the best model and scaler to the `models` folder.

---

## Future Enhancements
- Add more machine learning models (e.g., SVM, Gradient Boosting).
- Deploy the model using a web app (e.g., Streamlit or Flask).
- Perform feature engineering for better predictions.

---

## License
This project is licensed under the MIT License.

---

## Results
- **Random Forest Accuracy**: `85.6%`
- **Tuned Random Forest Accuracy**: `88.2%`
- **Logistic Regression Accuracy**: `80.4%`
