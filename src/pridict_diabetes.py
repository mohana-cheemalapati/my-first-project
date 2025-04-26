import pandas as pd
import joblib

print("Script is running...")  # Debugging statement

def predict_diabetes(patient_data):
    """
    Predict whether a patient has diabetes based on their medical data.

    Parameters:
    - patient_data: A list of lists containing patient data in the following order:
        [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]

    Returns:
    - A string indicating whether the patient is "Diabetic" or "Not Diabetic".
    """
    # Load the saved model and scaler
    model_path = "models/diabetes_model_rf_tuned.pkl"
    scaler_path = "models/scaler.pkl"
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        return "Model or scaler file not found. Please check the paths."

    # Define the feature names (must match the training data)
    feature_names = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]

    # Convert the patient data to a DataFrame
    patient_data_df = pd.DataFrame(patient_data, columns=feature_names)

    # Preprocess the patient data
    patient_data_scaled = scaler.transform(patient_data_df)

    # Make a prediction
    prediction = model.predict(patient_data_scaled)
    return "Diabetic" if prediction[0] == 1 else "Not Diabetic"


def get_patient_data():
    """
    Collect patient data interactively from the user.
    Returns a 2D list with patient inputs.
    """
    print("Enter the following details about the patient:")
    try:
        pregnancies = int(input("Pregnancies: "))
        glucose = float(input("Glucose: "))
        blood_pressure = float(input("Blood Pressure: "))
        skin_thickness = float(input("Skin Thickness: "))
        insulin = float(input("Insulin: "))
        bmi = float(input("BMI: "))
        dpf = float(input("Diabetes Pedigree Function: "))
        age = int(input("Age: "))
        return [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]]
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return get_patient_data()  # Retry input


if __name__ == "__main__":
    # Get patient data from the user
    print("Starting the diabetes prediction program...")
    new_patient_data = get_patient_data()

    # Predict diabetes
    result = predict_diabetes(new_patient_data)
    print(f"The patient is: {result}")