import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
import joblib


# Function to load and preprocess the dataset
def load_and_preprocess_data(filepath):
    """Load the dataset and preprocess it."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, scaler


# Train a Random Forest model
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model


# Hyperparameter tuning
def tune_random_forest(X_train, y_train, param_grid):
    if not param_grid:
        raise ValueError("Parameter grid for tuning is empty.")
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


# Train Logistic Regression
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(X_train, y_train)
    return model


# Evaluate the model
def evaluate_model(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)
    print(f"{model_name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Confusion Matrix for {model_name}:\n{confusion_matrix(y_test, y_pred)}")
    print(f"Classification Report for {model_name}:\n{classification_report(y_test, y_pred)}")
    return y_pred


# Plot ROC curves
def plot_roc_curves(models, X_test, y_test):
    plt.figure()
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_score = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")
        else:
            print(f"Skipping ROC curve for {name} (no predict_proba method).")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.show()


# Main workflow
def main():
    # Define file paths
    data_path = os.path.abspath(os.path.join("data", "diabetes.csv"))
    model_dir = os.path.abspath("models")
    model_path = os.path.join(model_dir, "diabetes_model_rf_tuned.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(data_path)

    # Define hyperparameter grid
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }

    # Train and tune models
    print("Training models...")
    rf_model = train_random_forest(X_train, y_train)
    tuned_rf_model = tune_random_forest(X_train, y_train, param_grid)
    log_model = train_logistic_regression(X_train, y_train)

    # Evaluate
    evaluate_model(rf_model, X_test, y_test, "Random Forest")
    evaluate_model(tuned_rf_model, X_test, y_test, "Tuned Random Forest")
    evaluate_model(log_model, X_test, y_test, "Logistic Regression")

    # ROC curves
    plot_roc_curves(
        {
            "Random Forest": rf_model,
            "Tuned RF": tuned_rf_model,
            "Logistic Regression": log_model,
        },
        X_test, y_test,
    )

    # Save best model and scaler
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(tuned_rf_model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")


if __name__ == "__main__":
    main()