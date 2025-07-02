import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os

def eval_metrics(actual, pred, pred_proba):
    """
    Helper function to evaluate model performance and return a dictionary of metrics.
    """
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, zero_division=0)
    recall = recall_score(actual, pred, zero_division=0)
    f1 = f1_score(actual, pred, zero_division=0)
    roc_auc = roc_auc_score(actual, pred_proba)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }

if __name__ == "__main__":
    print("Starting Task 5: Model Training and Tracking...")

    # Define file path for the final training data
    final_training_data_path = "../data/processed/final_training_data.csv"

    # --- Step 1: Load Data ---
    df_training = pd.read_csv(final_training_data_path)
    print(f"Loaded training data. Shape: {df_training.shape}")

    # --- Step 2: Split Data into Features (X) and Target (y) ---
    X = df_training.drop("is_high_risk", axis=1)
    y = df_training["is_high_risk"]

    # --- Step 3: Split Data into Training and Testing Sets ---
    # Using stratify=y to ensure the proportion of the target variable is the same
    # in both train and test sets, which is crucial for imbalanced datasets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Data split into training and testing sets.")
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")

    # --- Step 4: Define Models to Train ---
    # We will train two models: a simple baseline and a more complex one.
    # `class_weight='balanced'` helps handle the imbalanced nature of our proxy target.
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42)
    }

    # --- Step 5: Set up MLflow Experiment ---
    mlflow.set_experiment("Bati Bank Credit Risk")
    print("MLflow experiment 'Bati Bank Credit Risk' is set.")

    # --- Step 6: Train, Evaluate, and Log Models ---
    for model_name, model in models.items():
        print(f"\n--- Training {model_name} ---")
        
        # Start a new MLflow run for each model
        with mlflow.start_run(run_name=model_name):
            # Log model type as a parameter
            mlflow.log_param("model_type", model_name)

            # Train the model
            model.fit(X_train, y_train)

            # Make predictions on the test set
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Evaluate the model
            metrics = eval_metrics(y_test, y_pred, y_pred_proba)
            print("Evaluation Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")

            # Log metrics to MLflow
            mlflow.log_metrics(metrics)
            
            # Log the trained model to MLflow
            mlflow.sklearn.log_model(model, "model")
            print(f"{model_name} and its metrics have been logged to MLflow.")

    # --- Step 7: Identify and Register the Best Model ---
    print("\n--- Identifying and Registering Best Model ---")
    
    # Search all runs in the experiment, sorted by roc_auc score
    best_run = mlflow.search_runs(
        experiment_names=["Bati Bank Credit Risk"],
        order_by=["metrics.roc_auc DESC"]
    ).iloc[0]

    best_run_id = best_run.run_id
    best_model_name = best_run["params.model_type"]
    best_roc_auc = best_run["metrics.roc_auc"]

    print(f"Best model is '{best_model_name}' with ROC-AUC of {best_roc_auc:.4f} (Run ID: {best_run_id})")

    # Construct the model URI
    model_uri = f"runs:/{best_run_id}/model"
    
    # Register the best model in the MLflow Model Registry
    model_name_for_registry = "BatiBankCreditRiskModel"
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=model_name_for_registry
    )
    print(f"Model '{model_name_for_registry}' registered to the MLflow Model Registry, Version: {registered_model.version}")