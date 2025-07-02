import pandas as pd
import mlflow
from fastapi import FastAPI
from .pydantic_models import PredictionRequest, PredictionResponse

# Initialize the FastAPI app
app = FastAPI(title="Bati Bank Credit Risk API")

# --- Load Model from MLflow Registry ---
# The model name should be the same as the one registered in train.py
MODEL_NAME = "BatiBankCreditRiskModel"
# Load the latest production version of the model
# In a real scenario, you'd use "Production" or "Staging"
# For this project, we'll use "latest" for simplicity.
model_uri = f"models:/{MODEL_NAME}/latest"
model = mlflow.pyfunc.load_model(model_uri)

print("Model loaded successfully from MLflow Registry.")

# --- API Endpoints ---
@app.get("/")
def read_root():
    """A simple endpoint to test if the API is running."""
    return {"message": "Credit Risk Prediction API is running."}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Accepts customer feature data and returns the credit risk probability.
    """
    # Convert the Pydantic request model to a pandas DataFrame
    # The model expects a DataFrame with the same feature names.
    input_data = pd.DataFrame([request.model_dump()])

    # Get the prediction probability for the positive class (class 1)
    # The `predict` method of mlflow.pyfunc returns probabilities
    # if the underlying model supports `predict_proba`.
    # For sklearn models, this is often the case.
    # The result is usually a 2D array of [[prob_class_0, prob_class_1]]
    prediction_proba = model.predict(input_data)[0][1]

    return PredictionResponse(risk_probability_class_1=prediction_proba)