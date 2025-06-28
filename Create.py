import os

# Define folder structure
folders = [
    ".github/workflows",
    "data/raw",
    "data/processed",
    "notebooks",
    "src",
    "src/api",
    "tests"
]

# Define empty files to create
files = {
    ".github/workflows/ci.yml": "# GitHub CI/CD config\n",
    "notebooks/1.0-eda.ipynb": "",  # You can fill this later
    "src/__init__.py": "",
    "src/data_processing.py": "# Data processing and feature engineering logic\n",
    "src/train.py": "# Model training script\n",
    "src/predict.py": "# Prediction/inference logic\n",
    "src/api/main.py": "# FastAPI entry point\n",
    "src/api/pydantic_models.py": "# Pydantic models for request/response\n",
    "tests/test_data_processing.py": "# Unit tests for data processing\n",
    "Dockerfile": "# Dockerfile for containerizing the app\n",
    "docker-compose.yml": "# Docker Compose for multi-container setup\n",
    "requirements.txt": "# Required Python packages\n",
    ".gitignore": "data/\n__pycache__/\n.ipynb_checkpoints/\n.env\n",
    "README.md": "# Credit Risk Model\n\nProject for credit risk scoring using ML.\n"
}

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files with initial content
for path, content in files.items():
    with open(path, "w") as f:
        f.write(content)

print("✅ credit-risk-model structure created.")
