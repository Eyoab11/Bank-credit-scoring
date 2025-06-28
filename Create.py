import os

folders = [
    "data/raw", "data/processed",
    "notebooks", "scripts",
    "reports/figures", "reports/tables",
    "models"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create basic files
with open("README.md", "w") as f:
    f.write("# Bati Bank Credit Scoring Project\n\n## Project Summary\n...")
with open(".gitignore", "w") as f:
    f.write("data/\nmodels/\n__pycache__/\n.ipynb_checkpoints/\n")
with open("requirements.txt", "w") as f:
    f.write("pandas\nscikit-learn\nmatplotlib\nseaborn\njupyter\n")

print("✅ Project folder structure created.")
