# 🏦 Alternative Data Credit Risk Model   

### Folder structure
```
│ README.md
│ requirements.txt
│ Dockerfile
│
├───data
│ ├───raw
│ │ Xente_dataset.csv
│ │ Xente_Variable_Definitions.csv
│ │
│ └───processed
│ cleaned_data.csv
│ features.csv
│
├───notebooks
│ 1.0-eda.ipynb
│
├───plots
│ └───task-2
│ amount_dist.png
│ fraud_by_hour.png
│ category_fraud.png
│ correlation_heatmap.png
│
├───src
│ │ data_processing.py
│ │ train.py
│ │ predict.py
│ │
│ └───api
│ main.py
│ pydantic_models.py
│
└───tests
test_data_processing.py
test_models.py

```

## ✔️ Task 1: Data Collection & Understanding

- Loaded dataset: `data.csv` from Xente's transaction platform  
- Reference documentation: `Xente_Variable_Definitions.csv` for feature meanings  
- Identified target variable:  
  - `FraudResult` (1 = Fraud, 0 = Not Fraud)

## ✔️ Task 2: Exploratory Data Analysis (EDA)

### Data Distribution Analysis
- Analyzed class balance: Fraud vs Non-Fraud cases
- Examined transaction patterns:
  - Timing by hour of day
  - Amount distributions

### Categorical Variable Investigation
Visualized relationships between `FraudResult` and:
- `ProductCategory` 
- `ChannelId`
- `PricingStrategy` 
- `CountryCode`
- `CurrencyCode`

### Numerical Analysis
- Generated correlation heatmap for numeric features

### Outputs
All plots saved to: `plots/task-2/`

### Directory
EDA is done in `notebooks/1.0-eda.ipynb/`


# Credit Scoring: Business Understanding

This section outlines the business context for building a credit scoring model, focusing on:
- Regulatory requirements
- Data-driven challenges  
- Model selection trade-offs

---
## 1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

## Basel II Compliance and Model Interpretability Requirements

### Regulatory Mandates:
- **Basel II Accord** requires strong internal procedures for capital adequacy evaluation
- **Pillar 2 (Supervisory Review Process)** imposes strict requirements for IRB approach banks

### Key Implications:
✔ **Regulatory scrutiny** of internal credit risk models  
✔ Models must be:
  - **Understandable** in logic
  - **Verifiable** in assumptions  
  - **Contestable** in results

### Business Necessity:
- **Interpretable models** (e.g., Logistic Regression with Weight of Evidence) are:
  - Not just technical preferences
  - **Critical compliance requirements**
- Benefits:
  - Transparent risk measurement demonstration
  - Assurance of reliable, equitable non-"black box" models
  - Regulatory approval for capital calculations

---

## 2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

## Proxy Variables: Necessity and Risks

### Why Proxies Are Needed:
- True "default" = 90+ days past due (often unavailable)
- Proxy solution (e.g., 30+ days past due = "high risk") enables:
  - Predictive model training
  - Proactive risk identification

### Business Risks:

| Risk Type | Consequences |
|-----------|-------------|
| **False Positives** | - Rejecting creditworthy applicants <br> - Lost revenue <br> - Poor customer experience <br> - Market share erosion |
| **False Negatives** | - Approving bad loans <br> - Financial losses <br> - Increased collection costs <br> - Higher capital provisions |

### Critical Challenge:
Ensuring proxy is a **strong, stable indicator** of true default to avoid unreliable models.

---

## 3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

## Model Selection Trade-offs

### Simple Model (Logistic Regression with WoE)

**✅ Advantages**  
- High interpretability & transparency  
- Clear variable contribution explanation  
- Easier regulatory/compliance validation  
- Better fairness/bias assessment  

**❌ Limitations**  
- Lower predictive power  
- May miss complex patterns  
- Potentially suboptimal decisions  

### Complex Model (Gradient Boosting)

**✅ Advantages**  
- Superior accuracy (higher AUC)  
- Better risk segmentation  
- Lower default rates  
- Higher profitability potential  

**❌ Limitations**  
- Black box nature  
- Regulatory approval challenges  
- Difficult bias/fairness verification  

### Regulatory Context Conclusion:
For **core risk models**, interpretability often outweighs marginal performance gains due to:
- Compliance overhead  
- Justification requirements  
- Capital calculation scrutiny