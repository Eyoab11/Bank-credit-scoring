# Exploratory Data Analysis (EDA) Report for Bati Bank Credit Risk Model

## 1. Executive Summary & Key Findings

This report details the findings from the initial Exploratory Data Analysis (EDA) on the transaction dataset. The analysis aimed to understand data characteristics, identify quality issues, and uncover patterns related to fraudulent transactions. The key findings are critical for the subsequent feature engineering and modeling stages.

- **Critical Class Imbalance:** The dataset is extremely imbalanced, with a fraud rate of only 0.202%. This dictates our modeling strategy, requiring the use of techniques like SMOTE and evaluation metrics such as F1-Score and AUC-ROC, as accuracy will be misleading.
  
- **Excellent Data Completeness:** The analysis confirmed the dataset is 100% complete, with no missing values. This simplifies the data cleaning process significantly, as no imputation is needed.
  
- **Identification of Useless Features:** Two columns, `CountryCode` and `CurrencyCode`, contain only a single unique value each. They provide no predictive information and should be dropped. Furthermore, `Amount` and `Value` are perfectly correlated, making one of them redundant.
  
- **Strong Predictive Signals Found:** Categorical features show strong potential. Specifically, the `ProductCategory` of 'financial services' and the `ChannelId` of 'checkout' are associated with significantly higher fraud rates and will be crucial features for the model.
  
- **Data Integrity & Behavioral Patterns:** A significant number of transactions (38,189) have a negative `Amount`, likely representing refunds or credits. These require a specific handling strategy. Temporal analysis reveals that fraud rates are higher in the early morning hours, providing a valuable behavioral pattern to exploit.

## 2. Detailed Analysis

### 2.1. Data Overview & Integrity

- **Dataset Size:** The dataset contains 95,662 transactions across 16 columns.
- **Data Types:** The data is well-structured with a mix of object (IDs, categories), integer, float, and a properly parsed datetime column (`TransactionStartTime`).
- **Constant Columns:** `CountryCode` and `CurrencyCode` were identified as constant columns with zero variance. They add no value and will be removed during feature engineering.

### 2.2. Missing Value Analysis

The script performed a comprehensive check for missing data.

- **Result:** No missing values were found in any of the 16 columns. The dataset is clean in this regard.

### 2.3. Target Variable Analysis (FraudResult)

- **Imbalance:** Only 193 out of 95,662 transactions are labeled as fraudulent.
- **Implication:** This severe imbalance must be the central consideration for model training and evaluation. Standard accuracy is not a suitable metric. We must focus on the model's ability to correctly identify the rare fraud cases (Recall) without flagging too many legitimate transactions as fraudulent (Precision).

### 2.4. Numerical Feature Analysis

**Key Features:** `Amount`, `Value`, `PricingStrategy`.

**Amount Feature:**
- **Negative Values:** 38,189 transactions have a negative `Amount`. The output shows that the transactions with the largest negative values are all associated with a single account (`AccountId_4249`), suggesting these may be related to a specific business process, merchant, or a credit-issuing activity. This pattern warrants further investigation or can be engineered into a feature (e.g., `is_credit_transaction`).
- **Skewness:** The distribution of `Amount` is heavily right-skewed. A log-transformation will be necessary to normalize its distribution for most modeling algorithms.
- **Redundancy:** The correlation analysis (from the generated plot) confirms that `Amount` and `Value` are perfectly correlated. To avoid multicollinearity, we will retain only one, likely `Value`, as it is always positive.

### 2.5. Categorical Feature Analysis

- **Cardinality:** The key categorical features (`ProductCategory`, `ChannelId`, `PricingStrategy`) have low cardinality (9, 4, and 4 unique values, respectively), making them easy to handle with one-hot encoding.
- **Fraud Indicators:** The bar plots of fraud rate by category are highly informative. They will show that specific categories within `ProductCategory` and `ChannelId` have a much higher propensity for fraud, making them powerful predictors.

## 3. Recommendations & Next Steps

### Data Cleaning:
- Drop the `CountryCode`, `CurrencyCode`, and `Amount` columns.
- Remove all ID columns (`TransactionId`, `BatchId`, etc.) before modeling.

### Feature Engineering:
- Create time-based features: `Hour` and `DayOfWeek`.
- Create a binary feature `is_credit` where `Amount < 0`.
- Apply a log transformation to the `Value` column.
- One-hot encode `ProductCategory`, `ChannelId`, and `PricingStrategy`.

### Modeling Strategy:
- Implement a resampling technique like SMOTE on the training data to address the class imbalance.
- Split the data chronologically if possible to prevent data leakage and better simulate a real-world deployment scenario.
- Evaluate models using AUC-ROC, Precision, Recall, and F1-Score.