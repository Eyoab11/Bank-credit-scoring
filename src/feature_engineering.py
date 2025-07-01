import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import os

# --- Custom Transformers ---
# reusable transformation steps that work seamlessly with the library.

class AggregateFeatures(BaseEstimator, TransformerMixin):
    """
    Custom transformer to aggregate transaction-level data to customer-level.
    This creates the core RFM (Recency, Frequency, Monetary) features,
    along with other behavioral metrics.
    """
    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything from the data
        return self

    def transform(self, X, y=None):
        # Ensure 'TransactionStartTime' is in datetime format
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])

        # Create a 'snapshot date' for recency calculation. This is one day
        # after the last transaction in the dataset.
        snapshot_date = X['TransactionStartTime'].max() + pd.Timedelta(days=1)

        # --- Aggregate by CustomerId ---
        # Group by customer and aggregate various features
        aggregated_df = X.groupby('CustomerId').agg(
            # Recency: Days since last purchase
            Recency=('TransactionStartTime', lambda date: (snapshot_date - date.max()).days),
            # Frequency: Total number of transactions
            Frequency=('TransactionId', 'count'),
            # Monetary: Total value of all transactions
            Monetary=('Value', 'sum'),
            # Average transaction value
            AvgTransactionValue=('Value', 'mean'),
            # Standard deviation of transaction value
            StdTransactionValue=('Value', 'std'),
            # Number of unique product categories purchased
            NumUniqueProducts=('ProductCategory', 'nunique'),
            # Most frequent channel used by the customer
            MostFrequentChannel=('ChannelId', lambda x: x.mode()[0])
        ).reset_index()

        # Handle potential NaNs in StdTransactionValue for customers with only one transaction
        aggregated_df['StdTransactionValue'] = aggregated_df['StdTransactionValue'].fillna(0)

        return aggregated_df

class DropColumns(BaseEstimator, TransformerMixin):
    """
    Custom transformer to drop specified columns from a DataFrame.
    """
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Return a new DataFrame with the specified columns dropped
        return X.drop(columns=self.columns_to_drop, axis=1)

# --- Main Feature Engineering Pipeline ---

def create_feature_engineering_pipeline():
    """
    This function assembles and returns the full feature engineering pipeline.
    The pipeline will preprocess the customer-level aggregated data.
    """
    # Define which columns are numerical and which are categorical
    numeric_features = ['Recency', 'Frequency', 'Monetary', 'AvgTransactionValue', 'StdTransactionValue', 'NumUniqueProducts']
    categorical_features = ['MostFrequentChannel']

    # Create a pipeline for processing numerical features:
    # 1. Impute missing values with the median (robust to outliers)
    # 2. Scale features to have a mean of 0 and a standard deviation of 1
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Create a pipeline for processing categorical features:
    # 1. Impute missing values with the most frequent category
    # 2. One-hot encode the categories, converting them into numerical format
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a ColumnTransformer to apply different transformations to different columns
    # This is the core of preprocessing mixed data types.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns (like CustomerId) if any
    )

    # The final pipeline combines the preprocessor with a step to drop the original CustomerId
    # because it's an identifier, not a feature for the model.
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('column_dropper', DropColumns(columns_to_drop=['CustomerId'])) # Assuming CustomerId is at index 0
    ])

    # Note: The above DropColumns step is a bit tricky with ColumnTransformer.
    # A cleaner approach is to handle the CustomerId outside the pipeline
    # or ensure it is explicitly dropped by the preprocessor's remainder setting.
    # For now, will handle it outside.
    
    return preprocessor # return the preprocessor and handle ID columns outside.

# --- Execution Block ---
if __name__ == "__main__":
    print("Starting feature engineering process...")

    # Define file paths. Assuming you run from the `src` directory.
    raw_data_path = "../data/raw/data.csv"
    processed_data_dir = "../data/processed"
    processed_data_path = os.path.join(processed_data_dir, "processed_customer_data.csv")

    # Ensure the processed data directory exists
    os.makedirs(processed_data_dir, exist_ok=True)

    # 1. Load raw data
    df_raw = pd.read_csv(raw_data_path)

    # 2. Aggregate features to customer level
    aggregator = AggregateFeatures()
    df_customer_level = aggregator.transform(df_raw)
    print(f"Data aggregated to customer level. Shape: {df_customer_level.shape}")
    print("Sample of aggregated data:")
    print(df_customer_level.head())

    # 3. Create and apply the preprocessing pipeline
    preprocessing_pipeline = create_feature_engineering_pipeline()

    # Get the column names before transforming
    numeric_features = ['Recency', 'Frequency', 'Monetary', 'AvgTransactionValue', 'StdTransactionValue', 'NumUniqueProducts']
    
    # Fit and transform the data
    processed_data_array = preprocessing_pipeline.fit_transform(df_customer_level)


    # 4. Convert the processed NumPy array back to a DataFrame
    
    # Get the feature names from the OneHotEncoder step
    ohe_feature_names = preprocessing_pipeline.named_transformers_['cat']\
        .named_steps['onehot'].get_feature_names_out(['MostFrequentChannel'])

    # The final columns will be the numeric features (which were scaled)
    # plus the new one-hot encoded features.
    final_feature_names = numeric_features + list(ohe_feature_names)
    
    # The 'remainder' columns ('CustomerId') are added at the end by ColumnTransformer.
    # Let's find out which columns were considered 'remainder'.
    # This is a robust way to get the remainder columns in their original order.
    processed_cols = numeric_features + ['MostFrequentChannel']
    remainder_cols = [col for col in df_customer_level.columns if col not in processed_cols]

    # The processed_data_array contains all columns.
    # Let's create the DataFrame with all columns first.
    all_final_columns = final_feature_names + remainder_cols
    
    df_processed = pd.DataFrame(processed_data_array, columns=all_final_columns)

    # Now, df_processed has the scaled/encoded features AND the CustomerId.
    # Use CustomerId for merging later, but it shouldn't be a feature.
    # Let's reorder to have CustomerId at the front, which is conventional.
    if 'CustomerId' in df_processed.columns:
        # Get all columns except CustomerId
        feature_cols = [col for col in df_processed.columns if col != 'CustomerId']
        # Re-create the DataFrame with CustomerId first
        df_processed = df_processed[['CustomerId'] + feature_cols]

    print(f"\nPreprocessing complete. Final data shape: {df_processed.shape}")
    print("Sample of final processed data:")
    print(df_processed.head())

    # 5. Save the final, processed DataFrame
    df_processed.to_csv(processed_data_path, index=False)
    print(f"\nProcessed data saved to {processed_data_path}")