# Data processing and feature engineering logic

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import os

# --- Custom Transformers ---
class AggregateFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        snapshot_date = X['TransactionStartTime'].max() + pd.Timedelta(days=1)
        aggregated_df = X.groupby('CustomerId').agg(
            Recency=('TransactionStartTime', lambda date: (snapshot_date - date.max()).days),
            Frequency=('TransactionId', 'count'),
            Monetary=('Value', 'sum'),
            AvgTransactionValue=('Value', 'mean'),
            StdTransactionValue=('Value', 'std'),
            NumUniqueProducts=('ProductCategory', 'nunique'),
            MostFrequentChannel=('ChannelId', lambda x: x.mode()[0])
        ).reset_index()
        aggregated_df['StdTransactionValue'] = aggregated_df['StdTransactionValue'].fillna(0)
        return aggregated_df

def create_feature_engineering_pipeline():
    numeric_features = ['Recency', 'Frequency', 'Monetary', 'AvgTransactionValue', 'StdTransactionValue', 'NumUniqueProducts']
    categorical_features = ['MostFrequentChannel']
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='passthrough')
    return preprocessor

if __name__ == "__main__":
    print("Starting data processing and feature engineering...")
    raw_data_path = "../data/raw/data.csv"
    processed_data_dir = "../data/processed"
    processed_data_path = os.path.join(processed_data_dir, "processed_customer_data.csv")
    os.makedirs(processed_data_dir, exist_ok=True)
    df_raw = pd.read_csv(raw_data_path)
    aggregator = AggregateFeatures()
    df_customer_level = aggregator.transform(df_raw)
    print(f"Data aggregated to customer level. Shape: {df_customer_level.shape}")
    print(df_customer_level.head())
    preprocessing_pipeline = create_feature_engineering_pipeline()
    numeric_features = ['Recency', 'Frequency', 'Monetary', 'AvgTransactionValue', 'StdTransactionValue', 'NumUniqueProducts']
    processed_data_array = preprocessing_pipeline.fit_transform(df_customer_level)
    ohe_feature_names = preprocessing_pipeline.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(['MostFrequentChannel'])
    final_feature_names = numeric_features + list(ohe_feature_names)
    processed_cols = numeric_features + ['MostFrequentChannel']
    remainder_cols = [col for col in df_customer_level.columns if col not in processed_cols]
    all_final_columns = final_feature_names + remainder_cols
    df_processed = pd.DataFrame(processed_data_array, columns=all_final_columns)
    if 'CustomerId' in df_processed.columns:
        feature_cols = [col for col in df_processed.columns if col != 'CustomerId']
        df_processed = df_processed[['CustomerId'] + feature_cols]
    print(f"\nPreprocessing complete. Final data shape: {df_processed.shape}")
    print(df_processed.head())
    df_processed.to_csv(processed_data_path, index=False)
    print(f"\nProcessed data saved to {processed_data_path}")
