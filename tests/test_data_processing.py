import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the custom transformer from your src directory
from src.feature_engineering import AggregateFeatures

@pytest.fixture
def sample_raw_data():
    """
    Creates a sample raw transaction DataFrame for testing.
    """
    now = datetime.now()
    data = {
        'TransactionId': [f'T{i}' for i in range(5)],
        'CustomerId': ['C1', 'C2', 'C1', 'C3', 'C2'],
        'TransactionStartTime': [
            now - timedelta(days=10), # C1
            now - timedelta(days=5),  # C2
            now - timedelta(days=1),  # C1 (most recent)
            now - timedelta(days=20), # C3
            now - timedelta(days=2)   # C2 (most recent)
        ],
        'Value': [100, 200, 50, 500, 150],
        'ProductCategory': ['A', 'B', 'A', 'C', 'A'],
        'ChannelId': ['Web', 'Mobile', 'Web', 'Mobile', 'Mobile']
    }
    return pd.DataFrame(data)

def test_aggregate_features_output_shape(sample_raw_data):
    """
    Test 1: Ensures the output DataFrame has the correct shape.
    The number of rows should be the number of unique customers.
    The number of columns should be 8 (CustomerId + 7 aggregated features).
    """
    transformer = AggregateFeatures()
    result = transformer.transform(sample_raw_data)
    
    # There are 3 unique customers in the sample data (C1, C2, C3)
    assert result.shape[0] == 3
    # There should be 8 columns as defined in the transformer
    assert result.shape[1] == 8
    print("Test for output shape passed.")

def test_aggregate_features_recency_calculation(sample_raw_data):
    """
    Test 2: Verifies that the Recency calculation is correct.
    """
    transformer = AggregateFeatures()
    result = transformer.transform(sample_raw_data)
    
    # For customer C1, the most recent transaction was 1 day ago.
    # The snapshot date is 1 day after the max date (which is now - 1 day),
    # so Recency should be (now - (now - 1 day)) = 1 day.
    c1_recency = result[result['CustomerId'] == 'C1']['Recency'].iloc[0]
    assert c1_recency == 1

    # For customer C3, the only transaction was 20 days ago.
    c3_recency = result[result['CustomerId'] == 'C3']['Recency'].iloc[0]
    assert c3_recency == 20
    print("Test for recency calculation passed.")
    
def test_aggregate_features_frequency_and_monetary(sample_raw_data):
    """
    Test 3: Verifies Frequency and Monetary calculations.
    """
    transformer = AggregateFeatures()
    result = transformer.transform(sample_raw_data)
    
    # For customer C1 (2 transactions, Value 100 + 50 = 150)
    c1_data = result[result['CustomerId'] == 'C1']
    assert c1_data['Frequency'].iloc[0] == 2
    assert c1_data['Monetary'].iloc[0] == 150

    # For customer C2 (2 transactions, Value 200 + 150 = 350)
    c2_data = result[result['CustomerId'] == 'C2']
    assert c2_data['Frequency'].iloc[0] == 2
    assert c2_data['Monetary'].iloc[0] == 350
    print("Test for frequency and monetary calculation passed.")