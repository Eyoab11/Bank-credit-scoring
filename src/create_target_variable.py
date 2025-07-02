import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_clusters(df_clustered):
    """
    Analyzes the characteristics of each cluster to identify the high-risk group.
    High-risk is typically defined by high recency (not recent), low frequency, and low monetary value.
    
    Args:
        df_clustered (pd.DataFrame): DataFrame with customer data and a 'Cluster' column.
        
    Returns:
        int: The cluster label identified as high-risk.
    """
    # Calculate the mean of RFM values for each cluster
    cluster_analysis = df_clustered.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).reset_index()

    print("\n--- Cluster Analysis (Mean RFM Values) ---")
    print(cluster_analysis)

    # Identify the high-risk cluster based on RFM characteristics:
    # - Highest Recency (customers who haven't purchased in a long time)
    # - Lowest Frequency (few purchases)
    # - Lowest Monetary value (low total spending)
    
    # Prioritize high recency and low frequency/monetary.
    # A simple scoring system can help automate this:
    # Rank clusters for each metric (lower is better for R, higher for F/M)
    cluster_analysis['R_Rank'] = cluster_analysis['Recency'].rank(ascending=False) # Higher recency is worse - descending rank
    cluster_analysis['F_Rank'] = cluster_analysis['Frequency'].rank(ascending=True) # Lower frequency is worse
    cluster_analysis['M_Rank'] = cluster_analysis['Monetary'].rank(ascending=True) # Lower monetary is worse

    # Composite score: sum of ranks
    cluster_analysis['Risk_Score'] = cluster_analysis['R_Rank'] + cluster_analysis['F_Rank'] + cluster_analysis['M_Rank']
    
    print("\n--- Cluster Risk Scoring ---")
    print(cluster_analysis)

    # The cluster with the highest risk score is our target
    high_risk_cluster = cluster_analysis.sort_values(by='Risk_Score', ascending=False).iloc[0]['Cluster']
    
    print(f"\nIdentified High-Risk Cluster: {high_risk_cluster}")
    
    return int(high_risk_cluster)

def visualize_clusters(df_clustered):
    """
    Creates and saves visualizations of the customer clusters.
    """
    plot_dir = "../plots/task-4"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Create a pairplot to see distributions and relationships
    plt.figure()
    sns.pairplot(df_clustered, hue='Cluster', vars=['Recency', 'Frequency', 'Monetary'], palette='viridis')
    plt.suptitle('RFM Cluster Pairplot', y=1.02)
    plt.savefig(f"{plot_dir}/rfm_cluster_pairplot.png")
    plt.close()

    # Boxplots for each RFM variable by cluster
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.boxplot(data=df_clustered, x='Cluster', y='Recency', ax=axes[0], palette='viridis')
    axes[0].set_title('Recency by Cluster')
    sns.boxplot(data=df_clustered, x='Cluster', y='Frequency', ax=axes[1], palette='viridis')
    axes[1].set_title('Frequency by Cluster')
    sns.boxplot(data=df_clustered, x='Cluster', y='Monetary', ax=axes[2], palette='viridis')
    axes[2].set_title('Monetary by Cluster')
    
    fig.suptitle('RFM Characteristics by Cluster')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{plot_dir}/rfm_cluster_boxplots.png")
    plt.close()
    
    print(f"Cluster visualizations saved to {plot_dir}")


if __name__ == "__main__":
    print("Starting Task 4: Proxy Target Variable Engineering...")

    # Define file paths
 
    raw_data_path = "../data/raw/data.csv"
    processed_data_path = "../data/processed/processed_customer_data.csv"
    final_training_data_path = "../data/processed/final_training_data.csv"

    # --- Step 1: Get Aggregated Customer-Level Data (Unscaled) ---
    
    df_raw = pd.read_csv(raw_data_path)
    # A simplified aggregation for RFM
    df_raw['TransactionStartTime'] = pd.to_datetime(df_raw['TransactionStartTime'])
    snapshot_date = df_raw['TransactionStartTime'].max() + pd.Timedelta(days=1)
    
    df_rfm = df_raw.groupby('CustomerId').agg(
        Recency=('TransactionStartTime', lambda date: (snapshot_date - date.max()).days),
        Frequency=('TransactionId', 'count'),
        Monetary=('Value', 'sum')
    ).reset_index()

    print(f"Successfully created RFM data. Shape: {df_rfm.shape}")

    # --- Step 2: Pre-process (Scale) the RFM Features for Clustering ---
    # K-Means is sensitive to the scale of features.
    rfm_features = df_rfm[['Recency', 'Frequency', 'Monetary']]
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_features)

    print("RFM features scaled for K-Means.")

    # --- Step 3: Cluster Customers using K-Means ---
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled)
    
    # Assign the cluster labels back to our original RFM dataframe
    df_rfm['Cluster'] = kmeans.labels_
    print("K-Means clustering complete. Customers assigned to 3 clusters.")

    # --- Step 4: Analyze Clusters and Identify High-Risk Group ---
    # The 'analyze_clusters' function will tell us which cluster label (0, 1, or 2) is high-risk.
    high_risk_cluster_label = analyze_clusters(df_rfm)
    
    # Visualize the clusters to confirm our analysis
    visualize_clusters(df_rfm)

    # --- Step 5: Create the 'is_high_risk' Target Column ---
    # Assign 1 if the customer is in the high-risk cluster, 0 otherwise.
    df_rfm['is_high_risk'] = np.where(df_rfm['Cluster'] == high_risk_cluster_label, 1, 0)
    
    print("\n'is_high_risk' target column created.")
    print("Distribution of the new target variable:")
    print(df_rfm['is_high_risk'].value_counts(normalize=True))

    # --- Step 6: Integrate the Target Variable with the Main Processed Dataset ---
    # Load the main processed dataset from Task 3
    df_processed = pd.read_csv(processed_data_path)
    
    # Select only the CustomerId and the new target column for merging
    df_target = df_rfm[['CustomerId', 'is_high_risk']]
    
    # Merge the target column into the processed dataset
    df_final_training = pd.merge(df_processed, df_target, on='CustomerId', how='left')
    
    # Ensure the merge was successful and there are no NaNs in the target column
    if df_final_training['is_high_risk'].isnull().any():
        print("Warning: Null values found in target column after merge. Filling with 0.")
        df_final_training['is_high_risk'] = df_final_training['is_high_risk'].fillna(0).astype(int)

    # The CustomerId is no longer needed for model training, so we can drop it.
    df_final_training = df_final_training.drop(columns=['CustomerId'])
    
    print(f"\nTarget variable successfully merged. Final training data shape: {df_final_training.shape}")
    
    # Save the final, model-ready dataset
    df_final_training.to_csv(final_training_data_path, index=False)
    print(f"Final training data with target variable saved to: {final_training_data_path}")