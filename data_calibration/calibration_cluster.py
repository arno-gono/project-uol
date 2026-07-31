from config import KAGGLE_DATASET_NAME, MAX_CARDINALITY_NB
from data.sqlite_connector import connecting_to_sqlite
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

NB_SAMPLES = 10_000


def draft_kmeans(df: pd.DataFrame):

    # Storing the best score obtained and which K in memory
    best_score = 0
    best_k = 0

    # Output of the best model
    best_kmeans = None

    # The numerical data needs to be scaled, otherwise high values gets weighted higher by the model
    scaler = StandardScaler()

    # Testing with numerical columns first
    cols = [
        "AMT_INCOME_TOTAL",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "CNT_FAM_MEMBERS"
    ]

    # Iterating over 10k rows at the moment in order to iterate fast
    df_scaled = df[cols].sample(NB_SAMPLES)
    scaled = scaler.fit_transform(df_scaled)

    for k in range(2, 10):
        # Instantiating the model
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled)

        # Scoring the model: -1 to 1, higher is better
        score = silhouette_score(scaled, labels)

        # Saving this model if it performs better than the one selected so far.
        if score > best_score:
            best_score = score
            best_k = k
            best_kmeans = kmeans
        print(f"k={k}  silhouette={score:.2f}")

    # Transforming centroids values back to original scales
    centroids = scaler.inverse_transform(best_kmeans.cluster_centers_)

    # Profile of the clusters: there are K clusters, each one has a centroid.
    # Each data point belongs to a cluster (given by the "labels" attribute).
    # The centroid is expressed as value for each attribute, ie each axis of the data point
    # is attributed a value that corresponds to the centroid of that cluster.
    clusters = []
    for c in range(best_k):
        clusters.append({
            "share": float((best_kmeans.labels_ == c).mean()),
            "centroid": {col: v for col, v in zip(df_scaled.columns, centroids[c])}
        })

    return {"best_score": best_score,
            "best_k": best_k,
            "clusters": clusters}


if __name__ == "__main__":
    co = connecting_to_sqlite(KAGGLE_DATASET_NAME)
    df = pd.read_sql("SELECT * FROM application_record", co)

    kmeans = draft_kmeans(df)