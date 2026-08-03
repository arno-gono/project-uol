from config import KAGGLE_DATASET_NAME, MAX_CARDINALITY_NB
from data.sqlite_connector import connecting_to_sqlite
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

NB_SAMPLES = 10_000


def _get_kmeans_profile(df: pd.DataFrame):

    # Storing the best score obtained and which K in memory
    best_score = -1
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
    df_scaled = df[cols].sample(min(NB_SAMPLES, len(df)), random_state=42)
    scaled = scaler.fit_transform(df_scaled)

    scores_per_k = {}

    for k in range(2, 10):
        # Instantiating the model
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled)

        # Scoring the model: -1 to 1, higher is better
        score = silhouette_score(scaled, labels)

        # Saving scores per k:
        scores_per_k[f"K_{k}"] = round(float(score), 4)

        # Saving this model if it performs better than the one selected so far.
        if score > best_score:
            best_score = score
            best_k = k
            best_kmeans = kmeans

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
            "clusters": clusters,
            "scores_per_k": scores_per_k,
            }


def _get_disguised_missing_values(df: pd.DataFrame):

    # Highlighting disguised missing values ie numerical placeholders that act as NULL.
    # Using Tukey's Fence method, a statistical approach to isolate outliers, but with more conservative parameters
    # to leave outliers in the dataset and only flag potential placeholders. Hypothesis: if the top value in a column
    # appears more than param_frequency percent in the column and is outside the fences by a
    # multiplier of param_tukeys_fence_iqr_multiplier, it is considered to be a disguised missing value.
    # To be tested and validated over a large number of datasets.

    param_frequency = 0.01
    param_tukeys_fence_iqr_multiplier = 10

    # Needs to be changed to numerical columns only
    cols = [
        "AMT_INCOME_TOTAL",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "CNT_FAM_MEMBERS"
    ]

    suspects = {}

    for col in cols:

        # The .mode() method returns the value with highest frequency in the column.
        top_value = df[col].mode()[0]

        # Calculating how frequently this value appears in the dataset.
        frequency = df[df[col] == top_value][col].count() / len(df)

        # Comparing that value against the rest of the column. Calculating IQR as per Tukey's Fences method.
        # Programming example: https://www.askpython.com/python/examples/how-to-determine-outliers
        other_values = df[df[col] != top_value][col]
        q1 = other_values.quantile(0.25)
        q3 = other_values.quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - param_tukeys_fence_iqr_multiplier * iqr
        upper_fence = q3 + param_tukeys_fence_iqr_multiplier * iqr

        # Calculating if the top value is outside the fences.
        is_outside_fences = top_value < lower_fence or top_value > upper_fence

        # If a value is outside the fences and appears in more than param_frequency% of rows,
        # it is considered as a disguised missing value: FAHES approach
        if frequency > param_frequency and is_outside_fences:
            # Calculating how far the disguised missing value is from the fence. Dividing by IQR for scaling.
            if top_value > upper_fence:
                gap_to_fence = (top_value - upper_fence) / iqr
            else:
                gap_to_fence = (lower_fence - top_value) / iqr

            suspects[col] = {"value": float(top_value),
                             "frequency": float(frequency),
                             "gap_to_fence_iqr": round(float(gap_to_fence), 2)}
    return suspects


def _get_pca_profile(df: pd.DataFrame):
    # Dimensionality reduction using Principal Component Analysis (PCA).
    # PCA identifies the axes (principal components) that account for the largest amount of variance
    # in the dataset.
    # Reference: Géron, A. (2022). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.
    # 3rd Edition, Chapter 8: Dimensionality Reduction, O'Reilly Media.

    cols = [
        "ID",
        "CNT_CHILDREN",
        "AMT_INCOME_TOTAL",
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "CNT_FAM_MEMBERS",
        "FLAG_PHONE"
    ]

    # Standardising data first.
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[cols].dropna())

    # Fitting PCA with all components to inspect how much variance each one explains.
    pca = PCA()
    pca.fit(scaled)

    # explained_variance_ratio_: array where each entry is the share of total variance
    # captured by that component. The first component always captures the most.
    explained = pca.explained_variance_ratio_

    # Cumulative sum: shows how many components are needed to reach a defined threshold.
    cumsum = np.cumsum(explained)

    # Finding the minimum number of components to retain 95% of the variance, ie up to the first component above
    # 95% in the cumulative array, as per Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.
    param_variance_threshold = 0.95
    n_dimensions = int(np.argmax(cumsum >= param_variance_threshold) + 1)

    # Output: variance explained per component and the suggested number to keep.
    components = []
    for i, (var, cum) in enumerate(zip(explained, cumsum)):
        components.append({
            "component": i + 1,
            "variance_explained": round(float(var), 4),
            "cumulative_variance": round(float(cum), 4),
            "dimension_kept": True if i + 1 <= n_dimensions else False,
        })

    return {
        "n_original_features": len(cols),
        "param_variance_threshold": param_variance_threshold,
        "nb_dimensions": n_dimensions,
        "components": components
    }


def get_ml_profile(df: pd.DataFrame):

    # Disguised missing values are placeholders that can distort data, creating artificial outliers
    disguised_missing = _get_disguised_missing_values(df)

    df_clean = df.copy()
    for col, v in disguised_missing.items():
        df_clean = df_clean[df_clean[col] != v["value"]]

    # Reducing the dimensionality of the dataset to its main drivers
    pca_profile = _get_pca_profile(df_clean)

    # Getting a KMeans profile i.e. mapping clusters of data and complex relationships in the dataset
    kmeans_profile = _get_kmeans_profile(df_clean)

    return {
        "nb_entries_profiled": len(df_clean),
        "nb_entries_dropped": len(df) - len(df_clean),
        "disguised_missing_values": disguised_missing,
        "pca": pca_profile,
        "kmeans": kmeans_profile,
    }



if __name__ == "__main__":
    co = connecting_to_sqlite(KAGGLE_DATASET_NAME)
    df = pd.read_sql("SELECT * FROM application_record", co)

    q = get_ml_profile(df=df)
