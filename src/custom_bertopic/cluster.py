from hdbscan import HDBSCAN

class HDBSCANClusterer:

    def __init__(
        self,
        min_cluster_size=10,
        metric="euclidean"
    ):

        self.cluster_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric=metric,
            prediction_data=True
        )

    def fit_predict(
        self,
        reduced_embeddings
    ):

        labels = self.cluster_model.fit_predict(
            reduced_embeddings
        )

        return labels