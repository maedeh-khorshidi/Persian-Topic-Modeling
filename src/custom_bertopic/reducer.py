from umap import UMAP


class UMAPReducer:

    def __init__(
        self,
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine"
    ):

        self.umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            metric=metric,
            random_state=42
        )

    def fit_transform(self, embeddings):

        reduced_embeddings = (
            self.umap_model
            .fit_transform(embeddings)
        )

        return reduced_embeddings