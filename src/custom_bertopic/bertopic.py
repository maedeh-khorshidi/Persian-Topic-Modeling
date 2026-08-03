from encoder import ParsBERTEncoder
from reducer import UMAPReducer
from cluster import HDBSCANClusterer
from ctfidf import ClassTFIDF


class BertTopic:

    def __init__(
        self,
        embedding_model=None,
        umap_model=None,
        cluster_model=None,
        ctfidf_model=None
    ):

        self.embedding_model = (
            embedding_model
            if embedding_model is not None
            else ParsBERTEncoder()
        )

        self.umap_model = (
            umap_model
            if umap_model is not None
            else UMAPReducer()
        )

        self.cluster_model = (
            cluster_model
            if cluster_model is not None
            else HDBSCANClusterer()
        )

        self.ctfidf_model = (
            ctfidf_model
            if ctfidf_model is not None
            else ClassTFIDF()
        )

        self.embeddings = None
        self.reduced_embeddings = None
        self.labels = None
        self.topic_words = None
        self.ctfidf_matrix = None


    def fit_transform(self, documents):

        # 1. Embedding extraction
        self.embeddings = (
            self.embedding_model
            .encode(documents)
        )
        print("Embeddings:", self.embeddings.shape)


        # # Tensor -> NumPy
        # self.embeddings = (
        #     self.embeddings
        #     .cpu()
        #     .numpy()
        # )


        # 2. UMAP dimensionality reduction
        self.reduced_embeddings = (
            self.umap_model
            .fit_transform(
                self.embeddings
            )
        )
        print("Reduced Embeddings:", self.reduced_embeddings.shape)


        # 3. HDBSCAN clustering
        self.labels = (
            self.cluster_model
            .fit_predict(
                self.reduced_embeddings
            )
        )
        print("Labels:", self.labels.shape)


        # 4. Topic representation
        (
            self.ctfidf_matrix,
            self.topic_words
        ) = (
            self.ctfidf_model
            .fit_transform(
                documents,
                self.labels
            )
        )


        return (
            self.labels,
            self.topic_words
        )
    

    def get_topic(
        self,
        topic_id
    ):

        if self.topic_words is None:
            raise ValueError(
                "The model has not been fitted yet. "
                "Call fit_transform() first."
            )

        return self.topic_words.get(
            topic_id,
            None
        )