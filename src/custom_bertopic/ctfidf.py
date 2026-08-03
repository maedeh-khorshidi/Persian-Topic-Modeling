import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import stopwords


class ClassTFIDF:

    def __init__(
        self,
        ngram_range=(1, 1),
        stop_words=stopwords.stop_words
    ):

        self.vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            stop_words=stop_words
        )

        self.transformer = TfidfTransformer()

        self.ctfidf_matrix = None
        self.topic_words = None

    def create_topic_documents(
        self,
        documents,
        labels
    ):

        df = pd.DataFrame(
            {
                "Document": documents,
                "Topic": labels
            }
        )

        topic_documents = (
            df.groupby("Topic")["Document"]
            .apply(" ".join)
            .reset_index()
        )

        return topic_documents

    def fit_transform(
        self,
        documents,
        labels
    ):

        topic_documents = self.create_topic_documents(
            documents,
            labels
        )

        count_matrix = self.vectorizer.fit_transform(
            topic_documents["Document"]
        )

        self.ctfidf_matrix = (
            self.transformer.fit_transform(
                count_matrix
            )
        )

        self.topic_words = self.get_topic_words()

        return self.ctfidf_matrix, self.topic_words


    def get_topic_words(
        self,
        top_n=10
    ):

        feature_names = (
            self.vectorizer.get_feature_names_out()
        )

        topic_words = {}

        for topic_index in range(
            self.ctfidf_matrix.shape[0]
        ):

            scores = (
                self.ctfidf_matrix
                .getrow(topic_index)
                .toarray()[0]
            )

            top_indices = scores.argsort()[-top_n:][::-1]

            topic_words[topic_index] = [
                (
                    feature_names[i],
                    float(scores[i])
                )
                for i in top_indices
            ]

        self.topic_words = topic_words

        return topic_words