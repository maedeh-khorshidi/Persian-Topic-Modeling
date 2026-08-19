import numpy as np
import random
import pandas as pd
from collections import Counter
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sklearn.metrics import normalized_mutual_info_score
from config import CATEGORY_COLUMN, COHERENCE_METHOD, TOP_N_DIVERSITY, CSV_ENCODING, COHERENCE_SIZE_SAMPLE


class TopicEvaluator:

    def __init__(
        self,
        topic_model,
        documents,
        topics,
        probabilities=None,
        dataframe=None
    ):

        self.topic_model = topic_model
        self.documents = documents
        self.topics = np.array(topics)
        self.probabilities = probabilities
        self.df = dataframe
        self.metrics = {}
        self.metrics_df = None
        self.tokenized_documents = self._tokenize_documents()
        self.topic_words = self._get_topics()


    # --------------------------------------------------
    # Private Methods
    # --------------------------------------------------

    def _tokenize_documents(self):
        """
        TODO:
        Replace with Hazm tokenizer in future versions.
        """
        return [doc.split() for doc in self.documents]


    def _get_topics(self):

        topic_dict = self.topic_model.get_topics()
        topics = []

        for topic_id, topic_words in topic_dict.items():

            if topic_id == -1:
                continue

            words = [word for word, _ in topic_words]
            topics.append(words)

        return topics


    # --------------------------------------------------
    # Evaluation Metrics
    # --------------------------------------------------

    def evaluate_coherence(
        self,
        sample_size=None,
        random_state=42
    ):
        """
        Evaluate Topic Coherence (c_v).

        Parameters
        ----------
        sample_size : int
            Number of documents used for coherence calculation.
        random_state : int
            Seed for reproducible sampling.
        """

        sample_size = COHERENCE_SIZE_SAMPLE
        # Use all documents if sample_size is larger
        sample_size = min(
            sample_size,
            len(self.tokenized_documents)
        )

        # Reproducible sampling
        random.seed(random_state)

        sample_docs = random.sample(
            self.tokenized_documents,
            sample_size
        )

        dictionary = Dictionary(sample_docs)

        coherence_model = CoherenceModel(
            topics=self.topic_words,
            texts=sample_docs,
            dictionary=dictionary,
            coherence=COHERENCE_METHOD,
            processes=-1          # استفاده از تمام هسته‌های CPU
        )

        score = coherence_model.get_coherence()

        self.metrics[f"Coherence (using {COHERENCE_METHOD} methode)"] = score


        return score

    def evaluate_diversity(self, top_n=None):

        words = []
        top_n=TOP_N_DIVERSITY

        for topic in self.topic_words:
            words.extend(topic[:top_n])

        diversity = len(set(words)) / len(words)

        self.metrics["Diversity"] = diversity

        return diversity


    def evaluate_outlier_ratio(self):

        ratio = (np.sum(self.topics == -1) / len(self.topics)) * 100
        self.metrics["Outlier Ratio (%)"] = ratio

        return ratio


    def evaluate_nmi(self):

        if self.df is None:
            raise ValueError("Ground truth dataframe is required.")

        labels = (self.df[CATEGORY_COLUMN].reset_index(drop=True))

        valid_mask = self.topics != -1
        labels = labels[valid_mask]
        topics = self.topics[valid_mask]


        score = normalized_mutual_info_score(labels, topics)
        self.metrics["NMI"] = score

        return score


    def evaluate_purity(self):

        if self.df is None:
            raise ValueError("Ground truth dataframe is required.")

        labels = (self.df[CATEGORY_COLUMN].reset_index(drop=True))

        evaluation_df = pd.DataFrame({
            "Topic": self.topics,
            "Category": labels
        })

        # Remove outliers
        evaluation_df = evaluation_df[evaluation_df["Topic"] != -1]

        if evaluation_df.empty:
            self.metrics["Purity"] = np.nan
            return np.nan

        purity_sum = 0

        for _, group in evaluation_df.groupby("Topic"):

            majority = Counter(group["Category"]).most_common(1)[0][1]
            purity_sum += majority

        purity = (purity_sum / evaluation_df.shape[0])
        self.metrics["Purity"] = purity

        return purity


    # --------------------------------------------------
    # Run All Metrics
    # --------------------------------------------------

    def evaluate_all(self):

        self.metrics = {}
        try:
            self.evaluate_coherence()
        except Exception as e:
            print(f"Coherence failed: {e}")
            self.metrics["Coherence"] = np.nan
        self.evaluate_diversity()
        self.evaluate_outlier_ratio()
        self.evaluate_nmi()
        self.evaluate_purity()
        self.metrics_df = (
            pd.DataFrame(
                self.metrics.items(),
                columns=[
                    "Metric",
                    "Value"]).round(4))

        return self.metrics_df


    # --------------------------------------------------
    # Save Results
    # --------------------------------------------------

    def save_results(self, path):

        if self.metrics_df is None:
            raise ValueError("Run evaluate_all() before saving results." )

        self.metrics_df.to_csv(path, index=False, encoding=CSV_ENCODING)

        print( f"Evaluation results saved successfully:\n{path}")