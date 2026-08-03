import numpy as np
import pandas as pd
import os
from config import (
    CATEGORY_COLUMN,
    SUBCATEGORY_COLUMN,
    TEXT_COLUMN,
    SUMMARY_COLUMN,
    CSV_ENCODING,
    N_TOPIC_WORDS
)


class BaseStatistics:

    def __init__(self):

        # تمام جدول‌هایی که کلاس‌های فرزند تولید می‌کنند
        self.tables = {}

    # --------------------------------------------------
    # Private Methods
    # --------------------------------------------------

    def _save_dataframe(self, dataframe, path):
        """
        Save a DataFrame to CSV.
        """

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True
        )

        dataframe.to_csv(
            path,
            index=False,
            encoding=CSV_ENCODING
        )

        print(f"Saved successfully:\n{path}")

    # --------------------------------------------------
    # Public Methods
    # --------------------------------------------------

    def save_table(
        self,
        table_name,
        path
    ):
        """
        Save one generated table.
        """

        if table_name not in self.tables:
            raise ValueError(
                f"Table '{table_name}' does not exist."
            )

        self._save_dataframe(
            self.tables[table_name],
            path
        )

    def save_all_tables(
        self,
        output_directory
    ):
        """
        Save all generated tables.
        """

        os.makedirs(
            output_directory,
            exist_ok=True
        )

        if len(self.tables) == 0:
            raise ValueError(
                "No tables have been generated."
            )

        for table_name, dataframe in self.tables.items():

            file_path = os.path.join(
                output_directory,
                f"{table_name}.csv"
            )

            self._save_dataframe(
                dataframe,
                file_path
            )

        print(
            f"\nAll tables were saved to:\n{output_directory}"
        )


class DatasetStatistics(BaseStatistics):

    def __init__(self, dataframe, documents):

        super().__init__()

        self.df = dataframe
        self.documents = documents

        self.vocabulary_size = len(
            {
                word
                for doc in self.documents
                for word in str(doc).split()
            }
        )

    # ------------------------
    # Dataset Statistics
    # ------------------------

    def dataset_statistics(self):

        statistics = {

            "Number of Documents":
                len(self.documents),

            "Number of Categories":
                self.df[CATEGORY_COLUMN].nunique(),

            "Number of Subcategories":
                self.df[SUBCATEGORY_COLUMN].nunique(),

            "Average Document Length":
                round(
                    np.mean(
                        [
                            len(str(doc).split())
                            for doc in self.documents
                        ]
                    ),
                    2
                ),

            "Vocabulary Size":
                self.vocabulary_size
        }

        table = pd.DataFrame(
            statistics.items(),
            columns=[
                "Statistic",
                "Value"
            ]
        )

        self.tables["dataset_statistics"] = table

        return table

    # ------------------------

    def category_statistics(self):

        table = (
            self.df
            .groupby(CATEGORY_COLUMN)
            .agg(
                Number_of_Subcategories=(
                    SUBCATEGORY_COLUMN,
                    "nunique"
                ),
                Number_of_Documents=(
                    CATEGORY_COLUMN,
                    "size"
                )
            )
            .reset_index()
        )

        table["Percentage"] = (
            table["Number_of_Documents"]
            / table["Number_of_Documents"].sum()
            * 100
        ).round(2)

        lengths = pd.Series(
            [
                len(str(doc).split())
                for doc in self.documents
            ],
            index=self.df.index
        )

        table["Average_Document_Length"] = (
            self.df
            .assign(Document_Length=lengths)
            .groupby(CATEGORY_COLUMN)["Document_Length"]
            .mean()
            .round(2)
            .values
        )

        self.tables["category_statistics"] = table

        return table

    # ------------------------

    def subcategory_statistics(self):

        table = (
            self.df
            .groupby(
                [
                    CATEGORY_COLUMN,
                    SUBCATEGORY_COLUMN
                ]
            )
            .size()
            .reset_index(
                name="Number_of_Documents"
            )
        )

        table["Percentage"] = (
            table
            .groupby(CATEGORY_COLUMN)["Number_of_Documents"]
            .transform(
                lambda x:
                (
                    x / x.sum() * 100
                ).round(2)
            )
        )

        self.tables["subcategory_statistics"] = table

        return table

    # ------------------------

    def document_lengths(self):

        table = pd.DataFrame({

            "Document Length": [
                len(str(doc).split())
                for doc in self.documents
            ]

        })

        self.tables["document_lengths"] = table

        return table
    
    def generate_all_tables(self):

        self.dataset_statistics()
        self.category_statistics()
        self.subcategory_statistics()
        self.document_lengths()

        return self.tables

class TopicStatistics(BaseStatistics):

    def __init__(
        self,
        topic_model,
        documents,
        topics,
        dataframe,
        probabilities=None
    ):

        super().__init__()

        self.topic_model = topic_model
        self.documents = documents
        self.topics = np.array(topics)
        self.df = dataframe
        self.probabilities = probabilities

        self.topic_info = self.topic_model.get_topic_info()

    # --------------------------------------------------
    # Private
    # --------------------------------------------------

    def _get_topic_words(self):

        topic_dict = self.topic_model.get_topics()

        topic_words = {}

        for topic_id, words in topic_dict.items():

            if topic_id == -1:
                continue

            topic_words[topic_id] = [
                word for word, _ in words
            ]

        return topic_words

    # --------------------------------------------------
    # Topic Statistics
    # --------------------------------------------------

    def topic_statistics(self):

        valid_topics = self.topic_info[
            self.topic_info["Topic"] != -1
        ]

        statistics = {

            "Number of Topics":
                len(valid_topics),

            "Average Documents per Topic":
                round(
                    valid_topics["Count"].mean(),
                    2
                ),

            "Largest Topic":
                valid_topics["Count"].max(),

            "Smallest Topic":
                valid_topics["Count"].min(),

            "Outlier Documents":
                np.sum(self.topics == -1),

            "Outlier Percentage (%)":
                round(
                    np.sum(self.topics == -1)
                    / len(self.documents)
                    * 100,
                    2
                )
        }

        table = pd.DataFrame(
            statistics.items(),
            columns=[
                "Statistic",
                "Value"
            ]
        )

        self.tables["topic_statistics"] = table

        return table

    # --------------------------------------------------

    def topic_information(self):

        table = self.topic_info.copy()

        self.tables["topic_information"] = table

        return table

    # --------------------------------------------------

    def topic_keywords(self, top_n=N_TOPIC_WORDS):

        topic_words = self._get_topic_words()

        rows = []

        for topic_id, words in topic_words.items():

            rows.append({

                "Topic":
                    topic_id,

                "Keywords":
                    ", ".join(words[:top_n])

            })

        table = pd.DataFrame(rows)

        self.tables["topic_keywords"] = table

        return table

    # --------------------------------------------------

    def topic_sizes(self):

        table = (

            self.topic_info[
                self.topic_info["Topic"] != -1
            ][["Topic", "Count"]]

            .sort_values(
                "Count",
                ascending=False
            )

            .reset_index(drop=True)

        )

        self.tables["topic_sizes"] = table

        return table

    # --------------------------------------------------

    def topic_percentages(self):

        table = self.topic_sizes().copy()

        table["Percentage"] = (

            table["Count"]
            / len(self.documents)
            * 100

        ).round(2)

        self.tables["topic_percentages"] = table

        return table

    # --------------------------------------------------

    def representative_documents(self, top_n=3):

        info = self.topic_model.get_document_info(
            self.documents
        )

        info = (

            info[
                info["Topic"] != -1
            ]

            .sort_values(
                [
                    "Topic",
                    "Probability"
                ],
                ascending=[
                    True,
                    False
                ]
            )

            .groupby("Topic")

            .head(top_n)

            .reset_index(drop=True)

        )

        text_column = (

            "Document"

            if "Document" in info.columns

            else TEXT_COLUMN

        )

        table = info[
            [
                "Topic",
                "Probability",
                text_column
            ]
        ].copy()

        self.tables["representative_documents"] = table

        return table

    # --------------------------------------------------

    def average_document_length_per_topic(self):

        df = pd.DataFrame({

            "Topic":
                self.topics,

            "Length":
                [
                    len(str(doc).split())
                    for doc in self.documents
                ]

        })

        table = (

            df[
                df["Topic"] != -1
            ]

            .groupby("Topic")

            .agg(

                Average_Length=(
                    "Length",
                    "mean"
                ),

                Number_of_Documents=(
                    "Length",
                    "count"
                )

            )

            .reset_index()

        )

        table["Average_Length"] = (

            table["Average_Length"]

            .round(2)

        )

        self.tables[
            "average_document_length_per_topic"
        ] = table

        return table

    # --------------------------------------------------

    def topic_category_distribution(self):

        table = pd.DataFrame({

            "Topic":
                self.topics,

            CATEGORY_COLUMN:
                self.df[CATEGORY_COLUMN]

        })

        table = (

            table[
                table["Topic"] != -1
            ]

            .groupby(
                [
                    "Topic",
                    CATEGORY_COLUMN
                ]
            )

            .size()

            .reset_index(
                name="Count"
            )

        )

        table["Percentage"] = (

            table

            .groupby("Topic")["Count"]

            .transform(

                lambda x:
                (
                    x / x.sum()
                    * 100
                ).round(2)

            )

        )

        self.tables[
            "topic_category_distribution"
        ] = table

        return table

    # --------------------------------------------------

    def topic_subcategory_distribution(self):

        table = pd.DataFrame({

            "Topic":
                self.topics,

            SUBCATEGORY_COLUMN:
                self.df[SUBCATEGORY_COLUMN]

        })

        table = (

            table[
                table["Topic"] != -1
            ]

            .groupby(
                [
                    "Topic",
                    SUBCATEGORY_COLUMN
                ]
            )

            .size()

            .reset_index(
                name="Count"
            )

        )

        table["Percentage"] = (

            table

            .groupby("Topic")["Count"]

            .transform(

                lambda x:
                (
                    x / x.sum()
                    * 100
                ).round(2)

            )

        )

        self.tables[
            "topic_subcategory_distribution"
        ] = table

        return table
    

    def generate_all_tables(self):

        self.topic_statistics()
        self.topic_information()
        self.topic_keywords()
        self.topic_sizes()
        self.topic_percentages()
        self.representative_documents()
        self.average_document_length_per_topic()
        self.topic_category_distribution()
        self.topic_subcategory_distribution()

        return self.tables


