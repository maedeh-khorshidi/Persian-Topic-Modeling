import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.figure as mpl_figure
import plotly.graph_objects as go
from config import (
    FIGURE_DPI,
    FIGURE_FORMAT,
    FIGURE_SIZE,
    CSV_ENCODING,
    CATEGORY_COLUMN,
    SUBCATEGORY_COLUMN,
)


class BaseVisualizer:

    def __init__(self):

        # تمام شکل‌های ساخته شده
        self.figures = {}

    # --------------------------------------------------
    # Private Methods
    # --------------------------------------------------

    def _create_figure(self):

        fig, ax = plt.subplots(
            figsize=FIGURE_SIZE
        )

        return fig, ax


    def _apply_plot_style(self, ax):

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.3
        )


    def _save_figure(self, fig, path):

        os.makedirs(
            os.path.dirname(path),
            exist_ok=True
        )

        # Matplotlib Figure
        if isinstance(fig, mpl_figure.Figure):

            fig.tight_layout()

            fig.savefig(
                path,
                dpi=FIGURE_DPI,
                format=FIGURE_FORMAT,
                bbox_inches="tight"
            )

            plt.close(fig)

        # Plotly Figure
        elif isinstance(fig, go.Figure):

            fig.write_image(path)

        else:

            raise TypeError(
                f"Unsupported figure type: {type(fig)}"
            )


    def _save_figure_as_html(self, fig, path):
        os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

        # Matplotlib
        if isinstance(fig, mpl_figure.Figure):

            fig.tight_layout()

            fig.savefig(
                path,
                dpi=FIGURE_DPI,
                format=FIGURE_FORMAT,
                bbox_inches="tight"
            )

            plt.close(fig)

        # Plotly
        elif isinstance(fig, go.Figure):

            html_path = os.path.splitext(path)[0] + ".html"

            fig.write_html(html_path)

        else:

            raise TypeError(
                f"Unsupported figure type: {type(fig)}"
            )
    # --------------------------------------------------
    # Public Methods
    # --------------------------------------------------

    def save_figure(
        self,
        figure_name,
        path
    ):

        if figure_name not in self.figures:

            raise ValueError(
                f"Figure '{figure_name}' does not exist."
            )

        self._save_figure(
            self.figures[figure_name],
            path
        )


    def save_all_figures(
        self,
        output_directory,
        as_html = False
    ):

        os.makedirs(
            output_directory,
            exist_ok=True
        )

        if len(self.figures) == 0:

            raise ValueError(
                "No figures have been generated."
            )

        for figure_name, fig in self.figures.items():

            file_path = os.path.join(
                output_directory,
                f"{figure_name}.{FIGURE_FORMAT}"
            )

            if as_html:
                self._save_figure_as_html(
                    fig,
                    file_path
                )
            else:
                self._save_figure(
                    fig,
                    file_path
                )

        print(
            f"\nAll figures were saved to:\n{output_directory}"
        )


class DatasetVisualizer(BaseVisualizer):

    def __init__(self, statistics):

        super().__init__()
        self.statistics = statistics

    # --------------------------------------------------

    def plot_category_distribution(self):

        table = self.statistics.category_statistics()

        fig, ax = self._create_figure()

        ax.bar(
            table[CATEGORY_COLUMN],
            table["Number_of_Documents"]
        )

        self._apply_plot_style(ax)

        ax.set_xlabel("Category")
        ax.set_ylabel("Number of Documents")
        ax.set_title("Category Distribution")

        plt.xticks(rotation=30)

        self.figures["category_distribution"] = fig

        return fig

    # --------------------------------------------------

    def plot_subcategory_distribution(self):

        table = self.statistics.subcategory_statistics()

        pivot = table.pivot(
            index=SUBCATEGORY_COLUMN,
            columns=CATEGORY_COLUMN,
            values="Number_of_Documents"
        ).fillna(0)

        fig, ax = self._create_figure()

        pivot.plot(
            kind="bar",
            stacked=True,
            ax=ax
        )

        self._apply_plot_style(ax)

        ax.set_xlabel("Subcategory")
        ax.set_ylabel("Number of Documents")
        ax.set_title("Subcategory Distribution")

        ax.legend(
            title="Category",
            bbox_to_anchor=(1.02, 1),
            loc="upper left"
        )

        self.figures["subcategory_distribution"] = fig

        return fig

    # --------------------------------------------------

    def plot_document_length_distribution(self):

        table = self.statistics.document_lengths()

        fig, ax = self._create_figure()

        ax.hist(
            table["Document Length"],
            bins=30,
            edgecolor="black"
        )

        self._apply_plot_style(ax)

        ax.set_xlabel("Document Length")
        ax.set_ylabel("Number of Documents")
        ax.set_title("Document Length Distribution")

        self.figures["document_length_distribution"] = fig

        return fig
    

    def generate_all_figures(self):

        self.plot_category_distribution()
        self.plot_subcategory_distribution()
        self.plot_document_length_distribution()

        return self.figures


class TopicVisualizer(BaseVisualizer):
    def __init__(
        self,
        topic_model,
        statistics,
        evaluation=None
    ):
        super().__init__()

        self.topic_model = topic_model
        self.statistics = statistics
        self.evaluation = evaluation

    # --------------------------------------------------
    # BERTopic Visualizations
    # --------------------------------------------------

    def visualize_topics(self):

        fig = self.topic_model.visualize_topics()

        self.figures["topic_map"] = fig

        return fig


    def visualize_barchart(
        self,
        top_n_topics=20,
        n_words=10
    ):

        fig = self.topic_model.visualize_barchart(
            top_n_topics=top_n_topics,
            n_words=n_words
        )

        self.figures["topic_barchart"] = fig

        return fig


    def visualize_hierarchy(self):

        fig = self.topic_model.visualize_hierarchy()

        self.figures["topic_hierarchy"] = fig

        return fig


    def visualize_heatmap(self):

        fig = self.topic_model.visualize_heatmap()

        self.figures["topic_heatmap"] = fig

        return fig


    def visualize_documents(
        self,
        documents,
        embeddings=None
    ):

        fig = self.topic_model.visualize_documents(
            docs=documents,
            embeddings=embeddings
        )

        self.figures["document_map"] = fig

        return fig

    # --------------------------------------------------
    # Model Comparison
    # --------------------------------------------------

    def plot_metric_comparison(
        self,
        comparison_df
    ):

        fig, ax = self._create_figure()

        comparison_df.plot(
            x="Metric",
            kind="bar",
            ax=ax
        )

        self._apply_plot_style(ax)

        ax.set_xlabel("Metric")
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison")

        ax.legend(
            title="Model",
            bbox_to_anchor=(1.02, 1),
            loc="upper left"
        )

        self.figures["metric_comparison"] = fig

        return fig


    def plot_topic_count_comparison(
        self,
        comparison_df
    ):

        fig, ax = self._create_figure()

        comparison_df.plot(
            x="Model",
            y="Topic Count",
            kind="bar",
            legend=False,
            ax=ax
        )

        self._apply_plot_style(ax)

        ax.set_xlabel("Model")
        ax.set_ylabel("Number of Topics")
        ax.set_title("Topic Count Comparison")

        self.figures["topic_count_comparison"] = fig

        return fig


    def plot_outlier_comparison(
        self,
        comparison_df
    ):

        fig, ax = self._create_figure()

        comparison_df.plot(
            x="Model",
            y="Outlier Percentage",
            kind="bar",
            legend=False,
            ax=ax
        )

        self._apply_plot_style(ax)

        ax.set_xlabel("Model")
        ax.set_ylabel("Outlier Percentage (%)")
        ax.set_title("Outlier Comparison")

        self.figures["outlier_comparison"] = fig

        return fig


    def plot_radar_comparison(
        self,
        comparison_df
    ):

        data = comparison_df.copy()

        models = data.iloc[:, 0].tolist()
        metrics = data.columns[1:].tolist()

        values = data.iloc[:, 1:].astype(float)

        values = (
            values - values.min()
        ) / (
            values.max() - values.min() + 1e-9
        )

        angles = np.linspace(
            0,
            2 * np.pi,
            len(metrics),
            endpoint=False
        ).tolist()

        angles += angles[:1]

        fig = plt.figure(figsize=FIGURE_SIZE)

        ax = plt.subplot(
            111,
            polar=True
        )

        for i, model in enumerate(models):

            row = values.iloc[i].tolist()
            row += row[:1]

            ax.plot(
                angles,
                row,
                linewidth=2,
                label=model
            )

            ax.fill(
                angles,
                row,
                alpha=0.2
            )

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)

        ax.set_ylim(0, 1)

        ax.set_title(
            "Model Comparison (Normalized Metrics)"
        )

        ax.legend(
            bbox_to_anchor=(1.25, 1.10),
            loc="upper right"
        )

        self.figures["radar_comparison"] = fig

        return fig
    
    
    def generate_all_figs(self, document=None, embeding=None, compration_mode=False,
                        metric_df=None, topic_count_df=None,
                        outlier_df=None, radar_df=None):

        if not compration_mode:
            self.visualize_topics()
            self.visualize_barchart()
            self.visualize_hierarchy()
            self.visualize_heatmap()
            self.visualize_documents(document, embeding)

        else:
            self.plot_metric_comparison(metric_df)
            self.plot_topic_count_comparison(topic_count_df)
            self.plot_outlier_comparison(outlier_df)
            self.plot_radar_comparison(radar_df)

        return self.figures
