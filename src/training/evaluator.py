"""
A lightweight SentenceEvaluator: after each epoch, checks whether the
embedding space is getting better separated by category, using k-NN
classification accuracy on a held-out validation split.

Does NOT touch test.csv. This is purely for picking the best checkpoint
during fine-tuning and getting an early signal before running the full
03_comparison pipeline.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent
sys.path.append(str(PROJECT_ROOT / "src"))

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sentence_transformers.evaluation import SentenceEvaluator


class KNNCategoryEvaluator(SentenceEvaluator):

    def __init__(self, val_texts, val_labels, name="val", k=5, batch_size=64):
        self.val_texts = val_texts
        self.val_labels = np.array(val_labels)
        self.name = name
        self.k = k
        self.batch_size = batch_size

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):

        embeddings = model.encode(
            self.val_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        clf = KNeighborsClassifier(n_neighbors=self.k, metric="cosine")
        scores = cross_val_score(clf, embeddings, self.val_labels, cv=5)
        accuracy = scores.mean()

        print(f"[epoch {epoch}] KNN(k={self.k}) val accuracy: {accuracy:.4f}")

        return accuracy