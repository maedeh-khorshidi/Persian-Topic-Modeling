import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT / "src")
)


from sentence_transformers import SentenceTransformer, models

from config import FINETUNE_MODEL_NAME



def build_finetune_encoder(
    model_name=FINETUNE_MODEL_NAME
):

    transformer = models.Transformer(
        model_name,
        max_seq_length=512
    )


    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True
    )


    model = SentenceTransformer(
        modules=[
            transformer,
            pooling
        ]
    )


    return model