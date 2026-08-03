import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT / "src")
)


import torch

from torch.utils.data import DataLoader
from sentence_transformers import losses

from dataset import ContrastiveDataset
from finetune_encoder import build_finetune_encoder

from config import(
    FINETUNE_MODEL_NAME,
    FINETUNED_MODEL_DIR
)



class ContrastiveTrainer:


    def __init__(
        self,
        dataframe,
        model_name=FINETUNE_MODEL_NAME,
        batch_size=8,
        epochs=1,
        output_path=FINETUNED_MODEL_DIR
    ):


        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )


        print(
            f"Using device: {self.device}"
        )


        self.model = build_finetune_encoder(
            model_name
        )


        self.dataset = ContrastiveDataset(
            dataframe,
        )

        print(
            f"Number of training examples: {len(self.dataset)}"
        )


        self.dataloader = DataLoader(
            self.dataset,
            shuffle=True,
            batch_size=batch_size
        )

        print(
            f"Number of batches: {len(self.dataloader)}"
        )


        self.loss = losses.ContrastiveLoss(
            model=self.model
        )


        self.epochs = epochs

        self.output_path = output_path



    def train(self):


        warmup_steps = int(
            len(self.dataloader) *
            self.epochs *
            0.1
        )


        self.model.fit(

            train_objectives=[
                (
                    self.dataloader,
                    self.loss
                )
            ],

            epochs=self.epochs,

            warmup_steps=warmup_steps,

            output_path=self.output_path,

            optimizer_params={
                "lr": 2e-5
            },

            show_progress_bar=True
        )


        print(
            "Training finished!"
        )