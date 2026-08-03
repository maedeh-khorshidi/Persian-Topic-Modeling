from torch.utils.data import Dataset
from sentence_transformers import InputExample

import random

from pair_generator import PairGenerator


class ContrastiveDataset(Dataset):

    def __init__(
        self,
        dataframe,
    ):

        self.generator = PairGenerator(
            dataframe=dataframe,
        )


        self.examples = []

        self.build_dataset()


    # ---------------------------------
    # Generate training pairs
    # ---------------------------------

    def build_dataset(self):

        for index in self.generator.df.index:


            # -------------------------
            # Positive Pair
            # -------------------------

            anchor, positive = (
                self.generator.generate_positive_pair(
                    index
                )
            )


            self.examples.append(

                InputExample(

                    texts=[
                        anchor,
                        positive
                    ],

                    label=1.0
                )
            )


            # -------------------------
            # Negative Pair
            # -------------------------

            anchor, negative = (
                self.generator.generate_negative_pair(
                    index
                )
            )


            self.examples.append(

                InputExample(

                    texts=[
                        anchor,
                        negative
                    ],

                    label=0.0
                )
            )


        random.shuffle(
            self.examples
        )


    def __len__(self):

        return len(self.examples)


    def __getitem__(
        self,
        index
    ):

        return self.examples[index]