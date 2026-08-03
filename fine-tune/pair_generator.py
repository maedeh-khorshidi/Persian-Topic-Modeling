import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.append(
    str(PROJECT_ROOT / "src")
)

import random
from collections import defaultdict

import pandas as pd

from config import(
    SUBCATEGORY_COLUMN,
    CATEGORY_COLUMN,
    RANDOM_STATE,
    NEGATIVE_STRATEGY
)

class PairGenerator:
    """
    Generate positive and negative sentence pairs
    for Contrastive Learning.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
    ):

        self.df = dataframe.reset_index(drop=True)

        self.category_column = CATEGORY_COLUMN
        self.subcategory_column = SUBCATEGORY_COLUMN

        # easy | hard | mixed
        self.negative_strategy = NEGATIVE_STRATEGY

        print(NEGATIVE_STRATEGY)

        random.seed(RANDOM_STATE)

        # ---------------------------------
        # Category -> document indices
        # ---------------------------------

        self.category_groups = defaultdict(list)

        # ---------------------------------
        # (Category, Subcategory)
        # -> document indices
        # ---------------------------------

        self.subcategory_groups = defaultdict(list)

        # ---------------------------------
        # Category -> Subcategories
        # ---------------------------------

        self.category_to_subcategories = defaultdict(set)

        for index, row in self.df.iterrows():

            category = row[self.category_column]

            subcategory = row[self.subcategory_column]

            self.category_groups[
                category
            ].append(index)

            self.subcategory_groups[
                (category, subcategory)
            ].append(index)

            self.category_to_subcategories[
                category
            ].add(subcategory)

        self.categories = list(
            self.category_groups.keys()
        )

    # ---------------------------------
    # Build input text
    # ---------------------------------

    def build_text(self, row):

        title = ""

        summary = ""

        if pd.notna(row["title"]):
            title = str(
                row["title"]
            ).strip()

        if pd.notna(row["summary"]):
            summary = str(
                row["summary"]
            ).strip()

        return (
            f"{title}\n{summary}"
        ).strip()

    # ---------------------------------
    # Positive Pair
    # Same Category + Same Subcategory
    # ---------------------------------

    def generate_positive_pair(
        self,
        index: int
    ):

        anchor = self.df.iloc[index]

        key = (

            anchor[self.category_column],

            anchor[self.subcategory_column]

        )

        candidate_indices = [

            idx

            for idx in self.subcategory_groups[key]

            if idx != index

        ]

        # اگر فقط یک نمونه وجود داشت
        # از همان Category انتخاب کن

        if not candidate_indices:

            candidate_indices = [

                idx

                for idx in self.category_groups[
                    anchor[self.category_column]
                ]

                if idx != index

            ]

        if not candidate_indices:

            raise ValueError(
                f"No positive sample found for "
                f"{anchor[self.category_column]} / "
                f"{anchor[self.subcategory_column]}"
            )
            

        positive_index = random.choice(
            candidate_indices
        )

        return (

            self.build_text(
                self.df.iloc[index]
            ),

            self.build_text(
                self.df.iloc[positive_index]
            )

        )

    # ---------------------------------
    # Negative Pair
    # ---------------------------------

    def generate_negative_pair(
        self,
        index: int
    ):

        anchor = self.df.iloc[index]

        anchor_category = anchor[
            self.category_column
        ]

        anchor_subcategory = anchor[
            self.subcategory_column
        ]

        # ---------------------------------
        # Decide negative type
        # ---------------------------------

        use_hard_negative = False

        if self.negative_strategy == "hard":

            use_hard_negative = True

        elif self.negative_strategy == "mixed":

            # 70% Hard
            # 30% Easy

            use_hard_negative = (
                random.random() < 0.7
            )

        # ---------------------------------
        # Hard Negative
        #
        # Same Category
        # Different Subcategory
        #
        # ---------------------------------

        if use_hard_negative:

            candidate_subcategories = [

                subcategory

                for subcategory in
                self.category_to_subcategories[
                    anchor_category
                ]

                if subcategory != anchor_subcategory

            ]

            if candidate_subcategories:

                negative_subcategory = random.choice(
                    candidate_subcategories
                )

                negative_index = random.choice(

                    self.subcategory_groups[

                        (
                            anchor_category,
                            negative_subcategory
                        )

                    ]

                )

                return (

                    self.build_text(
                        self.df.iloc[index]
                    ),

                    self.build_text(
                        self.df.iloc[negative_index]
                    )

                )

        # ---------------------------------
        # Easy Negative
        #
        # Different Category
        #
        # ---------------------------------

        negative_categories = [

            category

            for category in self.categories

            if category != anchor_category

        ]

        negative_category = random.choice(
            negative_categories
        )

        negative_index = random.choice(

            self.category_groups[
                negative_category
            ]

        )

        return (

            self.build_text(
                self.df.iloc[index]
            ),

            self.build_text(
                self.df.iloc[negative_index]
            )

        )