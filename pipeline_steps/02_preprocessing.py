"""Inspect preprocessing output for one training example."""

import pandas as pd

from pipeline_steps.preprocessing_utils import (
    preprocess_article,
    stop_words,
)

train_clean: pd.DataFrame = globals()["train_clean"]

print("Number of English stopwords:", len(stop_words))

sentence_data = preprocess_article(
    train_clean["article"].iloc[0]
)

print("Valid sentences in sample article:", len(sentence_data))
