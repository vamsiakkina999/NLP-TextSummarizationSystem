"""Sentence-scoring helpers shared by analysis stages."""

import numpy as np


def normalize_scores(scores):
    min_score = np.min(scores)
    max_score = np.max(scores)
    if max_score == min_score:
        return np.ones(len(scores))
    return (scores - min_score) / (max_score - min_score)


def position_scores(num_sentences):
    return np.array(
        [1 - (index / num_sentences) for index in range(num_sentences)]
    )


def length_scores(sentence_data):
    lengths = np.array(
        [len(item["processed"].split()) for item in sentence_data]
    )
    min_length = np.min(lengths)
    max_length = np.max(lengths)
    if max_length == min_length:
        return np.ones(len(lengths))
    return (lengths - min_length) / (max_length - min_length)
