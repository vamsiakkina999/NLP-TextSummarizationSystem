"""Optimization"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from pipeline_steps.evaluation_utils import evaluate_summary
from pipeline_steps.preprocessing_utils import preprocess_article
from pipeline_steps.scoring_utils import (
    length_scores,
    normalize_scores,
    position_scores,
)

sentence_data: list[dict[str, str]] = globals()["sentence_data"]
train_clean: pd.DataFrame = globals()["train_clean"]
validation_clean: pd.DataFrame = globals()["validation_clean"]

def calculate_keyword_scores(tfidf_matrix):
    """
    Calculate keyword importance for each sentence
    based on the TF-IDF values of its words.
    """

    keyword_scores = np.asarray(tfidf_matrix.mean(axis=1)).ravel()

    return keyword_scores

def generate_feature_scores(sentence_data):

    processed_sentences = [
        item["processed"]
        for item in sentence_data
    ]

    # TF-IDF representation
    vectorizer = TfidfVectorizer()

    tfidf_matrix = csr_matrix(vectorizer.fit_transform(processed_sentences))

    # 1. TF-IDF score
    tfidf_scores = np.asarray(tfidf_matrix.sum(axis=1)).ravel()
    tfidf_score = normalize_scores(tfidf_scores)

    # 2. Position score
    position_score = position_scores(
        len(sentence_data)
    )

    # 3. Length score
    length_score = length_scores(
        sentence_data
    )

    # 4. Keyword importance
    keyword_scores = calculate_keyword_scores(
        tfidf_matrix
    )
    keyword_score = normalize_scores(
        keyword_scores
    )

    return (
        tfidf_score,
        position_score,
        length_score,
        keyword_score
    )

def calculate_final_scores(sentence_data):

    tfidf_score, position_score, length_score, keyword_score = \
        generate_feature_scores(sentence_data)

    final_scores = (
        0.40 * tfidf_score +
        0.25 * position_score +
        0.15 * length_score +
        0.20 * keyword_score
    )

    return final_scores

final_scores = calculate_final_scores(
    sentence_data
)

for i, score in enumerate(final_scores):
    print(
        f"Sentence {i+1}: {score:.4f}"
    )

ranking = pd.DataFrame({
    "sentence_no": range(
        1,
        len(sentence_data) + 1
    ),

    "sentence": [
        item["original"]
        for item in sentence_data
    ],

    "final_score": final_scores
})

ranking = ranking.sort_values(
    by="final_score",
    ascending=False
)

print(ranking)

top_n = 3

selected = ranking.head(top_n)

# Restore original article order
selected = selected.sort_values(
    by="sentence_no"
)

summary = " ".join(
    selected["sentence"].tolist()
)

print("GENERATED SUMMARY:\n")
print(summary)

def generate_optimized_summary(article, top_n=3):

    # Preprocessing
    sentence_data = preprocess_article(article)

    if len(sentence_data) == 0:
        return ""

    # Calculate feature scores
    final_scores = calculate_final_scores(
        sentence_data
    )

    # Ranking
    ranking = pd.DataFrame({
        "sentence_no": range(
            len(sentence_data)
        ),
        "sentence": [
            item["original"]
            for item in sentence_data
        ],
        "score": final_scores
    })

    # Select important sentences
    selected = ranking.nlargest(
        top_n,
        "score"
    )

    # Restore original order
    selected = selected.sort_values(
        "sentence_no"
    )

    # Generate summary
    summary = " ".join(
        selected["sentence"].tolist()
    )

    return summary

optimized_summary = generate_optimized_summary(
    train_clean.iloc[0]["article"],
    top_n=3
)

print(optimized_summary)

"""Weight Optimization"""

validation_sample = validation_clean.sample(
    100,
    random_state=42
).reset_index(drop=True)

print("Validation samples:", len(validation_sample))

optimization_data = []

for article_number, (_, row) in enumerate(validation_sample.iterrows(), start=1):

    article = row["article"]
    reference = row["highlights"]

    try:
        sentence_data = preprocess_article(article)

        if len(sentence_data) == 0:
            continue

        tfidf_score, position_score, length_score, keyword_score = \
            generate_feature_scores(sentence_data)

        optimization_data.append({
            "article_no": article_number,
            "sentence_data": sentence_data,
            "reference": reference,
            "tfidf": tfidf_score,
            "position": position_score,
            "length": length_score,
            "keyword": keyword_score
        })

    except Exception as e:
        print(f"Error in article {article_number}: {e}")

print("Articles prepared:", len(optimization_data))

def generate_weighted_summary(data, w_tfidf, w_position, w_length, w_keyword, top_n=3):

    sentence_data = data["sentence_data"]

    final_scores = (
        w_tfidf * data["tfidf"] +
        w_position * data["position"] +
        w_length * data["length"] +
        w_keyword * data["keyword"]
    )

    ranking = pd.DataFrame({
        "sentence_no": range(len(sentence_data)),
        "sentence": [item["original"] for item in sentence_data],
        "score": final_scores
    })

    selected = ranking.nlargest(top_n, "score")

    # Restore original article order
    selected = selected.sort_values("sentence_no")

    return " ".join(selected["sentence"].tolist())

weight_combinations = []

for tfidf in [0.2, 0.3, 0.4, 0.5, 0.6]:
    for position in [0.1, 0.2, 0.3]:
        for length in [0.1, 0.2]:

            keyword = 1 - (tfidf + position + length)

            if 0 <= keyword <= 0.4:
                weight_combinations.append(
                    (tfidf, position, length, keyword)
                )

print("Weight combinations:", len(weight_combinations))

optimization_results = []

for weights in weight_combinations:

    w_tfidf, w_position, w_length, w_keyword = weights

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for data in optimization_data:

        generated = generate_weighted_summary(
            data,
            w_tfidf,
            w_position,
            w_length,
            w_keyword,
            top_n=3
        )

        scores = evaluate_summary(
            generated,
            data["reference"]
        )

        rouge1_scores.append(scores["rouge1"])
        rouge2_scores.append(scores["rouge2"])
        rougeL_scores.append(scores["rougeL"])

    optimization_results.append({
        "tfidf": w_tfidf,
        "position": w_position,
        "length": w_length,
        "keyword": w_keyword,
        "rouge1": np.mean(rouge1_scores),
        "rouge2": np.mean(rouge2_scores),
        "rougeL": np.mean(rougeL_scores)
    })

optimization_df = pd.DataFrame(optimization_results)

print("Optimization completed!")

optimization_df = optimization_df.sort_values(
    by="rougeL",
    ascending=False
)
best_result = optimization_df.iloc[0]

print("Best Weights:")
print("TF-IDF   :", best_result["tfidf"])
print("Position :", best_result["position"])
print("Length   :", best_result["length"])
print("Keyword  :", best_result["keyword"])

print("\nBest ROUGE Scores:")
print("ROUGE-1 :", best_result["rouge1"])
print("ROUGE-2 :", best_result["rouge2"])
print("ROUGE-L :", best_result["rougeL"])

best_weights = {
    "tfidf": best_result["tfidf"],
    "position": best_result["position"],
    "length": best_result["length"],
    "keyword": best_result["keyword"]
}

print(best_weights)
