import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from pipeline_steps.evaluation_utils import evaluate_summary, scorer
from pipeline_steps.preprocessing_utils import preprocess_article
from pipeline_steps.scoring_utils import (
    length_scores,
    normalize_scores,
    position_scores,
)

train_clean: pd.DataFrame = globals()["train_clean"]
validation_clean: pd.DataFrame = globals()["validation_clean"]

article = train_clean["article"].iloc[0]
sentence_data = preprocess_article(article)

processed_sentences = [
    item["processed"] for item in sentence_data
]
print("Number of sentences:", len(processed_sentences))

tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = csr_matrix(tfidf_vectorizer.fit_transform(processed_sentences))

print("TF-IDF Matrix Shape:", tfidf_matrix.shape)

sentence_scores = tfidf_matrix.toarray().sum(axis=1)

for i, score in enumerate(sentence_scores):
    print(f"Sentence {i+1}: {score:.4f}")

"""Improve sentence importance scoring"""

tfidf_normalized = normalize_scores(sentence_scores)

print(tfidf_normalized
)

position_score = position_scores(len(sentence_data))

for i, score in enumerate(position_score):
    print(f"Sentence {i+1}: {score:.3f}")

length_score = length_scores(sentence_data)

for i, score in enumerate(length_score):
    print(f"Sentence {i+1}: {score:.3f}")

final_scores = (
    0.50 * tfidf_normalized +
    0.30 * position_score +
    0.20 * length_score
)

for i, score in enumerate(final_scores):
    print(f"Sentence {i+1}: {score:.4f}")

ranking = pd.DataFrame({
    "sentence_no": range(1, len(sentence_data) + 1),
    "sentence": [item["original"] for item in sentence_data],
    "tfidf_score": tfidf_normalized,
    "position_score": position_score,
    "length_score": length_score,
    "final_score": final_scores
})

ranking = ranking.sort_values(
    by="final_score",
    ascending=False
)

print(ranking)

top_n = 3

selected = ranking.head(top_n)

selected = selected.sort_values(
    by="sentence_no"
)

summary = " ".join(
    selected["sentence"].tolist()
)

print("SUMMARY:\n")
print(summary)

"""Evaluation (install project dependencies with pip install -r requirements.txt)."""

reference_summary = train_clean.iloc[0]["highlights"]

print("REFERENCE SUMMARY:\n")
print(reference_summary)

print("\nGENERATED SUMMARY:\n")
print(summary)

scores = scorer.score(
    reference_summary,
    summary
)

for metric, score in scores.items():
    print(metric)
    print("Precision:", round(score.precision, 4))
    print("Recall   :", round(score.recall, 4))
    print("F1       :", round(score.fmeasure, 4))
    print()

"""On 100 articles"""

validation_sample = validation_clean.sample(
    100,
    random_state=42
).reset_index(drop=True)

print("Articles selected:", len(validation_sample))

def generate_summary(article, top_n=3):

    sentence_data = preprocess_article(article)

    if len(sentence_data) == 0:
        return ""

    processed_sentences = [
        item["processed"] for item in sentence_data
    ]

    # TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = csr_matrix(vectorizer.fit_transform(processed_sentences))

    tfidf_scores = tfidf_matrix.toarray().sum(axis=1)

    # Normalize TF-IDF
    tfidf_normalized = normalize_scores(tfidf_scores)

    # Position
    position_score = position_scores(len(sentence_data))

    # Length
    length_score = length_scores(sentence_data)

    # Final score
    final_scores = (
        0.50 * tfidf_normalized +
        0.30 * position_score +
        0.20 * length_score
    )

    # Ranking
    ranking = pd.DataFrame({
        "sentence_no": range(len(sentence_data)),
        "sentence": [
            item["original"] for item in sentence_data
        ],
        "score": final_scores
    })

    # Select top sentences
    selected = ranking.nlargest(top_n, "score")

    # Restore original order
    selected = selected.sort_values("sentence_no")

    summary = " ".join(
        selected["sentence"].tolist()
    )

    return summary

test_article = validation_sample.iloc[0]["article"]

generated_summary = generate_summary(
    test_article,
    top_n=3
)

print("GENERATED SUMMARY:\n")
print(generated_summary)

results = []

for article_number, (_, row) in enumerate(validation_sample.iterrows(), start=1):

    article = row["article"]
    reference = row["highlights"]

    try:
        generated = generate_summary(
            article,
            top_n=3
        )

        scores = evaluate_summary(
            generated,
            reference
        )

        results.append({
            "article_no": article_number,
            "rouge1": scores["rouge1"],
            "rouge2": scores["rouge2"],
            "rougeL": scores["rougeL"]
        })

    except Exception as e:
        print(f"Error in article {article_number}: {e}")

results_df = pd.DataFrame(results)

print(results_df.head())

average_scores = results_df[
    ["rouge1", "rouge2", "rougeL"]
].mean()

print("Average ROUGE Scores")
print("--------------------")
print(f"ROUGE-1: {average_scores['rouge1']:.4f}")
print(f"ROUGE-2: {average_scores['rouge2']:.4f}")
print(f"ROUGE-L: {average_scores['rougeL']:.4f}")

print("ROUGE Standard Deviation")
print("------------------------")

print(f"ROUGE-1: {results_df['rouge1'].std():.4f}")
print(f"ROUGE-2: {results_df['rouge2'].std():.4f}")
print(f"ROUGE-L: {results_df['rougeL'].std():.4f}")
