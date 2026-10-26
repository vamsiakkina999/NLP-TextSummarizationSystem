"""TEXT-RANK"""

import pandas as pd
import networkx as nx

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pipeline_steps.evaluation_utils import evaluate_summary
from pipeline_steps.preprocessing_utils import preprocess_article

validation_clean: pd.DataFrame = globals()["validation_clean"]
validation_sample: pd.DataFrame = globals()["validation_sample"]
test_clean: pd.DataFrame = globals()["test_clean"]

def textrank_summary(article, top_n=3):

    # Step 1: Preprocess article
    sentence_data = preprocess_article(article)

    if len(sentence_data) == 0:
        return ""

    # Step 2: Get processed sentences
    processed_sentences = [
        item["processed"] for item in sentence_data
    ]

    # Step 3: Convert sentences into TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_sentences)

    # Step 4: Calculate sentence similarity
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Step 5: Build graph
    graph = nx.from_numpy_array(similarity_matrix)

    # Step 6: Apply PageRank
    scores = nx.pagerank(graph)

    # Step 7: Create ranking table
    ranking = pd.DataFrame({
        "sentence_no": range(len(sentence_data)),
        "sentence": [
            item["original"] for item in sentence_data
        ],
        "textrank_score": [
            scores[i] for i in range(len(sentence_data))
        ]
    })

    # Step 8: Select top sentences
    selected = ranking.nlargest(top_n, "textrank_score")

    # Step 9: Restore original article order
    selected = selected.sort_values("sentence_no")

    # Step 10: Generate summary
    summary = " ".join(selected["sentence"].tolist())

    return summary

article = validation_clean.iloc[0]["article"]

summary = textrank_summary(
    article,
    top_n=3
)

print("TEXT RANK SUMMARY:\n")
print(summary)

generated_summary = summary

reference_summary = validation_clean.iloc[0]["highlights"]

scores = evaluate_summary(
    generated_summary,
    reference_summary
)

print("ROUGE-1:", scores["rouge1"])
print("ROUGE-2:", scores["rouge2"])
print("ROUGE-L:", scores["rougeL"])

textrank_results = []

for article_number, (_, row) in enumerate(validation_sample.iterrows(), start=1):

    article = row["article"]
    reference = row["highlights"]

    try:
        generated = textrank_summary(
            article,
            top_n=3
        )

        scores = evaluate_summary(
            generated,
            reference
        )

        textrank_results.append({
            "article_no": article_number,
            "rouge1": scores["rouge1"],
            "rouge2": scores["rouge2"],
            "rougeL": scores["rougeL"]
        })

    except Exception as e:
        print(f"Error in article {article_number}: {e}")

textrank_results_df = pd.DataFrame(textrank_results)
textrank_average = textrank_results_df[
    ["rouge1", "rouge2", "rougeL"]
].mean()

print("TextRank Average Scores")
print("----------------------")
print(f"ROUGE-1 : {textrank_average['rouge1']:.4f}")
print(f"ROUGE-2 : {textrank_average['rouge2']:.4f}")
print(f"ROUGE-L : {textrank_average['rougeL']:.4f}")

final_test_results = []

for article_number, (_, row) in enumerate(test_clean.iterrows(), start=1):

    article = row["article"]
    reference = row["highlights"]

    try:
        generated = textrank_summary(
            article,
            top_n=3
        )

        scores = evaluate_summary(
            generated,
            reference
        )

        final_test_results.append({
            "article_no": article_number,
            "rouge1": scores["rouge1"],
            "rouge2": scores["rouge2"],
            "rougeL": scores["rougeL"]
        })

    except Exception as e:
        print(f"Error in article {article_number}: {e}")

final_test_df = pd.DataFrame(final_test_results)

final_scores = final_test_df[
    ["rouge1", "rouge2", "rougeL"]
].mean()

print("FINAL TEST RESULTS")
print("------------------")
print(f"ROUGE-1 : {final_scores['rouge1']:.4f}")
print(f"ROUGE-2 : {final_scores['rouge2']:.4f}")
print(f"ROUGE-L : {final_scores['rougeL']:.4f}")

final_test_df.to_csv(
    "textrank_test_results.csv",
    index=False
)

print("Results saved successfully.")