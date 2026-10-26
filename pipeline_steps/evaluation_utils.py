"""ROUGE evaluation helpers shared by analysis stages."""

from rouge_score import rouge_scorer


scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True,
)


def evaluate_summary(generated_summary, reference_summary):
    scores = scorer.score(reference_summary, generated_summary)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }
