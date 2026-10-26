"""Dataset statistics and method comparisons for the Streamlit app."""

import csv
import hashlib
import math
import os
import random
import statistics
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import streamlit as st
from rouge_score import rouge_scorer


METHODS = ("TF-IDF", "Weighted TF-IDF", "TextRank")
ROUGE_METRICS = ("rouge1", "rouge2", "rougeL")
ROUGE_LABELS = {
    "rouge1": "ROUGE-1",
    "rouge2": "ROUGE-2",
    "rougeL": "ROUGE-L",
}
SUMMARY_RATIO = 0.30
EVALUATION_SAMPLE_SIZE = 100
EVALUATION_RANDOM_STATE = 42


def _record_digest(*values: str) -> bytes:
    """Create an unambiguous compact key for deduplication and overlap checks."""
    digest = hashlib.blake2b(digest_size=20)
    for value in values:
        encoded = value.encode("utf-8", errors="replace")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.digest()


@st.cache_data(show_spinner=False)
def _load_dataset_statistics(
    dataset_path: str, modified_ns: int
) -> Tuple[List[int], List[int], List[float]]:
    """Read full-corpus article and reference-summary distributions."""
    article_lengths: List[int] = []
    summary_lengths: List[int] = []
    compression_ratios: List[float] = []

    with open(dataset_path, encoding="utf-8-sig", newline="") as dataset_file:
        for row in csv.DictReader(dataset_file):
            article = (row.get("article") or "").strip()
            reference = (row.get("highlights") or "").strip()
            article_length = len(article.split())
            summary_length = len(reference.split())

            if not article_length:
                continue

            article_lengths.append(article_length)
            summary_lengths.append(summary_length)
            compression_ratios.append(summary_length / article_length)

    return article_lengths, summary_lengths, compression_ratios


@st.cache_data(show_spinner=False)
def _load_dataset_sample(
    dataset_path: str,
    modified_ns: int,
    sample_size: int = EVALUATION_SAMPLE_SIZE,
    random_state: int = EVALUATION_RANDOM_STATE,
) -> List[Tuple[str, str]]:
    """Select a reproducible sample uniformly from a CSV dataset."""
    with open(dataset_path, encoding="utf-8-sig", newline="") as dataset_file:
        record_count = sum(1 for _ in csv.DictReader(dataset_file))

    if not record_count:
        return []

    sample_indices = set(
        np.random.RandomState(random_state).choice(
            record_count,
            size=min(sample_size, record_count),
            replace=False,
        ).tolist()
    )
    evaluation_records: List[Tuple[str, str]] = []
    with open(dataset_path, encoding="utf-8-sig", newline="") as dataset_file:
        for index, row in enumerate(csv.DictReader(dataset_file)):
            if index in sample_indices:
                evaluation_records.append(
                    (
                        (row.get("article") or "").strip(),
                        (row.get("highlights") or "").strip(),
                    )
                )

    return evaluation_records


@st.cache_data(show_spinner=False)
def _load_clean_training_sample(
    train_path: str,
    validation_path: str,
    train_modified_ns: int,
    validation_modified_ns: int,
    sample_size: int = EVALUATION_SAMPLE_SIZE,
    random_state: int = EVALUATION_RANDOM_STATE,
) -> Tuple[List[Tuple[str, str]], int]:
    """Reservoir-sample cleaned training records without loading the CSV in memory."""
    validation_articles = set()
    with open(validation_path, encoding="utf-8-sig", newline="") as dataset_file:
        for row in csv.DictReader(dataset_file):
            article = row.get("article") or ""
            validation_articles.add(_record_digest(article))

    seen_pairs = set()
    sample: List[Tuple[str, str]] = []
    eligible_count = 0
    rng = random.Random(random_state)

    with open(train_path, encoding="utf-8-sig", newline="") as dataset_file:
        for row in csv.DictReader(dataset_file):
            article = row.get("article") or ""
            highlights = row.get("highlights") or ""
            article_words = len(article.split())
            if not article_words:
                continue

            pair_digest = _record_digest(article, highlights)
            if pair_digest in seen_pairs:
                continue
            seen_pairs.add(pair_digest)

            if _record_digest(article) in validation_articles:
                continue

            highlight_words = len(highlights.split())
            if highlight_words / article_words > 1:
                continue

            eligible_count += 1
            record = (article, highlights)
            if len(sample) < sample_size:
                sample.append(record)
            else:
                replacement = rng.randrange(eligible_count)
                if replacement < sample_size:
                    sample[replacement] = record

    return sample, eligible_count


@st.cache_data(show_spinner=False)
def _evaluate_methods(
    evaluation_records: Tuple[Tuple[str, str], ...],
    _summarize: Callable,
    _tokenize_sentences: Callable,
) -> Tuple[List[Dict[str, float]], int]:
    """Compute comparable ROUGE means from the app's current algorithms."""
    scorer = rouge_scorer.RougeScorer(list(ROUGE_METRICS), use_stemmer=True)
    scores: Dict[str, Dict[str, List[float]]] = {
        method: {metric: [] for metric in ROUGE_METRICS} for method in METHODS
    }
    evaluated_articles = 0

    for article, reference in evaluation_records:
        if not reference or len(_tokenize_sentences(article)) < 2:
            continue

        evaluated_articles += 1
        for method in METHODS:
            summary = _summarize(article, method, SUMMARY_RATIO)[0]
            article_scores = scorer.score(reference, summary)
            for metric in ROUGE_METRICS:
                scores[method][metric].append(article_scores[metric].fmeasure)

    means = [
        {
            "Method": method,
            **{
                ROUGE_LABELS[metric]: (
                    statistics.fmean(scores[method][metric])
                    if scores[method][metric]
                    else 0.0
                )
                for metric in ROUGE_METRICS
            },
        }
        for method in METHODS
    ]
    return means, evaluated_articles


def _histogram_data(values: List[float], bin_count: int) -> Dict[str, List]:
    observations = np.asarray(values, dtype=float)
    observations = observations[np.isfinite(observations)]
    if not len(observations):
        return {"Range": [], "Count": []}

    zero_count = int(np.count_nonzero(observations == 0))
    positive_values = observations[observations > 0]
    use_log_bins = (
        len(positive_values) > 1
        and positive_values.max() / positive_values.min() >= 20
    )

    if use_log_bins:
        edges = np.geomspace(
            positive_values.min(), positive_values.max(), bin_count + 1
        )
        counts, edges = np.histogram(positive_values, bins=edges)
        labels = ["0"] if zero_count else []
        if zero_count:
            counts = np.insert(counts, 0, zero_count)
        labels.extend(
            f"{start:,.2g}–{end:,.2g}"
            for start, end in zip(edges[:-1], edges[1:])
        )
    else:
        minimum = observations.min()
        maximum = observations.max()
        if minimum == maximum:
            padding = max(abs(minimum) * 0.01, 0.5)
            edges = np.linspace(
                minimum - padding, maximum + padding, bin_count + 1
            )
        else:
            edges = np.linspace(minimum, maximum, bin_count + 1)
        counts, edges = np.histogram(observations, bins=edges)
        labels = [
            f"{start:,.2f}–{end:,.2f}"
            for start, end in zip(edges[:-1], edges[1:])
        ]

    return {"Range": labels, "Count": counts.tolist()}


def _distribution_chart(
    title: str,
    values: List[float],
    x_label: str,
    y_label: str,
    description: str,
) -> None:
    st.subheader(title)
    if len(values) < 2:
        st.info("At least two dataset observations are required to draw this distribution.")
        return

    bins = min(30, max(10, round(math.sqrt(len(values)))))
    finite_values = [value for value in values if math.isfinite(value)]
    positive_values = [value for value in finite_values if value > 0]
    uses_log_bins = (
        len(positive_values) > 1
        and max(positive_values) / min(positive_values) >= 20
    )
    st.bar_chart(
        _histogram_data(values, bins),
        x="Range",
        y="Count",
        x_label=x_label,
        y_label=y_label,
        height=320,
    )
    bin_note = (
        " Log-spaced bins are used to spread out a highly skewed range; "
        "zero values are shown separately."
        if uses_log_bins
        else " Equal-width bins are used."
    )
    st.caption(description + bin_note)


def _article_method_comparison(
    article: str,
    summarize: Callable,
    tokenize_sentences: Callable,
    rouge_rows: List[Dict[str, float]],
) -> None:
    st.subheader("Method Performance for Current Article")
    sentences = tokenize_sentences(article)
    if len(sentences) < 2:
        st.info(
            "Enter an article with at least two complete sentences on the "
            "Summarizer page to compare its methods."
        )
        return

    rows = []
    for method in METHODS:
        summary, _, sentence_scores, selected = summarize(
            article, method, SUMMARY_RATIO
        )
        selected_scores = [float(sentence_scores[index]) for index in selected]
        score = statistics.fmean(selected_scores) if selected_scores else 0.0
        article_length = len(article.split())
        summary_length = len(summary.split())
        rows.append(
            {
                "Method": method,
                "Summary Length": summary_length,
                "Compression Ratio": (
                    summary_length / article_length if article_length else 0.0
                ),
                "Score": score,
                "Summary": summary,
            }
        )

    st.bar_chart(
        {
            "Method": [row["Method"] for row in rows],
            "Mean Selected Sentence Score": [row["Score"] for row in rows],
        },
        x="Method",
        y="Mean Selected Sentence Score",
        x_label="Summarization Method",
        y_label="Mean Selected Sentence Score",
        height=300,
    )
    st.caption(
        "All methods use the same 30% sentence-selection target. The score is "
        "the mean existing sentence-ranking score of the selected sentences; "
        "it is a salience proxy, not reference-based quality."
    )
    st.dataframe(
        [
            {
                "Method": row["Method"],
                "Summary Length": row["Summary Length"],
                "Compression Ratio": round(row["Compression Ratio"], 3),
                "Score": round(row["Score"], 4),
                "Summary": row["Summary"],
            }
            for row in rows
        ],
        hide_index=True,
        width="stretch",
    )

    best_score = max(row["Score"] for row in rows)
    best_methods = [
        row["Method"]
        for row in rows
        if math.isclose(row["Score"], best_score, rel_tol=1e-9, abs_tol=1e-12)
    ]
    macro_scores = {
        row["Method"]: statistics.fmean(
            row[ROUGE_LABELS[metric]] for metric in ROUGE_METRICS
        )
        for row in rouge_rows
    }
    recommended_method = max(
        best_methods,
        key=lambda method: macro_scores.get(method, 0.0),
    )
    st.markdown("### Suitable Method for This Article")
    with st.container(border=True):
        st.markdown(f"**Recommended Method: {recommended_method}**")
        if len(best_methods) == 1:
            st.write(
                f"{recommended_method} has the highest mean selected-sentence "
                f"score ({best_score:.4f}) for this article. The score measures "
                "the method's own sentence-ranking signal; without a reference "
                "summary, it is not a ROUGE-based quality judgment."
            )
        else:
            tie_break = (
                f"Its dataset macro-average ROUGE score ({macro_scores[recommended_method]:.4f}) "
                "is highest among the tied methods."
                if macro_scores
                else "The listed method order resolves the exact tie."
            )
            st.write(
                f"The methods tie for the highest mean selected-sentence "
                f"score ({best_score:.4f}) for this article. {tie_break} "
                "The article-specific score is a sentence-ranking salience "
                "proxy, not a ROUGE-based quality judgment."
            )


def _render_split_comparison(
    training_rows: List[Dict[str, float]],
    test_rows: List[Dict[str, float]],
    training_sample_size: int,
    test_sample_size: int,
    training_evaluated: int,
    test_evaluated: int,
    cleaned_training_count: int,
) -> None:
    """Show comparable sample-level scores and test-minus-training differences."""
    st.caption(
        f"From {cleaned_training_count:,} cleaned training records, a "
        f"reproducible sample of {training_sample_size} was compared with "
        f"{test_sample_size} test records. ROUGE was calculated for "
        f"{training_evaluated} eligible training articles and "
        f"{test_evaluated} eligible test articles."
    )
    st.info(
        "These are split-based evaluation scores for the app's fixed "
        "extractive methods, not model training results: the methods do not "
        "fit parameters on the training split. Training-split ROUGE is not "
        "an independent measure of generalization; the test split is held out."
    )

    training_by_method = {row["Method"]: row for row in training_rows}
    test_by_method = {row["Method"]: row for row in test_rows}
    difference_rows = []

    for metric in ROUGE_METRICS:
        label = ROUGE_LABELS[metric]
        chart_data = {"Method": [], "Split": [], "Score": []}
        for method in METHODS:
            chart_data["Method"].extend([method, method])
            chart_data["Split"].extend(["Training", "Test"])
            chart_data["Score"].extend(
                [
                    training_by_method[method][label],
                    test_by_method[method][label],
                ]
            )
        st.subheader(f"{label}: training vs test sample")
        st.bar_chart(
            chart_data,
            x="Method",
            y="Score",
            color="Split",
            stack=False,
            x_label="Extractive Method",
            y_label=f"{label} F-measure",
            height=300,
        )

    for method in METHODS:
        row: Dict[str, object] = {"Method": method}
        for metric in ROUGE_METRICS:
            label = ROUGE_LABELS[metric]
            training_score = training_by_method[method][label]
            test_score = test_by_method[method][label]
            row[f"Training {label}"] = round(training_score, 4)
            row[f"Test {label}"] = round(test_score, 4)
            row[f"Test - Train {label}"] = round(
                test_score - training_score, 4
            )
        difference_rows.append(row)

    st.subheader("Test-minus-training score differences")
    st.dataframe(difference_rows, hide_index=True, width="stretch")
    st.caption(
        "A positive Test - Train difference means the sample's test ROUGE "
        "score is higher; a negative value means it is lower. Differences "
        "describe these samples and are not statistical significance tests."
    )


def render_statistics_page(
    current_article: str,
    summarize: Callable,
    tokenize_sentences: Callable,
) -> None:
    """Render the standalone statistics and comparison page."""
    st.title("Summarized Statistics")
    st.caption(
        "Dataset-level distributions and transparent comparisons of the "
        "project's three extractive summarization methods."
    )

    dataset_path = Path(__file__).with_name("test.csv")
    validation_path = Path(__file__).with_name("validation.csv")
    train_path = Path(__file__).with_name("train.csv")
    if not dataset_path.is_file():
        st.error(
            "The held-out dataset file test.csv was not found beside the app. "
            "Place the project's test.csv file there to display dataset statistics."
        )
        _article_method_comparison(
            current_article, summarize, tokenize_sentences, []
        )
        return
    if not validation_path.is_file():
        st.error(
            "The validation dataset file validation.csv was not found beside the app. "
            "Place the project's validation.csv file there to compare ROUGE scores."
        )

    try:
        dataset_mtime = os.stat(dataset_path).st_mtime_ns
        article_lengths, summary_lengths, compression_ratios = (
            _load_dataset_statistics(str(dataset_path), dataset_mtime)
        )
        if validation_path.is_file():
            validation_mtime = os.stat(validation_path).st_mtime_ns
            evaluation_records = _load_dataset_sample(
                str(validation_path), validation_mtime
            )
        else:
            evaluation_records = []
    except (OSError, csv.Error, UnicodeError) as error:
        st.error(f"Could not read the held-out dataset: {error}")
        _article_method_comparison(
            current_article, summarize, tokenize_sentences, []
        )
        return

    if not article_lengths:
        st.error("The held-out dataset contains no non-empty articles.")
        _article_method_comparison(
            current_article, summarize, tokenize_sentences, []
        )
        return

    rouge_rows, evaluated_articles = _evaluate_methods(
        tuple(evaluation_records), summarize, tokenize_sentences
    )
    st.subheader("Dataset Overview")
    if not evaluated_articles:
        st.warning(
            "No eligible multi-sentence articles were found in the evaluation sample."
        )
    else:
        macro_scores = {
            row["Method"]: statistics.fmean(
                row[ROUGE_LABELS[metric]] for metric in ROUGE_METRICS
            )
            for row in rouge_rows
        }
        best_method = max(METHODS, key=lambda method: macro_scores[method])
        best_by_metric = {
            metric: max(
                rouge_rows, key=lambda row: row[ROUGE_LABELS[metric]]
            )
            for metric in ROUGE_METRICS
        }
        distribution_cards = st.columns(4)
        distribution_cards[0].metric(
            "Number of Articles", f"{len(article_lengths):,}"
        )
        distribution_cards[1].metric(
            "Average Article Length", f"{statistics.fmean(article_lengths):,.0f} words"
        )
        distribution_cards[2].metric(
            "Average Summary Length", f"{statistics.fmean(summary_lengths):,.0f} words"
        )
        distribution_cards[3].metric(
            "Average Compression Ratio",
            f"{statistics.fmean(compression_ratios):.3f}",
        )
        performance_cards = st.columns(4)
        performance_cards[0].metric("Best Performing Method", best_method)
        performance_cards[1].metric(
            "Best ROUGE-1",
            f"{best_by_metric['rouge1']['ROUGE-1']:.3f}",
            help=f"Highest mean on the evaluation sample: {best_by_metric['rouge1']['Method']}",
        )
        performance_cards[2].metric(
            "Best ROUGE-2",
            f"{best_by_metric['rouge2']['ROUGE-2']:.3f}",
            help=f"Highest mean on the evaluation sample: {best_by_metric['rouge2']['Method']}",
        )
        performance_cards[3].metric(
            "Best ROUGE-L",
            f"{best_by_metric['rougeL']['ROUGE-L']:.3f}",
            help=f"Highest mean on the evaluation sample: {best_by_metric['rougeL']['Method']}",
        )
        st.caption(
            f"Distribution metrics use all {len(article_lengths):,} rows with "
            "non-empty articles in test.csv. ROUGE averages use "
            f"{evaluated_articles} eligible articles among the "
            f"{EVALUATION_SAMPLE_SIZE} validation records sampled with "
            f"random_state={EVALUATION_RANDOM_STATE} to match the project "
            "pipeline's evaluation split. "
            "Best Performing Method is selected by the highest macro-average "
            "of ROUGE-1, ROUGE-2, and ROUGE-L."
        )

    article_median = statistics.median(article_lengths)
    article_p10, article_p90 = np.percentile(article_lengths, [10, 90])
    summary_median = statistics.median(summary_lengths)
    summary_p10, summary_p90 = np.percentile(summary_lengths, [10, 90])
    compression_median = statistics.median(compression_ratios)

    st.divider()
    _distribution_chart(
        "Article Length Distribution",
        article_lengths,
        "Article Length (Words)",
        "Number of Articles",
        f"The median article contains about {article_median:,.0f} words. "
        f"The central 80% range is approximately {article_p10:,.0f}–"
        f"{article_p90:,.0f} words, indicating the observed spread.",
    )
    _distribution_chart(
        "Summary Length Distribution",
        summary_lengths,
        "Summary Length (Words)",
        "Number of Summaries",
        f"Reference summaries have a median length of about "
        f"{summary_median:,.0f} words, with a central 80% range of "
        f"{summary_p10:,.0f}–{summary_p90:,.0f} words.",
    )
    _distribution_chart(
        "Compression Ratio Distribution",
        compression_ratios,
        "Compression Ratio",
        "Number of Articles",
        f"Compression ratio is reference-summary words divided by article "
        f"words (median {compression_median:.3f}). A lower ratio means the "
        "summary contains substantially fewer words than the original article.",
    )

    st.divider()
    st.subheader("ROUGE Score Comparison of Summarization Methods")
    if evaluated_articles:
        chart_rows = {
            "Method": [],
            "Metric": [],
            "Score": [],
        }
        for row in rouge_rows:
            for metric in ROUGE_METRICS:
                chart_rows["Method"].append(row["Method"])
                chart_rows["Metric"].append(ROUGE_LABELS[metric])
                chart_rows["Score"].append(row[ROUGE_LABELS[metric]])
        st.bar_chart(
            chart_rows,
            x="Method",
            y="Score",
            color="Metric",
            stack=False,
            x_label="Summarization Method",
            y_label="ROUGE Score",
            height=360,
        )
        st.caption(
            "Bars show mean ROUGE F-measures against reference highlights, "
            f"calculated with the same {SUMMARY_RATIO:.0%} sentence-selection "
            "target for each app method."
        )
        st.dataframe(rouge_rows, hide_index=True, width="stretch")

    st.divider()
    st.subheader("Training-split vs held-out test-split evaluation")
    st.caption(
        "Run a separate, reproducible sample comparison for the existing "
        "extractive methods. This does not change how the summarizer works."
    )
    if not train_path.is_file() or not validation_path.is_file():
        st.warning(
            "Both train.csv and validation.csv are required to prepare the "
            "cleaned training sample and check train-validation overlap."
        )
    else:
        comparison_key = (
            os.stat(train_path).st_mtime_ns,
            os.stat(validation_path).st_mtime_ns,
            os.stat(dataset_path).st_mtime_ns,
        )
        if st.button(
            "Run train vs test comparison",
            key="run_train_test_comparison",
        ):
            try:
                with st.spinner(
                    "Preparing reproducible samples and calculating ROUGE. "
                    "The first run reads the full training CSV."
                ):
                    training_sample, cleaned_training_count = (
                        _load_clean_training_sample(
                            str(train_path),
                            str(validation_path),
                            comparison_key[0],
                            comparison_key[1],
                        )
                    )
                    test_sample = _load_dataset_sample(
                        str(dataset_path), comparison_key[2]
                    )
                    training_rows, training_evaluated = _evaluate_methods(
                        tuple(training_sample), summarize, tokenize_sentences
                    )
                    test_rows, test_evaluated = _evaluate_methods(
                        tuple(test_sample), summarize, tokenize_sentences
                    )
                st.session_state["train_test_comparison"] = {
                    "key": comparison_key,
                    "training_rows": training_rows,
                    "test_rows": test_rows,
                    "training_sample_size": len(training_sample),
                    "test_sample_size": len(test_sample),
                    "training_evaluated": training_evaluated,
                    "test_evaluated": test_evaluated,
                    "cleaned_training_count": cleaned_training_count,
                }
            except (OSError, csv.Error, UnicodeError) as error:
                st.error(f"Could not prepare the train/test comparison: {error}")

        comparison = st.session_state.get("train_test_comparison")
        if comparison and comparison.get("key") == comparison_key:
            if comparison["training_evaluated"] and comparison["test_evaluated"]:
                _render_split_comparison(
                    comparison["training_rows"],
                    comparison["test_rows"],
                    comparison["training_sample_size"],
                    comparison["test_sample_size"],
                    comparison["training_evaluated"],
                    comparison["test_evaluated"],
                    comparison["cleaned_training_count"],
                )
            else:
                st.warning(
                    "The samples did not contain eligible multi-sentence "
                    "articles for both splits, so a comparison could not be shown."
                )

    _article_method_comparison(
        current_article, summarize, tokenize_sentences, rouge_rows
    )

    st.divider()
    st.subheader("Academic Interpretation")
    st.markdown(
        "- Reference summaries are substantially shorter than source articles, "
        "as reflected by the summary-length and compression distributions.\n"
        "- A lower compression ratio indicates greater reduction in text size.\n"
        "- ROUGE-1, ROUGE-2, and ROUGE-L quantify unigram, bigram, and "
        "longest-common-subsequence overlap with reference summaries.\n"
        "- Comparing mean ROUGE scores on the same held-out examples provides "
        "a quantitative comparison of the three implemented extractive methods.\n"
        "- The optional training-vs-test panel evaluates the fixed methods on "
        "equal-sized samples; it does not train a learned summarization model."
    )
