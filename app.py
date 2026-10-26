"""Streamlit app for an extractive, traditional-NLP text summarization project."""

import re
from collections import Counter
from typing import Any, List, Tuple

import nltk
import numpy as np
import streamlit as st
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from statistics_page import render_statistics_page


st.set_page_config(page_title="SmartBrief | NLP Summarizer", page_icon="📝", layout="wide")


@st.cache_resource
def setup_tokenizer():
    """Download the NLTK sentence model when it is not available locally."""
    for resource, path in (("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab")):
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


def clean_sentence(sentence: str) -> str:
    """Normalize text used for feature calculation without altering displayed text."""
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-z0-9\s]", " ", sentence)
    return re.sub(r"\s+", " ", sentence).strip()


def tokenize_sentences(text: str) -> List[str]:
    """Split an input document into usable sentences, with an offline fallback."""
    try:
        sentences = sent_tokenize(text)
    except LookupError:
        sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if len(sentence.split()) >= 3]


def requested_count(sentence_count: int, ratio: float) -> int:
    return min(sentence_count, max(1, round(sentence_count * ratio)))


def normalise(values: np.ndarray) -> np.ndarray:
    maximum = values.max() if len(values) else 0
    return values / maximum if maximum else np.zeros(len(values))


def tfidf_scores(processed: List[str]) -> np.ndarray:
    """Score a sentence by the amount of distinctive vocabulary it contains."""
    try:
        matrix: Any = TfidfVectorizer(stop_words="english").fit_transform(processed)
        return normalise(np.asarray(matrix.sum(axis=1)).ravel())
    except ValueError:  # e.g., the document contains only stop words
        return np.ones(len(processed))


def weighted_tfidf_scores(processed: List[str]) -> np.ndarray:
    """Combine TF-IDF, sentence position, length, and repeated-keyword signals."""
    count = len(processed)
    tfidf = tfidf_scores(processed)
    position = np.array([1 - index / count for index in range(count)])
    lengths = normalise(np.array([len(sentence.split()) for sentence in processed], dtype=float))
    all_words = [word for sentence in processed for word in sentence.split()
                 if word not in ENGLISH_STOP_WORDS and len(word) > 2]
    frequency = Counter(all_words)
    keyword = np.array([sum(frequency[word] for word in sentence.split() if word in frequency)
                        for sentence in processed], dtype=float)
    return 0.35 * tfidf + 0.25 * position + 0.15 * lengths + 0.25 * normalise(keyword)


def textrank_scores(processed: List[str], damping: float = 0.85, iterations: int = 50) -> np.ndarray:
    """Rank sentences using PageRank on a TF-IDF cosine-similarity graph."""
    count = len(processed)
    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform(processed)
        graph = cosine_similarity(matrix)
    except ValueError:
        return np.ones(count)
    np.fill_diagonal(graph, 0)
    row_sums = graph.sum(axis=1, keepdims=True)
    transition = np.divide(graph, row_sums, out=np.zeros_like(graph), where=row_sums != 0)
    scores = np.full(count, 1 / count)
    for _ in range(iterations):
        scores = (1 - damping) / count + damping * transition.T.dot(scores)
    return normalise(scores)


def summarize(text: str, method: str, ratio: float) -> Tuple[str, List[str], np.ndarray, List[int]]:
    """Tokenize, score, select, and restore selected sentences to original order."""
    sentences = tokenize_sentences(text)
    if not sentences:
        return "", [], np.array([]), []
    processed = [clean_sentence(sentence) for sentence in sentences]
    scorers = {
        "TF-IDF": tfidf_scores,
        "Weighted TF-IDF": weighted_tfidf_scores,
        "TextRank": textrank_scores,
    }
    if method not in scorers:
        raise ValueError(f"Unsupported summarization method: {method}")
    scores = scorers[method](processed)
    selected = sorted(np.argsort(scores)[-requested_count(len(sentences), ratio):].tolist())
    return " ".join(sentences[index] for index in selected), sentences, scores, selected


def document_keywords(text: str, limit: int = 8) -> List[str]:
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    frequency = Counter(word for word in words if word not in ENGLISH_STOP_WORDS)
    return [word for word, _ in frequency.most_common(limit)]


DEMO_TEXT = """Artificial intelligence is changing how organisations process information. Long reports, news articles, and research documents can take significant time to read. Extractive text summarization identifies the most informative sentences from an original document. Traditional NLP approaches can score sentences using important words, their position, and their similarity to other sentences. TF-IDF gives more importance to distinctive terms, while TextRank treats sentences as connected nodes in a graph. The selected sentences are kept in their original order so the result remains readable. A concise summary helps readers understand a document before deciding whether to read it in full."""


setup_tokenizer()

with st.sidebar:
    page = st.radio("Navigate", ["Summarizer", "Summarized Statistics"])

if page == "Summarized Statistics":
    render_statistics_page(
        st.session_state.get("document", ""),
        summarize,
        tokenize_sentences,
    )
    st.stop()

st.title("📝 SmartBrief: NLP Text Summarization System")
st.caption("An extractive NLP system that converts long articles or documents into concise, readable summaries.")

with st.sidebar:
    st.header("Project modules")
    st.markdown("✅ Text / document input\n\n✅ Sentence tokenization\n\n✅ Important sentence extraction\n\n✅ Summary generation\n\n✅ Summary display & analytics")
    st.divider()
    st.markdown("**How it works**\n\nInput → Tokenize → Score sentences → Select key sentences → Display summary")
    if st.button("Load demo article", use_container_width=True):
        st.session_state.document = DEMO_TEXT

uploaded = st.file_uploader("Upload a text document (.txt or .md)", type=["txt", "md"])
if uploaded is not None:
    try:
        st.session_state.document = uploaded.getvalue().decode("utf-8")
        st.caption(f"Loaded: {uploaded.name}")
    except UnicodeDecodeError:
        st.error("Please upload a UTF-8 encoded text file.")

text = st.text_area("Paste an article or document", value=st.session_state.get("document", ""), height=260,
                    placeholder="Paste a multi-sentence article here, or upload a text file above…", key="document") or ""
settings, explanation = st.columns([2, 1])
with settings:
    method = st.selectbox("Sentence-scoring method", ["TF-IDF", "Weighted TF-IDF", "TextRank"])
    label = st.select_slider("Summary length", options=["Low", "Medium", "High"], value="Medium")
with explanation:
    descriptions = {"TF-IDF": "Scores sentences containing distinctive, informative terms.",
                    "Weighted TF-IDF": "Combines TF-IDF, position, length, and repeated keywords.",
                    "TextRank": "Ranks sentences by similarity using a PageRank-style graph."}
    st.info(descriptions[method])

ratio = {"Low": 0.20, "Medium": 0.30, "High": 0.40}[label]
generate_clicked = st.button("Generate summary", type="primary", use_container_width=True)
current_config = (text, method, ratio)
if generate_clicked or st.session_state.get("result_config") != current_config:
    if len(tokenize_sentences(text)) < 2:
        st.session_state.pop("result", None)
        st.session_state.pop("result_config", None)
        if generate_clicked:
            st.warning("Please provide at least two complete sentences to summarize.")
    else:
        with st.spinner("Tokenizing and ranking important sentences…"):
            summary, sentences, scores, selected = summarize(text, method, ratio)
        st.session_state.result = (summary, sentences, scores, selected, method)
        st.session_state.result_config = current_config

if st.session_state.get("result_config") == current_config and "result" in st.session_state:
    summary, sentences, scores, selected, used_method = st.session_state.result
    st.divider()
    st.subheader("Generated summary")
    st.success(summary)
    original_words, summary_words = len(text.split()), len(summary.split())
    first, second, third, fourth = st.columns(4)
    first.metric("Input sentences", len(sentences))
    second.metric("Selected sentences", len(selected))
    third.metric("Original words", original_words)
    fourth.metric("Compression", f"{(1 - summary_words / original_words) * 100:.0f}%" if original_words else "0%")
    keywords = document_keywords(text)
    if keywords:
        st.caption("Key terms detected: " + " • ".join(keywords))
    with st.expander("Show important sentence extraction (for project demonstration)"):
        st.caption(f"Method: {used_method}. Highlighted sentences were selected for the final summary.")
        for index, (sentence, score) in enumerate(zip(sentences, scores), start=1):
            marker = "⭐ Selected" if index - 1 in selected else "Not selected"
            st.markdown(f"**{index}. {marker} — score: {score:.3f}**")
            st.write(sentence)

st.divider()
st.caption("Traditional NLP project | Extractive summarization | TF-IDF • Weighted TF-IDF • TextRank")
