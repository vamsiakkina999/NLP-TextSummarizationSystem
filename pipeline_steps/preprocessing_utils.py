"""Text preprocessing helpers shared by the analysis stages."""

import re
from typing import List

from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_valid_sentence(processed_sentence, min_words=2):
    return bool(processed_sentence) and len(processed_sentence.split()) >= min_words


def is_metadata_sentence(sentence):
    text = sentence.strip()
    if not text:
        return True
    if re.fullmatch(
        r"(associated press|reuters|afp|cnn|bbc)",
        text,
        flags=re.IGNORECASE,
    ):
        return True
    if re.search(r"\b\d{1,2}:\d{2}\b", text):
        return True
    return bool(re.search(r"\b(19|20)\d{2}\b", text))


def remove_special_characters(text):
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_stopwords(words):
    return [word for word in words if word not in stop_words]


def preprocess_sentence(sentence):
    normalized = normalize_text(sentence)
    cleaned = remove_special_characters(normalized)
    words = word_tokenize(cleaned)
    filtered_words = remove_stopwords(words)

    lemmas = []
    for word in filtered_words:
        tag = pos_tag([word])[0][1]
        lemmas.append(lemmatizer.lemmatize(word, get_wordnet_pos(tag)))

    return " ".join(lemmas)


def preprocess_article(article):
    sentence_data: List[dict[str, str]] = []
    for sentence in sent_tokenize(article):
        if is_metadata_sentence(sentence):
            continue

        processed = preprocess_sentence(sentence)
        if not is_valid_sentence(processed):
            continue

        sentence_data.append({"original": sentence, "processed": processed})

    return sentence_data
