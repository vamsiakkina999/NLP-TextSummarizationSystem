# SmartBrief: NLP Text Summarization System

SmartBrief is an **extractive text summarization** project. It automatically identifies important sentences in a long article and combines them into a shorter, readable summary. It is built with Python, Streamlit, NLTK, and scikit-learn.

## Required project modules

| Module | Implementation in this project |
| --- | --- |
| Text/document input | Paste text directly or upload a UTF-8 `.txt` / `.md` file. |
| Sentence tokenization | NLTK `sent_tokenize` divides the document into sentences. |
| Important sentence extraction | Each sentence is assigned an importance score; the highest-scoring sentences are selected. |
| Summary generation | Selected sentences are returned in their original document order. |
| Summary display | The app shows the summary, compression metrics, keywords, and an explainable sentence-score view. |

## NLP methods used

1. **TF-IDF**: gives higher scores to sentences containing distinctive terms.
2. **Weighted TF-IDF**: combines TF-IDF, sentence position, sentence length, and repeated keyword frequency.
3. **TextRank**: builds a graph from cosine similarity between sentences, then uses a PageRank-style algorithm to find central sentences.

The system is extractive: it selects original sentences rather than inventing new text. This makes the result explainable and avoids unsupported claims.

## How to run

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown in the terminal. For a quick presentation, select **Load demo article** and choose a method. Choose a **Low**, **Medium**, or **High** summary length; the summary is regenerated when the method, document, or summary length changes. Click **Generate summary** to rerun it with the current settings. Expand **Show important sentence extraction** to demonstrate the selected method's scoring stage.

Use **Summarized Statistics** in the sidebar to open the academic dashboard. Its length and compression distributions are calculated from the complete `test.csv` held-out dataset (article text and reference highlights). Highly skewed distributions use log-spaced histogram bins so outliers do not collapse most records into one bar; zero compression ratios are counted separately. The grouped ROUGE chart evaluates the app's three existing methods against a reproducible 100-record sample from `validation.csv` (`random_state=42`), matching the evaluation split used by `pipeline.py`; it uses ROUGE-1, ROUGE-2, and ROUGE-L F-measures. The current-article recommendation compares the mean sentence-ranking scores of the sentences selected by each method; this is an input-specific salience proxy, not a reference-based ROUGE score.

The dashboard also offers an optional **Run train vs test comparison**. It applies the pipeline's training-data cleaning rules, selects a reproducible 100-record sample from the cleaned training data and a 100-record sample from `test.csv`, and compares the same three fixed methods using ROUGE-1, ROUGE-2, and ROUGE-L F-measures. It visualizes both split scores and their test-minus-training differences. The first run reads the full training CSV and may take time; results are cached for the current dataset files. This is a split-based evaluation, not model training: the app's extractive methods do not fit parameters on the training data, so training-split scores should not be interpreted as a learned model's training performance.

The analysis pipeline remains runnable with `python pipeline.py`. Its sequential stages are separated into `pipeline_steps/` to make data exploration, preprocessing, scoring/evaluation, weight optimization, and TextRank easier to inspect and modify.

## Suggested viva explanation

“The user provides an article. The system tokenizes it into sentences and converts sentences into TF-IDF feature vectors. Depending on the chosen algorithm, it assigns each sentence an importance score. It selects the top-ranked sentences, restores their original order for readability, and displays the final summary with compression statistics. The extraction panel makes the model’s selection transparent.”
