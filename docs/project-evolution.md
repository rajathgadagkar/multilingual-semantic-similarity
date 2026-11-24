# Project Evolution

This repository captures the complete technical progression of the project across three major phases.

---

## Phase 1 — Initial Concept & Problem Formulation

- **Core Problem**: Quantify semantic similarity between arbitrary sentence pairs on a continuous scale ($0$–$5$), moving beyond surface lexical overlap (Jaccard, TF-IDF) and static word vectors (Word2Vec, LSA).
- **Initial Research Direction**: Surveyed contextual transformer architectures (BERT, RoBERTa, SBERT, DeBERTa) with Pearson correlation, Spearman rank correlation, and MSE as primary evaluation metrics.
- **Target Applications**: Semantic search, content recommendation, duplicate question detection, and paraphrase/plagiarism analysis.

---

## Phase 2 — Multilingual Transformer Fine-Tuning & Comparative Evaluation

- **Scope Expansion**: Expanded from monolingual STS to a **10-language multilingual STS benchmark** (`PhilipMay/stsb_multi_mt`: DE, EN, ES, FR, IT, NL, PL, PT, RU, ZH — 86,280 total sentence pairs).
- **Controlled Comparison**: Fine-tuned **XLM-RoBERTa**, **mBERT**, and **DistilBERT** under identical training hyperparameters (`lr=2e-5`, `batch_size=16`, `epochs=3`, `AdamW`).
- **Outcome**: Established **XLM-RoBERTa** as the strongest backbone (`0.8273` Test Pearson, `0.8169` Test Spearman, `0.7521` MSE), outperforming mBERT and DistilBERT across every single language split.
- **Artifacts**: Preserved in `experiments/notebooks/` and `experiments/results/`.

---

## Phase 3 — Real-Time Multilingual Semantic FAQ Search System

- **Productionizing the Model**: Integrated the fine-tuned XLM-RoBERTa cross-encoder into a practical, local-first **real-time multilingual FAQ search engine** evaluated on the Hugging Face `clips/mfaq` dataset.
- **Hybrid Engineering**: Combined a fast multilingual Bi-Encoder (`paraphrase-multilingual-mpnet-base-v2`) for Top-20 candidate recall with the fine-tuned XLM-RoBERTa Cross-Encoder for token-level re-ranking, achieving sub-3-second CPU query latency.
- **Full-Stack Delivery**: Built a FastAPI inference microservice (`backend/`), an Express + React + Tailwind web application (`web/`), automated CI benchmark verification (`.github/workflows/ci.yml`), and an interactive GitHub Pages dashboard (`docs/index.html`).
