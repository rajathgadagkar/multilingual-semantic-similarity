# System Architecture: Hybrid Bi-Encoder + Cross-Encoder FAQ Search

## Design Goals

The real-time multilingual FAQ search system was engineered around four operational constraints:

1. **Sub-3-Second Latency on CPU** — Fast interactive responses without requiring GPU infrastructure.
2. **High Retrieval Precision** — Accurate semantic matching even when user queries paraphrase or omit exact FAQ keywords.
3. **Local-First Deployment** — Fully self-contained inference with zero reliance on external commercial APIs.
4. **Multilingual Coverage** — Unified pipeline supporting multiple European languages out of the box (`en`, `de`, `es`, `fr`, `it`), backed by a 10-language STS cross-encoder.

---

## Two-Stage Hybrid Pipeline

### Stage 1: Dense Bi-Encoder Candidate Retrieval

- **Model**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- **Offline Indexing**: At server startup, all FAQ questions per language are encoded into 768-dimensional vectors and L2-normalized so dot product equals cosine similarity.
- **Query-Time Retrieval**: The incoming user query is encoded once ($O(1)$ transformer forward pass), compared against all $N$ stored FAQ vectors via vectorized matrix multiplication, and the **Top-$K$ ($K = 20$)** candidates are forwarded to Stage 2.

### Stage 2: Fine-Tuned XLM-RoBERTa Cross-Encoder Re-Ranking

- **Model**: `xlm-roberta-base` fine-tuned for 3 epochs on 57,490 multilingual STS-B pairs (`PhilipMay/stsb_multi_mt`).
- **Joint Token-Level Attention**: Each candidate pair `[CLS] query [SEP] question ||| answer [SEP]` is passed jointly through XLM-RoBERTa's self-attention layers, enabling fine-grained lexical, syntactic, and semantic alignment across query and candidate tokens.
- **Final Ranking**: Candidates are sorted by their predicted semantic relevance score and returned alongside both `cross_score` and `cosine_score` plus request latency in milliseconds.

---

## Comparative Architecture Analysis

| Approach | Precision | CPU Latency | Scalability | Semantic Capability | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Pure Cross-Encoder** | Very High | Moderate–High ($O(N)$ passes) | Low | High (token-level) | Rejected — latency degrades linearly as corpus grows |
| **Pure Bi-Encoder** | Medium | Very Low ($O(1)$ pass) | Very High | Medium (embedding-level) | Rejected — independent encoding misses nuanced token interactions |
| **BM25 / Lexical** | Low | Very Low | Very High | Low (keyword overlap) | Rejected — fails on paraphrases and synonymy |
| **Cloud LLMs** | High | Very High (network + compute) | Moderate | Very High | Rejected — violates local-deployment and cost constraints |
| **Hybrid Bi-Encoder + Cross-Encoder (Ours)** | **High** | **Low–Moderate (< 3s)** | **High (FAISS-ready)** | **High (two-stage)** | **Selected** — combines high recall with token-level precision |

---

## Service Topology

- **`backend/semantic_engine.py` (FastAPI, port `8000`)**: Loads both transformer models and in-memory normalized FAQ indices on startup; exposes `/health`, `/languages`, and `/search`.
- **`web/server/` (Express + TypeScript, port `8080`)**: Serves the React client, loads language-specific FAQ datasets from `data/mfaq/`, and proxies search requests to the FastAPI microservice.
- **`web/client/` (React + Vite + Tailwind + Radix UI)**: Provides language switching, live FAQ browsing, and semantic search results rendering.
