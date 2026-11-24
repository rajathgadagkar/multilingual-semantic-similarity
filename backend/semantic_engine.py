#!/usr/bin/env python3
"""
Hybrid Multilingual Semantic FAQ Search Engine (FastAPI Microservice).

Architecture:
    Stage 1 (Bi-Encoder Retrieval):
        Pre-computes L2-normalized 768-dim embeddings for all FAQ questions
        using `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
        and retrieves the top-K candidates via cosine similarity.
    Stage 2 (Cross-Encoder Re-Ranking):
        Jointly encodes each (query, candidate FAQ) pair using the fine-tuned
        XLM-RoBERTa STS cross-encoder to produce token-level relevance scores.

Author: Rajath Gadagkar <rajathgadagkar@gmail.com>
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder, SentenceTransformer, util

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SUPPORTED_LANGS: List[str] = ["en", "de", "es", "fr", "it"]
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("MFAQ_DATA_DIR", str(BASE_DIR.parent / "data" / "mfaq")))
BI_ENCODER_MODEL = os.getenv(
    "BI_ENCODER_MODEL",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)
CROSS_ENCODER_PATH = os.getenv(
    "CROSS_ENCODER_PATH",
    str(BASE_DIR / "models" / "xlmroberta_sts_finetuned"),
)
DEFAULT_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "20"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# Request / Response Schemas
# --------------------------------------------------------------------------- #
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language user query")
    lang: str = Field("en", description="ISO 639-1 language code (en, de, es, fr, it)")
    top_k: int = Field(1, ge=1, le=10, description="Number of ranked FAQ answers to return")


class SearchResultItem(BaseModel):
    question: str
    answer: str
    cross_score: float
    cosine_score: float
    lang: str


class SearchResponse(BaseModel):
    question: str
    answer: str
    cross_score: float
    cosine_score: float
    lang: str
    latency_ms: float
    results: List[SearchResultItem]


# --------------------------------------------------------------------------- #
# Engine Lifecycle
# --------------------------------------------------------------------------- #
class HybridSemanticEngine:
    """Two-stage Bi-Encoder + Cross-Encoder multilingual FAQ search engine."""

    def __init__(self) -> None:
        self.device = DEVICE
        self.bi_encoder: Optional[SentenceTransformer] = None
        self.cross_encoder: Optional[CrossEncoder] = None
        self.index: Dict[str, Dict[str, object]] = {}

    def load(self) -> None:
        print(f"[engine] Loading bi-encoder ({BI_ENCODER_MODEL}) on {self.device}...")
        self.bi_encoder = SentenceTransformer(BI_ENCODER_MODEL, device=self.device)

        print(f"[engine] Loading cross-encoder ({CROSS_ENCODER_PATH}) on {self.device}...")
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_PATH, device=self.device)

        for lang in SUPPORTED_LANGS:
            csv_path = DATA_DIR / f"mfaq_{lang}_test.csv"
            if not csv_path.is_file():
                raise FileNotFoundError(f"MFAQ dataset for '{lang}' not found at {csv_path}")

            df = pd.read_csv(csv_path).dropna(subset=["question", "answer"])
            questions = df["question"].astype(str).tolist()
            answers = df["answer"].astype(str).tolist()

            embeddings = self.bi_encoder.encode(
                questions,
                batch_size=32,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            self.index[lang] = {
                "questions": questions,
                "answers": answers,
                "embeddings": embeddings,
            }
            print(f"[engine] Indexed {len(questions)} FAQs for language '{lang}'")

    def search(self, query: str, lang: str, top_k: int = 1) -> SearchResponse:
        if self.bi_encoder is None or self.cross_encoder is None:
            raise RuntimeError("Semantic engine is not loaded.")

        lang_key = lang.lower()
        if lang_key not in self.index:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language '{lang}'. Available: {', '.join(SUPPORTED_LANGS)}",
            )

        t0 = time.perf_counter()
        lang_store = self.index[lang_key]
        questions: List[str] = lang_store["questions"]  # type: ignore[assignment]
        answers: List[str] = lang_store["answers"]  # type: ignore[assignment]
        embeddings = lang_store["embeddings"]

        # Stage 1: Fast Bi-Encoder Retrieval
        query_vec = self.bi_encoder.encode(
            query, convert_to_tensor=True, show_progress_bar=False
        )
        query_vec = torch.nn.functional.normalize(query_vec, p=2, dim=0)
        sims = util.cos_sim(query_vec, embeddings)[0]
        cand_k = min(DEFAULT_TOP_K, len(sims))
        cand_indices = torch.topk(sims, k=cand_k).indices.tolist()

        # Stage 2: Precision Cross-Encoder Re-Ranking
        pairs = [
            (query, f"{questions[idx]} ||| {answers[idx]}") for idx in cand_indices
        ]
        cross_scores = self.cross_encoder.predict(pairs)

        ranked = sorted(
            zip(cand_indices, cross_scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        items = [
            SearchResultItem(
                question=questions[idx],
                answer=answers[idx],
                cross_score=float(c_score),
                cosine_score=float(sims[idx]),
                lang=lang_key,
            )
            for idx, c_score in ranked[:top_k]
        ]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        best = items[0]
        return SearchResponse(
            question=best.question,
            answer=best.answer,
            cross_score=best.cross_score,
            cosine_score=best.cosine_score,
            lang=best.lang,
            latency_ms=round(elapsed_ms, 2),
            results=items,
        )


# --------------------------------------------------------------------------- #
# FastAPI Application
# --------------------------------------------------------------------------- #
engine = HybridSemanticEngine()
app = FastAPI(
    title="Multilingual Semantic FAQ Search Engine",
    description="Hybrid Bi-Encoder + Fine-Tuned XLM-RoBERTa Cross-Encoder microservice.",
    version="1.0.0",
)


@app.on_event("startup")
def _startup() -> None:
    engine.load()


@app.get("/health")
def health() -> Dict[str, object]:
    return {
        "status": "ok",
        "device": engine.device,
        "languages": list(engine.index.keys()),
    }


@app.get("/languages")
def languages() -> Dict[str, int]:
    return {lang: len(store["questions"]) for lang, store in engine.index.items()}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    return engine.search(req.query, req.lang, top_k=req.top_k)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("semantic_engine:app", host="0.0.0.0", port=8000, reload=False)
