#!/usr/bin/env python3
"""
pncp_reranker.py — reranking in-process para item × produto.

Usa modelos de sequence-classification do HuggingFace, como:
  BAAI/bge-reranker-v2-m3

Se o modelo ou dependências não estiverem disponíveis, o chamador pode
fazer fallback para o ranking bruto do retrieval.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    _RERANK_OK = True
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    _RERANK_OK = False
    _DEVICE = "cpu"


@dataclass
class RerankerResult:
    score: float
    label: str
    candidate_id: str
    raw_score: float
    candidate: dict[str, Any]


class PortfolioReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        if not _RERANK_OK:
            raise ImportError(
                "transformers/torch não encontrados.\n"
                "Execute: pip install transformers torch"
            )
        self.model_name = model_name
        self.device = _DEVICE
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.to(self.device)
        self._model.eval()

    def rerank(self, query: str, candidates: list[dict], top_k: int | None = None) -> list[dict]:
        if not query or not query.strip() or not candidates:
            return []

        pares = [(query, c.get("texto") or c.get("label") or "") for c in candidates]
        inputs = self._tokenizer(
            pares,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits

        logits = logits.view(-1).detach().cpu()
        scores = torch.sigmoid(logits).tolist()

        resultados: list[dict] = []
        for cand, score, raw_score in zip(candidates, scores, logits.tolist()):
            resultados.append({
                "candidate_id": cand.get("id", ""),
                "label": cand.get("label", ""),
                "score": float(score),
                "raw_score": float(raw_score),
                "candidate": cand,
            })

        resultados.sort(key=lambda x: -x["score"])
        if top_k and top_k > 0:
            resultados = resultados[:top_k]
        return resultados

