#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from pncp_embeddings import IndicePortfolio
from pncp_reranker import PortfolioReranker


EDITAIS_DIR = Path("editais")
PORTFOLIO_FILE = Path("prompts/portfolio_mestre_resgatecnica_lite_v2.json")
DEFAULT_OUTPUT = Path("benchmark_licitacoes.json")


PATTERNS: dict[str, str] = {
    "ambulancia": r"ambul[aâ]ncia|ambulancia",
    "uti_movel": r"uti m[oó]vel|uti movel",
    "simples_remocao": r"simples remo[cç][aã]o|simples remocao",
    "furgoneta": r"furgoneta|furg[aã]o|furgao",
    "minivan": r"\bminivan\b",
    "viatura": r"\bviatura\b",
    "transporte_sanitario": r"transporte sanit[aá]rio|transporte sanitario",
    "remocao_terrestre": r"remo[cç][aã]o terrestre|remocao terrestre",
}

EXCLUDE_PATTERN = re.compile(
    r"\b("
    r"carrinh(?:o|os)|brinquedo|boneca|cartela de adesivos|pecas de montar|pecas para montar"
    r"|galerinha|divertir|certificacao pelo inmetro|caminhao de sorvete"
    r")\b",
    flags=re.IGNORECASE,
)

CONTEXT_EXCLUDE_PATTERNS: dict[str, str] = {
    "ambulancia": r"caso n[aã]o tenha ambul[aâ]ncia|posto m[eé]dico|decora[cç][aã]o|ornamenta[cç][aã]o|evento|festa|camarim|tenda 3x3",
    "minivan": r"hemodi[aá]lise|tratamento de hemodi[aá]lise",
}


def _matches(text: str, selected_labels: set[str]) -> set[str]:
    hits: set[str] = set()
    for label in selected_labels:
        pattern = PATTERNS[label]
        if re.search(pattern, text, flags=re.IGNORECASE):
            exclude = CONTEXT_EXCLUDE_PATTERNS.get(label)
            if exclude and re.search(exclude, text, flags=re.IGNORECASE):
                continue
            hits.add(label)
    return hits


def _scenario_type(text: str) -> str:
    text = text.lower()
    # Aquisição veicular explícita tem precedência quando "veículo tipo" aparece no início
    # da descrição (título), não em qualquer posição — evita falso positivo em
    # "locação de ambulância... veículo tipo D" onde "locação + ambulância" é APH.
    if re.match(r"\s*ve[ií]culo\s+tipo\b", text) or re.search(r"\baquisic[aã]o\s+de\s+ve[ií]culo", text):
        return "aquisicao_veicular"
    if re.search(r"\b(servi[cç]o|contrata[cç][aã]o|loca[cç][aã]o|presta[cç][aã]o)\b", text):
        if re.search(r"\b(uti m[oó]vel|uti movel|ambul[aâ]ncia|ambulancia|remo[cç][aã]o|remocao|transporte de pacientes)\b", text):
            return "servico_aph"
        return "servico_generico"
    if re.search(r"\b(ambul[aâ]ncia|ambulancia|minivan|viatura|pick[\s-]?up|furg[aã]o|furgao|furgoneta)\b", text):
        return "aquisicao_veicular"
    return "outro"


def _candidate_domain(label: str) -> str:
    label_norm = label.lower()
    if "atendimento pré-hospitalar" in label_norm or "atendimento pr" in label_norm:
        return "aph"
    if "veículos especiais customizados" in label_norm or "veiculos especiais customizados" in label_norm:
        return "veicular_customizado"
    if "resgate veicular" in label_norm:
        return "resgate_veicular"
    return "outro"


def _iter_matching_items(editais_dir: Path, selected_labels: set[str], max_cases: int | None) -> list[dict]:
    cases: list[dict] = []
    for itens_path in sorted(editais_dir.glob("*/itens.json")):
        try:
            data = json.loads(itens_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        itens = data if isinstance(data, list) else data.get("itens", [])
        for item in itens:
            desc = str(item.get("descricao") or "").strip()
            if not desc:
                continue
            if EXCLUDE_PATTERN.search(desc):
                continue
            hits = _matches(desc, selected_labels)
            if not hits:
                continue
            cases.append(
                {
                    "processo_dir": itens_path.parent.name,
                    "itens_path": str(itens_path),
                    "numeroItem": item.get("numeroItem"),
                    "descricao": desc,
                    "hits": sorted(hits),
                    "scenario_type": _scenario_type(desc),
                }
            )
            if max_cases and len(cases) >= max_cases:
                return cases
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local de retrieval/rerank em editais/*/itens.json")
    parser.add_argument("--patterns", nargs="+", choices=sorted(PATTERNS), default=["ambulancia", "uti_movel", "minivan"])
    parser.add_argument("--max-cases", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--backend", choices=["semantic", "tfidf", "auto"], default="semantic")
    parser.add_argument("--retrieval-model", default="BAAI/bge-m3")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    selected = set(args.patterns)
    cases = _iter_matching_items(EDITAIS_DIR, selected, args.max_cases)

    indice = IndicePortfolio.carregar_ou_construir(
        PORTFOLIO_FILE, backend=args.backend, modelo=args.retrieval_model
    )
    reranker = None if args.no_rerank else PortfolioReranker(model_name=args.reranker_model)

    results: list[dict] = []
    hit_counts = {label: 0 for label in sorted(selected)}
    scenario_counts: dict[str, int] = {}
    top1_domain_counts: dict[str, int] = {}
    mismatch_counts: dict[str, int] = {}
    for case in cases:
        for hit in case["hits"]:
            hit_counts[hit] = hit_counts.get(hit, 0) + 1
        scenario = case["scenario_type"]
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        recuperados = indice.top_k_candidatos(case["descricao"], k=args.top_k)
        reranqueados = reranker.rerank(case["descricao"], recuperados, top_k=args.top_k) if (reranker and recuperados) else []
        top1_label = ""
        if reranqueados:
            top1_label = reranqueados[0]["label"]
        elif recuperados:
            top1_label = recuperados[0]["label"]
        top1_domain = _candidate_domain(top1_label) if top1_label else "sem_resultado"
        top1_domain_counts[top1_domain] = top1_domain_counts.get(top1_domain, 0) + 1
        expected_domain = ""
        if scenario == "servico_aph":
            expected_domain = "aph"
        elif scenario == "aquisicao_veicular":
            expected_domain = "veicular_customizado"
        if expected_domain and top1_domain != expected_domain:
            key = f"{scenario}->{top1_domain}"
            mismatch_counts[key] = mismatch_counts.get(key, 0) + 1
        results.append(
            {
                **case,
                "top1_domain": top1_domain,
                "expected_domain": expected_domain,
                "domain_match": (not expected_domain) or (top1_domain == expected_domain),
                "top_recuperados": [
                    {"label": c["label"], "score": round(float(c["score"]), 4)} for c in recuperados
                ],
                "top_reranqueados": [
                    {
                        "label": c["label"],
                        "score": round(float(c["score"]), 4),
                        "raw_score": round(float(c["raw_score"]), 4),
                    }
                    for c in reranqueados
                ],
            }
        )

    summary = {
        "patterns": sorted(selected),
        "cases": len(results),
        "hit_counts": hit_counts,
        "scenario_counts": scenario_counts,
        "top1_domain_counts": top1_domain_counts,
        "mismatch_counts": mismatch_counts,
        "backend": indice.backend,
        "retrieval_model": indice.modelo,
        "reranker_model": None if args.no_rerank else args.reranker_model,
    }
    payload = {"summary": summary, "results": results}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saída salva em: {args.output}")


if __name__ == "__main__":
    main()
