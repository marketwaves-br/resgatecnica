#!/usr/bin/env python3
"""
resumo_resultados.py — Relatório compacto dos JSONs gerados pelo pipeline.

Uso (sem dependências ML — roda em segundos):
  python resumo_resultados.py                    # mostra tudo
  python resumo_resultados.py similaridades      # só pncp_similaridades.json
  python resumo_resultados.py benchmark          # só benchmark_licitacoes.json
  python resumo_resultados.py --top 20           # top-N aprovadas (default: 15)
  python resumo_resultados.py --mismatches       # detalha mismatches do benchmark
  python resumo_resultados.py --itens            # mostra top itens de cada aprovada

Cole o output deste script no chat para análise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SIM_FILE  = Path("pncp_similaridades.json")
BENCH_FILE = Path("benchmark_licitacoes.json")

SEP  = "=" * 68
SEP2 = "-" * 68


# ── Similaridades ──────────────────────────────────────────────────────────────

def relatorio_similaridades(top_n: int = 15, mostrar_itens: bool = False) -> None:
    if not SIM_FILE.exists():
        print(f"[!] {SIM_FILE} não encontrado. Execute pncp_similaridade.py primeiro.")
        return

    data   = json.loads(SIM_FILE.read_text(encoding="utf-8"))
    meta   = data.get("_meta", {})
    lics   = data.get("licitacoes", [])

    print(SEP)
    print("SIMILARIDADES — pncp_similaridades.json")
    print(SEP)
    print(f"  Gerado em       : {meta.get('gerado_em','?')}")
    print(f"  Backend         : {meta.get('backend','?')} | modelo: {meta.get('retrieval_model','?')}")
    print(f"  Reranker        : {meta.get('reranker_model','?')} [{meta.get('reranker_status','?')}]")
    print(f"  Processadas     : {meta.get('n_processadas','?')} licitações")
    print(f"  APROVADAS       : {meta.get('total','?')} (score_bid >= {meta.get('threshold','?')})")
    print()

    if not lics:
        print("  Nenhuma licitação aprovada.")
        return

    scores_bid  = [l["score_bid"]  for l in lics]
    scores_max  = [l.get("score_max", 0) for l in lics]
    scores_rer  = [l.get("score_rerank_top1", 0) for l in lics]
    valores     = [l.get("valor_estimado", 0) for l in lics]
    n_penal     = sum(1 for l in lics if l.get("penalizacao_variancia"))
    n_ambiguo   = sum(1 for l in lics if l.get("prefiltro_decisao") == "AMBIGUO")
    n_fallback  = sum(1 for l in lics if l.get("fallback_rerank"))

    print("MÉTRICAS GERAIS:")
    print(f"  score_bid  : min={min(scores_bid):.3f}  med={_med(scores_bid):.3f}  max={max(scores_bid):.3f}")
    print(f"  score_max  : min={min(scores_max):.3f}  med={_med(scores_max):.3f}  max={max(scores_max):.3f}")
    print(f"  rerank_top1: min={min(scores_rer):.3f}  med={_med(scores_rer):.3f}  max={max(scores_rer):.3f}")
    print(f"  valor_est  : min=R${min(valores):,.0f}  max=R${max(valores):,.0f}  total=R${sum(valores):,.0f}")
    print(f"  Penaliz. variância : {n_penal}")
    print(f"  Pré-filtro AMBIGUO : {n_ambiguo}")
    print(f"  Fallback rerank    : {n_fallback}")
    print()

    # Distribuição de peso_contexto
    ctx = {}
    for l in lics:
        k = f"{l.get('peso_contexto', 0):.2f}"
        ctx[k] = ctx.get(k, 0) + 1
    print(f"  peso_contexto dist : {dict(sorted(ctx.items(), reverse=True))}")
    print()

    print(SEP2)
    print(f"TOP {top_n} APROVADAS (ordenadas por score_bid):")
    print(SEP2)

    for i, l in enumerate(lics[:top_n], 1):
        sbid  = l["score_bid"]
        smax  = l.get("score_max", 0)
        srer  = l.get("score_rerank_top1", 0)
        penal = " ⚠VAR" if l.get("penalizacao_variancia") else ""
        ctx_v = l.get("peso_contexto", 0)
        dens  = l.get("densidade_relevancia", 0)
        pf    = l.get("prefiltro_decisao", "?")
        prod  = l.get("produto_ref_rerank", "")[:55]
        valor = l.get("valor_estimado", 0)
        enc   = l.get("data_encerramento", "")

        print(f"\n  #{i:02d} [{sbid:.3f}] {l['municipio']} — {l['uf']}")
        print(f"       {l['objeto'][:75]}")
        print(f"       score_max={smax:.3f} rerank={srer:.3f} ctx={ctx_v:.2f} dens={dens:.2f} pf={pf}{penal}")
        print(f"       prod_ref : {prod}")
        print(f"       valor    : R${valor:,.0f}  |  enc: {enc}  |  {l.get('link','')[:60]}")

        if mostrar_itens:
            top_itens = l.get("top_itens", [])[:3]
            if top_itens:
                print(f"       itens top:")
                for it in top_itens:
                    print(f"         [{it['score']:.3f}] {it['descricao'][:60]}")

    if len(lics) > top_n:
        print(f"\n  ... e mais {len(lics) - top_n} aprovadas (use --top N para ver mais)")
    print()


# ── Benchmark ─────────────────────────────────────────────────────────────────

def relatorio_benchmark(mostrar_mismatches: bool = True) -> None:
    if not BENCH_FILE.exists():
        print(f"[!] {BENCH_FILE} não encontrado. Execute benchmark_licitacoes.py primeiro.")
        return

    raw     = json.loads(BENCH_FILE.read_text(encoding="utf-8"))
    # benchmark_licitacoes.py grava {"summary": {...}, "results": [...]}
    data    = raw.get("summary", raw)   # compatível com ambos os formatos
    results = raw.get("results", [])

    print(SEP)
    print("BENCHMARK — benchmark_licitacoes.json")
    print(SEP)
    print(f"  Backend         : {data.get('backend','?')}")
    print(f"  Retrieval model : {data.get('retrieval_model','?')}")
    print(f"  Reranker model  : {data.get('reranker_model','?')}")
    print(f"  Total de casos  : {data.get('cases',0)}")
    print()

    print("HIT COUNTS (padrões encontrados nos editais):")
    for k, v in sorted(data.get("hit_counts", {}).items()):
        print(f"  {k:<25}: {v}")
    print()

    print("SCENARIO COUNTS:")
    for k, v in sorted(data.get("scenario_counts", {}).items()):
        print(f"  {k:<25}: {v}")
    print()

    print("TOP1 DOMAIN COUNTS:")
    for k, v in sorted(data.get("top1_domain_counts", {}).items()):
        print(f"  {k:<25}: {v}")
    print()

    mismatches = data.get("mismatch_counts", {})
    total_mis  = sum(mismatches.values())
    total_exp  = sum(
        1 for r in results
        if r.get("expected_domain") and r.get("domain_match") is not None
    )
    acertos    = sum(
        1 for r in results
        if r.get("expected_domain") and r.get("domain_match") is True
    )

    print("MISMATCHES:")
    if not mismatches:
        print("  ✅ Nenhum mismatch de domínio!")
    else:
        for k, v in mismatches.items():
            print(f"  ⚠  {k}: {v}")
    print()
    if total_exp:
        pct = 100.0 * acertos / total_exp
        print(f"  Acurácia de domínio: {acertos}/{total_exp} = {pct:.1f}%")
    print()

    if mostrar_mismatches and mismatches:
        print(SEP2)
        print("DETALHE DOS MISMATCHES:")
        print(SEP2)
        for r in results:
            if r.get("expected_domain") and not r.get("domain_match"):
                rerank  = r.get("top_reranqueados", [])
                recuper = r.get("top_recuperados",  [])
                top1_r  = rerank[0]  if rerank  else None
                top1_rec = recuper[0] if recuper else None
                print(f"\n  scenario : {r['scenario_type']} → expected: {r['expected_domain']} | got: {r['top1_domain']}")
                print(f"  hits     : {r['hits']}")
                print(f"  desc     : {r['descricao'][:100]}")
                if top1_r:
                    print(f"  rerank#1 : [{top1_r['score']:.4f} raw={top1_r['raw_score']:.2f}] {top1_r['label'][:70]}")
                elif top1_rec:
                    print(f"  retrieval#1: [{top1_rec['score']:.4f}] {top1_rec['label'][:70]}")
        print()


# ── Utilitários ───────────────────────────────────────────────────────────────

def _med(lst: list[float]) -> float:
    s = sorted(lst)
    return s[len(s) // 2] if s else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Relatório compacto dos JSONs do pipeline (sem ML, roda em segundos)"
    )
    p.add_argument("modo", nargs="?", default="tudo",
                   choices=["tudo", "similaridades", "benchmark"],
                   help="O que mostrar (default: tudo)")
    p.add_argument("--top",        type=int, default=15,
                   help="Número de aprovadas a listar (default: 15)")
    p.add_argument("--itens",      action="store_true",
                   help="Mostrar top itens de cada aprovada")
    p.add_argument("--mismatches", action="store_true", default=True,
                   help="Detalhar mismatches do benchmark (default: ativo)")
    p.add_argument("--no-mismatches", dest="mismatches", action="store_false",
                   help="Suprimir detalhe de mismatches")
    args = p.parse_args()

    if args.modo in ("tudo", "similaridades"):
        relatorio_similaridades(top_n=args.top, mostrar_itens=args.itens)

    if args.modo in ("tudo", "benchmark"):
        relatorio_benchmark(mostrar_mismatches=args.mismatches)


if __name__ == "__main__":
    main()
