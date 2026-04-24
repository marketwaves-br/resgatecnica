#!/usr/bin/env python3
"""
pncp_similaridade.py — Análise por similaridade (zero custo, sem LLM)

Substitui pncp_agente.py quando o orçamento de API está indisponível.
Aplica os mesmos filtros estruturais (situação, prazo, tipoBeneficio) e
usa IndicePortfolio (TF-IDF ou semântico) para calcular score por licitação.

Saída: pncp_similaridades.json — usado por pncp_radar.py para gerar o mapa.

Uso:
  python pncp_similaridade.py                    # processa tudo
  python pncp_similaridade.py --backend tfidf    # força TF-IDF
  python pncp_similaridade.py --backend semantic # força semântico (GPU)
  python pncp_similaridade.py --threshold 0.15   # threshold mínimo de saída
  python pncp_similaridade.py --max 100          # limita número de licitações
"""
from __future__ import annotations

import argparse
import json
import sys
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path

from pncp_filtros import (
    calcular_score_bid,
    peso_contexto_licitacao,
    pre_filtrar_licitacao,
)
from pncp_config import carregar_config, obter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Configuração — defaults (sobrescritos por config.yaml se presente) ────────
# Para usar em IDE sem CLI, edite os valores aqui OU crie/edite config.yaml.

PORTFOLIO_FILE   = Path("prompts/portfolio_mestre_resgatecnica_lite_v2.json")
LICITACOES_FILE  = Path("pncp_licitacoes.json")
EDITAIS_DIR      = Path("editais")
OUTPUT_FILE      = Path("pncp_similaridades.json")

BACKEND_PADRAO          = "auto"         # auto | tfidf | semantic
THRESHOLD_MINIMO        = 0.10           # licitações abaixo disso são descartadas
THRESHOLD_AMBIGUO       = 0.20           # ambíguas precisam de evidência mais forte
DIAS_MINIMOS_PREPARO    = 7
SITUACOES_INVALIDAS     = {2, 3, 4}
TIPOS_BENEFICIO_ELEGIVEIS = {4, 5}       # exclui exclusivos ME/EPP (1, 3)
TOP_ITENS_POR_LICITACAO = 5              # itens de maior score por licitação no output
TOP_K_RETRIEVAL         = 10
TOP_K_DEBUG_JSON        = 5
RETRIEVAL_MODELO_PADRAO = "BAAI/bge-m3"
RERANKER_MODELO_PADRAO  = "BAAI/bge-reranker-v2-m3"

# Aplica config.yaml sobre os defaults acima
_CFG = carregar_config()
if _CFG:
    _p = _CFG.get("pipeline", {})
    BACKEND_PADRAO          = _p.get("backend",          BACKEND_PADRAO)
    THRESHOLD_MINIMO        = float(_p.get("threshold",          THRESHOLD_MINIMO))
    THRESHOLD_AMBIGUO       = float(_p.get("threshold_ambiguo",  THRESHOLD_AMBIGUO))
    DIAS_MINIMOS_PREPARO    = int(_p.get("dias_minimos_preparo",  DIAS_MINIMOS_PREPARO))
    TOP_K_RETRIEVAL         = int(_p.get("top_k_retrieval",       TOP_K_RETRIEVAL))
    TOP_K_DEBUG_JSON        = int(_p.get("top_k_debug_json",      TOP_K_DEBUG_JSON))
    RETRIEVAL_MODELO_PADRAO = _p.get("retrieval_model",   RETRIEVAL_MODELO_PADRAO)
    RERANKER_MODELO_PADRAO  = _p.get("reranker_model",    RERANKER_MODELO_PADRAO)

FUSO_BRASILIA = timezone(timedelta(hours=-3))

# Termos que indicam itens claramente fora do domínio da Resgatécnica.
# Se a descrição do item contém qualquer um desses padrões, o score é zerado.
import re as _re
_NEGATIVOS_ITEM = _re.compile(
    r'\b('
    # Informática / escritório
    r'multim[eé]dia|apontador|projetor|datashow'
    r'|impressora|toner|cartucho|refil|suprimento'
    r'|notebook|computador|desktop|tablet|servidor'
    r'|teclado|mouse\b|monitor\b|nobreak|estabilizador'
    r'|papel\b|resma|envelope|caneta\b|lápis|borracha|grampo|clips|pasta\b'
    # Mobiliário / utilidades domésticas
    r'|cadeira|mesa\b|armário|estante|arquivo\b|gaveteiro'
    r'|bebedouro|cafeteira|microondas|geladeira|refrigerador'
    r'|copo\b|prato\b|talher|guardanapo'
    # Limpeza / jardinagem
    r'|vassoura|rodo\b|balde\b|detergente|desinfetante|sabão|sabonete'
    r'|jardinagem|adubo|fertilizante|semente'
    # Climatização / utilidades prediais
    r'|ar\s*condicionado|split\b|btu(?:s)?\b|climatiza(?:ç|c)[aã]o'
    # Material didático
    r'|livro\b|apostila|material did[aá]tico|did[aá]tico'
    # Seguros e serviços técnicos genéricos
    r'|seguro\b|ap[oó]lice|franquia\b|frota\b'
    r'|pcmso|ltcat|aso\b|pgr\b|medicina do trabalho'
    r'|engenharia de seguran[çc]a'
    r'|stent\b|hipoclorito|condutiv[ií]metro'
    # Esportes / materiais esportivos
    r'|esportivo|esporte\b|futebol|futsal|society\b'
    r'|goleiro|árbitro|arbitro|apito\b'
    r'|vôlei|volei|voleibol|basquete|handebol|atletismo'
    r'|raquete|bola\b|chuteira|uniforme esportivo|tênis\b'
    r'|rede para gol|gol de futebol|trave\b|placar\b'
    r'|academia|ginástica|musculação|halter|anilha'
    r'|suplemento|whey|creatina|hipercal[oó]rico'
    r')',
    _re.IGNORECASE | _re.UNICODE,
)


# ── Utilidades ────────────────────────────────────────────────────────────────

def _agora_brasilia() -> datetime:
    return datetime.now(tz=FUSO_BRASILIA)


def _parse_dt(s: str) -> datetime | None:
    """Parseia dataEncerramentoProposta — idêntico ao pncp_agente.py."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s)[:19]).replace(tzinfo=FUSO_BRASILIA)
    except ValueError:
        return None  # data inválida → manter (sem descarte por precaução)


def _localizar_pasta(ctrl: str, editais_dir: Path) -> Path | None:
    """Encontra a pasta editais/ pelo numeroControlePNCP."""
    ctrl_slug = ctrl.replace("/", "_")
    candidatas = list(editais_dir.glob(f"*_{ctrl_slug}"))
    return candidatas[0] if candidatas else None


def _ler_itens(pasta: Path) -> list[dict]:
    path = pasta / "itens.json"
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get("itens", [])
    except Exception:
        return []


def _itens_elegiveis(itens: list[dict]) -> list[dict]:
    """
    Filtra itens por tipoBeneficio — lógica idêntica ao pncp_agente.py.
      tipoBeneficio 1 = ME/EPP exclusivo
      tipoBeneficio 3 = Cota reservada ME/EPP
      tipoBeneficio 4 = Sem benefício (ampla concorrência)   ← elegível
      tipoBeneficio 5 = Não se aplica                        ← elegível
    Se nenhum item for elegível, retorna lista vazia → licitação descartada como meepp.
    """
    return [i for i in itens if i.get("tipoBeneficio") in TIPOS_BENEFICIO_ELEGIVEIS]


def _calibrar_score_item(score_retrieval: float, reranqueados: list[dict]) -> tuple[float, float]:
    """
    Converte retrieval + rerank em score final conservador do item.

    Regras:
      - sem rerank: usa score de retrieval
      - com rerank: score final fica ancorado no retrieval
      - diferença top1-top2 aumenta confiança; empate derruba confiança
    """
    if not reranqueados:
        return score_retrieval, 0.0

    top1 = reranqueados[0]
    top2 = reranqueados[1] if len(reranqueados) > 1 else None
    raw1 = float(top1.get("raw_score", 0.0))
    raw2 = float(top2.get("raw_score", 0.0)) if top2 else 0.0
    gap = max(0.0, raw1 - raw2)

    # O score final fica ancorado no retrieval do candidato vencedor.
    score_base = float(top1.get("candidate", {}).get("score", score_retrieval))

    # Raw score negativo indica que o reranker não gostou do par; penaliza,
    # mas sem zerar automaticamente. Raw positivo preserva mais do retrieval.
    if raw1 <= 0:
        fator_raw = 0.55
    elif raw1 <= 2.0:
        fator_raw = 0.75
    else:
        fator_raw = 0.95

    # Gap entre top1 e top2 mede confiança relativa do rerank.
    fator_gap = 0.85 + 0.15 * min(1.0, gap / 2.0)
    score_final = score_base * fator_raw * fator_gap
    return min(score_final, 1.0), gap


def _threshold_ajustado(
    threshold_base: float,
    prefiltro,
    peso_contexto: float,
    objeto: str,
) -> float:
    """
    Ajusta threshold em casos fortemente positivos de ambulância/APH/remoção,
    sem afrouxar o restante do pipeline.
    """
    threshold = threshold_base
    obj = (objeto or "").lower()

    epi_generico = (
        ("equipamento" in obj or "equipamentos" in obj)
        and ("prote" in obj)
        and ("individual" in obj)
    )
    ancora_resgate = any(
        termo in obj
        for termo in (
            "resgate",
            "salvamento",
            "socorro",
            "aph",
            "ambul",
            "incênd",
            "incend",
            "bombeiro",
            "balíst",
            "balist",
            "tátic",
            "tatic",
            "desencarcer",
        )
    )

    # Compras genéricas de EPI sem âncora operacional continuam sendo um dos
    # principais falsos positivos residuais. Nesses casos, exigimos evidência
    # mais forte sem prejudicar ambulância/APH e outros objetos claramente
    # aderentes.
    if epi_generico and not ancora_resgate:
        threshold = max(threshold, 0.26)

    if prefiltro.decisao == "AMBIGUO":
        return max(threshold, THRESHOLD_AMBIGUO)
    fortes_objeto = (
        "ambul" in obj
        or "uti movel" in obj
        or "uti móvel" in obj
        or "furgoneta" in obj
        or "simples remo" in obj
        or "remoção" in obj
        or "remocao" in obj
    )
    fortes_prefiltro = any(
        t in {"ambulancia", "aph", "socorro", "uti movel"}
        for t in getattr(prefiltro, "positivos", [])
    )
    if peso_contexto >= 1.0 and (fortes_objeto or fortes_prefiltro):
        return min(threshold, 0.08)
    return threshold


# ── Pipeline principal ────────────────────────────────────────────────────────

def processar(args, portfolio_path: Path, editais_path: Path) -> None:
    from pncp_embeddings import IndicePortfolio
    from pncp_reranker import PortfolioReranker

    # 1. Carregar portfólio e construir índice
    print(f"Carregando portfólio de {portfolio_path}...")
    with open(portfolio_path, encoding="utf-8") as f:
        portfolio = json.load(f)

    print(f"Construindo índice [{args.backend}]...")
    indice = IndicePortfolio.carregar_ou_construir(
        portfolio_path,
        backend=args.backend,
        modelo=args.retrieval_model,
    )
    print(f"  Índice pronto: {indice.n_produtos} produtos | backend={indice.backend}")

    reranker = None
    reranker_status = "disabled"
    reranker_error = ""
    try:
        reranker = PortfolioReranker(model_name=args.reranker_model)
        reranker_status = "active"
        print(f"  Reranker pronto: {args.reranker_model}")
    except Exception as e:
        reranker_status = "fallback"
        reranker_error = str(e)
        print(f"  [aviso] Reranker indisponível — usando ranking bruto do retrieval: {e}")

    # 2. Carregar licitações
    print(f"\nCarregando {LICITACOES_FILE}...")
    with open(LICITACOES_FILE, encoding="utf-8") as f:
        dados = json.load(f)
    licitacoes = dados.get("licitacoes", dados) if isinstance(dados, dict) else dados
    print(f"  {len(licitacoes):,} licitações encontradas.")

    agora = _agora_brasilia()
    limite_dt = agora + timedelta(days=DIAS_MINIMOS_PREPARO)

    resultados: list[dict] = []
    n_sem_pasta = 0
    n_prazo = 0
    n_situacao = 0
    n_meepp = 0
    n_prefiltro = 0
    n_score_baixo = 0
    n_processadas = 0
    max_proc = args.max or len(licitacoes)

    print(f"\nProcessando licitações (threshold={args.threshold:.2f})...\n")

    for lic in licitacoes:
        if n_processadas >= max_proc:
            break

        ctrl = lic.get("numeroControlePNCP", "")
        unidade = lic.get("unidadeOrgao", {})
        uf       = unidade.get("ufSigla", "")
        municipio = unidade.get("municipioNome", "")
        ibge      = unidade.get("codigoIbge", "")
        orgao     = lic.get("orgaoEntidade", {}).get("razaoSocial", "")
        objeto    = lic.get("objetoCompra", "")
        valor     = lic.get("valorTotalEstimado") or 0.0
        link      = lic.get("linkSistemaOrigem", "")
        data_enc  = lic.get("dataEncerramentoProposta", "")
        situacao  = lic.get("situacaoCompraId")

        # Filtro 1 — situação inválida
        if situacao in SITUACOES_INVALIDAS:
            n_situacao += 1
            continue

        # Filtro 2 — prazo mínimo
        dt_enc = _parse_dt(data_enc)
        if dt_enc and dt_enc < limite_dt:
            n_prazo += 1
            continue

        # Filtro 3 — pasta com itens
        pasta = _localizar_pasta(ctrl, editais_path)
        if pasta is None:
            n_sem_pasta += 1
            continue

        # Filtro 4 — ler e filtrar itens por tipoBeneficio
        itens = _ler_itens(pasta)
        if not itens:
            n_sem_pasta += 1
            continue

        elegiveis = _itens_elegiveis(itens)
        if not elegiveis:
            n_meepp += 1
            continue

        # Filtro 5 — pré-filtro contextual do agente
        prefiltro = pre_filtrar_licitacao(objeto, elegiveis)
        if prefiltro.decisao == "DESCARTAR":
            n_prefiltro += 1
            continue

        n_processadas += 1

        # Calcular scores
        scored: list[dict] = []
        scores_itens: list[float] = []
        lic_fallback_rerank = False
        for item in elegiveis:
            desc = (item.get("descricao") or "").strip()
            if not desc:
                continue
            # Filtro negativo: itens claramente fora do domínio da Resgatécnica
            if _NEGATIVOS_ITEM.search(desc):
                scores_itens.append(0.0)
                continue

            recuperados = indice.top_k_candidatos(desc, k=args.top_k_retrieval)
            if not recuperados:
                scores_itens.append(0.0)
                continue

            score_retrieval = float(recuperados[0]["score"])
            produto_ref = recuperados[0]["label"]
            score_item = score_retrieval
            score_rerank_top1 = score_retrieval
            score_rerank_gap = 0.0
            produto_ref_rerank = produto_ref
            reranqueados: list[dict] = []

            if reranker is not None:
                try:
                    reranqueados = reranker.rerank(desc, recuperados, top_k=args.top_k_debug_json)
                except Exception:
                    reranqueados = []
                    lic_fallback_rerank = True

            if reranqueados:
                score_rerank_top1 = float(reranqueados[0]["score"])
                score_item, score_rerank_gap = _calibrar_score_item(score_retrieval, reranqueados)
                produto_ref_rerank = reranqueados[0]["label"]
            else:
                lic_fallback_rerank = lic_fallback_rerank or (reranker is not None)

            scores_itens.append(score_item)
            scored.append({
                "score":       score_item,
                "score_retrieval": score_retrieval,
                "score_rerank_top1": score_rerank_top1,
                "score_rerank_gap": score_rerank_gap,
                "descricao":   desc[:120],
                "produto_ref": produto_ref_rerank,
                "produto_ref_rerank": produto_ref_rerank,
                "numeroItem":  item.get("numeroItem"),
                "valor":       item.get("valorTotal") or 0.0,
                "top_candidatos_recuperados": [
                    {
                        "id": c["id"],
                        "label": c["label"],
                        "score": round(float(c["score"]), 4),
                    } for c in recuperados[:args.top_k_debug_json]
                ],
                "top_candidatos_reranqueados": [
                    {
                        "candidate_id": c["candidate_id"],
                        "label": c["label"],
                        "score": round(float(c["score"]), 4),
                        "raw_score": round(float(c["raw_score"]), 4),
                    } for c in reranqueados[:args.top_k_debug_json]
                ],
            })

        if not scored:
            n_score_baixo += 1
            continue

        scored.sort(key=lambda x: -x["score"])
        score_max   = scored[0]["score"]
        score_medio = sum(s["score"] for s in scored) / len(scored)
        score_retrieval_max = max(s.get("score_retrieval", 0.0) for s in scored)
        peso_contexto = peso_contexto_licitacao(objeto)
        score_bid_info = calcular_score_bid(
            scores_itens=scores_itens,
            n_elegiveis=len(elegiveis),
            peso_contexto=peso_contexto,
        )
        score_bid = score_bid_info["score_bid"]

        threshold_aplicado = _threshold_ajustado(
            args.threshold,
            prefiltro,
            peso_contexto,
            objeto,
        )

        if score_bid < threshold_aplicado:
            n_score_baixo += 1
            continue

        resultados.append({
            "processo":         ctrl,
            "uf":               uf,
            "municipio":        municipio,
            "codigoIbge":       ibge,
            "orgao":            orgao,
            "objeto":           objeto[:200],
            "valor_estimado":   round(valor, 2),
            "data_encerramento": data_enc[:10] if data_enc else "",
            "link":             link,
            "score_max":        round(score_retrieval_max, 4),
            "score_medio":      round(score_medio, 4),
            "score_bid":        round(score_bid, 4),
            "score_top1":       round(score_bid_info["score_top1"], 4),
            "score_top3_medio": round(score_bid_info["score_top3_medio"], 4),
            "score_rerank_top1": round(scored[0].get("score_rerank_top1", 0.0), 4),
            "produto_ref_rerank": scored[0].get("produto_ref_rerank", ""),
            "densidade_relevancia": round(score_bid_info["densidade_relevancia"], 4),
            "n_itens_relevantes": score_bid_info["n_itens_relevantes"],
            "penalizacao_variancia": score_bid_info["penalizacao_variancia"],
            "peso_contexto":    round(score_bid_info["peso_contexto"], 4),
            "prefiltro_decisao": prefiltro.decisao,
            "prefiltro_score":  round(prefiltro.score, 4),
            "prefiltro_positivos": prefiltro.positivos[:10],
            "prefiltro_negativos": prefiltro.negativos[:10],
            "prefiltro_ncm_hit": prefiltro.ncm_hit,
            "threshold_aplicado": round(threshold_aplicado, 4),
            "fallback_rerank": lic_fallback_rerank or reranker_status != "active",
            "n_itens_elegiveis": len(elegiveis),
            "top_itens":        scored[:TOP_ITENS_POR_LICITACAO],
            "top_candidatos_recuperados": scored[0].get("top_candidatos_recuperados", []),
            "top_candidatos_reranqueados": scored[0].get("top_candidatos_reranqueados", []),
            "backend":          indice.backend,
            "modelo":           indice.modelo,
        })

        if n_processadas % 50 == 0:
            print(
                f"  [{n_processadas:4d}] score_bid={score_bid:.3f} "
                f"(max={score_max:.3f}) {municipio} — {objeto[:50]}"
            )

    # 3. Ordenar por score descendente
    resultados.sort(key=lambda x: (-x["score_bid"], -x["score_max"]))

    # 4. Salvar
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "_meta": {
                "gerado_em":    agora.isoformat(),
                "backend":      indice.backend,
                "modelo":       indice.modelo,
                "retrieval_backend": indice.backend,
                "retrieval_model": indice.modelo,
                "reranker_model": args.reranker_model,
                "reranker_status": reranker_status,
                "reranker_error": reranker_error,
                "top_k_retrieval": args.top_k_retrieval,
                "top_k_debug_json": args.top_k_debug_json,
                "threshold":    args.threshold,
                "metrica_principal": "score_bid",
                "total":        len(resultados),
                "n_processadas": n_processadas,
            },
            "licitacoes": resultados,
        }, f, ensure_ascii=False, indent=2)

    # 5. Resumo
    score_bid_medio = (sum(r['score_bid'] for r in resultados) / len(resultados)) if resultados else 0.0
    score_bid_max = max((r['score_bid'] for r in resultados), default=0.0)
    score_max_medio = (sum(r['score_max'] for r in resultados) / len(resultados)) if resultados else 0.0
    print(f"""
{'='*60}
RESUMO
{'='*60}
Licitações no arquivo       : {len(licitacoes):>6,}
  Filtradas (situação)      : {n_situacao:>6,}
  Filtradas (prazo)         : {n_prazo:>6,}
  Sem pasta/itens           : {n_sem_pasta:>6,}
  ME/EPP exclusivo          : {n_meepp:>6,}
  Pré-filtro contextual     : {n_prefiltro:>6,}
  Score_bid abaixo threshold: {n_score_baixo:>6,}
  APROVADAS (score_bid >= {args.threshold:.2f})  : {len(resultados):>6,}

Score_bid médio aprovadas   : {score_bid_medio:.3f}  (se > 0)
Score_bid máximo encontrado : {score_bid_max:.3f}
Score_max médio aprovado    : {score_max_medio:.3f}  (se > 0)

Saída salva em: {OUTPUT_FILE}
{'='*60}
""")


def main() -> None:
    p = argparse.ArgumentParser(description="Análise por similaridade — zero custo LLM")
    p.add_argument("--config",     default=None,
                   help="Arquivo de configuração YAML/JSON (default: config.yaml se existir)")
    p.add_argument("--backend",    default=BACKEND_PADRAO, choices=["auto", "tfidf", "semantic"])
    p.add_argument("--threshold",  type=float, default=THRESHOLD_MINIMO,
                   help=f"Score mínimo para incluir no output (default: {THRESHOLD_MINIMO})")
    p.add_argument("--max",        type=int, default=0,
                   help="Limitar número de licitações processadas (0 = todas)")
    p.add_argument("--retrieval-model", default=RETRIEVAL_MODELO_PADRAO,
                   help="Modelo HuggingFace para retrieval semântico")
    p.add_argument("--reranker-model", default=RERANKER_MODELO_PADRAO,
                   help="Modelo HuggingFace para reranking item × produto")
    p.add_argument("--top-k-retrieval", type=int, default=TOP_K_RETRIEVAL,
                   help="Quantidade de candidatos recuperados por item")
    p.add_argument("--top-k-debug-json", type=int, default=TOP_K_DEBUG_JSON,
                   help="Quantidade de candidatos gravados no JSON para debug")
    p.add_argument("--portfolio",  default=str(PORTFOLIO_FILE))
    p.add_argument("--editais",    default=str(EDITAIS_DIR))

    # Suporte a IDE: permite chamar main() com lista de args explícita
    # Exemplo Jupyter/Spyder: main(["--threshold", "0.12", "--max", "50"])
    args = p.parse_args()

    # Config explícito via --config sobrescreve o carregado no módulo
    if args.config:
        from pncp_config import carregar_config as _cc
        _extra = _cc(args.config).get("pipeline", {})
        if _extra.get("backend")         and args.backend    == BACKEND_PADRAO:   args.backend    = _extra["backend"]
        if _extra.get("threshold")       and args.threshold  == THRESHOLD_MINIMO: args.threshold  = float(_extra["threshold"])
        if _extra.get("retrieval_model") and args.retrieval_model == RETRIEVAL_MODELO_PADRAO: args.retrieval_model = _extra["retrieval_model"]
        if _extra.get("reranker_model")  and args.reranker_model  == RERANKER_MODELO_PADRAO:  args.reranker_model  = _extra["reranker_model"]

    portfolio_path = Path(args.portfolio)
    editais_path   = Path(args.editais)

    if not portfolio_path.exists():
        print(f"[ERRO] Portfólio não encontrado: {portfolio_path}")
        sys.exit(1)
    if not LICITACOES_FILE.exists():
        print(f"[ERRO] {LICITACOES_FILE} não encontrado. Execute pncp_scanner.py primeiro.")
        sys.exit(1)

    processar(args, portfolio_path, editais_path)


if __name__ == "__main__":
    main()
