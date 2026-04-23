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
from datetime import datetime, timezone, timedelta
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Configuração ──────────────────────────────────────────────────────────────

PORTFOLIO_FILE   = Path("prompts/portfolio_mestre_resgatecnica_lite_v2.json")
LICITACOES_FILE  = Path("pncp_licitacoes.json")
EDITAIS_DIR      = Path("editais")
OUTPUT_FILE      = Path("pncp_similaridades.json")

BACKEND_PADRAO          = "auto"         # auto | tfidf | semantic
THRESHOLD_MINIMO        = 0.10           # licitações abaixo disso são descartadas
DIAS_MINIMOS_PREPARO    = 7
SITUACOES_INVALIDAS     = {2, 3, 4}
TIPOS_BENEFICIO_ELEGIVEIS = {4, 5}       # exclui exclusivos ME/EPP (1, 3)
TOP_ITENS_POR_LICITACAO = 5              # itens de maior score por licitação no output

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
    # Material didático
    r'|livro\b|apostila|material did[aá]tico|did[aá]tico'
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


# ── Pipeline principal ────────────────────────────────────────────────────────

def processar(args, portfolio_path: Path, editais_path: Path) -> None:
    from pncp_embeddings import IndicePortfolio

    # 1. Carregar portfólio e construir índice
    print(f"Carregando portfólio de {portfolio_path}...")
    with open(portfolio_path, encoding="utf-8") as f:
        portfolio = json.load(f)

    print(f"Construindo índice [{args.backend}]...")
    indice = IndicePortfolio.carregar_ou_construir(
        portfolio_path,
        backend=args.backend,
    )
    print(f"  Índice pronto: {indice.n_produtos} produtos | backend={indice.backend}")

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

        n_processadas += 1

        # Calcular scores
        scored: list[dict] = []
        for item in elegiveis:
            desc = (item.get("descricao") or "").strip()
            if not desc:
                continue
            # Filtro negativo: itens claramente fora do domínio da Resgatécnica
            if _NEGATIVOS_ITEM.search(desc):
                continue
            sc = indice.score_maximo(desc)
            top = indice.top_k(desc, k=1)
            produto_ref = top[0][1] if top else ""
            scored.append({
                "score":       sc,
                "descricao":   desc[:120],
                "produto_ref": produto_ref,
                "numeroItem":  item.get("numeroItem"),
                "valor":       item.get("valorTotal") or 0.0,
            })

        if not scored:
            n_score_baixo += 1
            continue

        scored.sort(key=lambda x: -x["score"])
        score_max   = scored[0]["score"]
        score_medio = sum(s["score"] for s in scored) / len(scored)

        if score_max < args.threshold:
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
            "score_max":        round(score_max, 4),
            "score_medio":      round(score_medio, 4),
            "n_itens_elegiveis": len(elegiveis),
            "top_itens":        scored[:TOP_ITENS_POR_LICITACAO],
            "backend":          indice.backend,
            "modelo":           indice.modelo,
        })

        if n_processadas % 50 == 0:
            print(f"  [{n_processadas:4d}] score_max={score_max:.3f}  {municipio} — {objeto[:50]}")

    # 3. Ordenar por score descendente
    resultados.sort(key=lambda x: -x["score_max"])

    # 4. Salvar
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "_meta": {
                "gerado_em":    agora.isoformat(),
                "backend":      indice.backend,
                "modelo":       indice.modelo,
                "threshold":    args.threshold,
                "total":        len(resultados),
                "n_processadas": n_processadas,
            },
            "licitacoes": resultados,
        }, f, ensure_ascii=False, indent=2)

    # 5. Resumo
    print(f"""
{'='*60}
RESUMO
{'='*60}
Licitações no arquivo       : {len(licitacoes):>6,}
  Filtradas (situação)      : {n_situacao:>6,}
  Filtradas (prazo)         : {n_prazo:>6,}
  Sem pasta/itens           : {n_sem_pasta:>6,}
  ME/EPP exclusivo          : {n_meepp:>6,}
  Score abaixo do threshold : {n_score_baixo:>6,}
  APROVADAS (score >= {args.threshold:.2f})  : {len(resultados):>6,}

Score médio aprovadas       : {sum(r['score_max'] for r in resultados)/len(resultados):.3f}  (se > 0)
Score máximo encontrado     : {max((r['score_max'] for r in resultados), default=0):.3f}

Saída salva em: {OUTPUT_FILE}
{'='*60}
""")


def main() -> None:
    p = argparse.ArgumentParser(description="Análise por similaridade — zero custo LLM")
    p.add_argument("--backend",    default=BACKEND_PADRAO, choices=["auto", "tfidf", "semantic"])
    p.add_argument("--threshold",  type=float, default=THRESHOLD_MINIMO,
                   help="Score mínimo para incluir no output (default: 0.10)")
    p.add_argument("--max",        type=int, default=0,
                   help="Limitar número de licitações processadas (0 = todas)")
    p.add_argument("--portfolio",  default=str(PORTFOLIO_FILE))
    p.add_argument("--editais",    default=str(EDITAIS_DIR))
    args = p.parse_args()

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
