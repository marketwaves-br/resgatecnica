#!/usr/bin/env python3
"""
pncp_powerbi.py — Exportador de dados para Power BI / Excel

Fontes (todas opcionais — o script usa o que encontrar):
  editais/*/aderencia.json   → classificação por licitação e por item
  pncp_licitacoes.json       → objeto, órgão, município, valor estimado
  pncp_telemetria.jsonl      → custo e tokens por chamada LLM

Saídas:
  powerbi_licitacoes.csv     → 1 linha por licitação processada
  powerbi_itens.csv          → 1 linha por item analisado
  powerbi_resumo.txt         → estatísticas da exportação

Uso:
  python pncp_powerbi.py
  python pncp_powerbi.py --editais outro_dir --out resultados/
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Garante UTF-8 no console Windows (Python 3.7+)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Configuração ──────────────────────────────────────────────────────────────
EDITAIS_DIR     = Path("editais")
LICITACOES_FILE = Path("pncp_licitacoes.json")
TELEMETRIA_FILE = Path("pncp_telemetria.jsonl")
OUT_DIR         = Path(".")
OUT_LICITACOES  = "powerbi_licitacoes.csv"
OUT_ITENS       = "powerbi_itens.csv"
OUT_RESUMO      = "powerbi_resumo.txt"

# UTF-8 com BOM: Excel PT-BR reconhece acentos automaticamente
ENCODING = "utf-8-sig"

# Mapeamento de prioridade para número (para ordenação)
_PRIORIDADE_ORD = {"ALTA": 1, "MEDIA": 2, "?": 3, "DESCARTAR": 4}

# ── Helpers de extração ───────────────────────────────────────────────────────

def _uf_municipio_da_pasta(nome: str) -> tuple[str, str]:
    """Extrai UF e município de nomes como MG_JUIZ_DE_FORA_01203..._2026."""
    m = re.match(r'^([A-Z]{2})_(.+?)_\d{14}-', nome)
    if m:
        uf = m.group(1)
        municipio = m.group(2).replace("_", " ").title()
        return uf, municipio
    return "", ""


def _processo_da_pasta(nome: str) -> str:
    """Converte MG_..._CNPJ-N-SEQ_ANO → CNPJ-N-SEQ/ANO."""
    m = re.search(r'(\d{14}-\d-\d+)_(\d{4})$', nome)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return nome


def _lista_para_str(lst: list | None, sep: str = " | ") -> str:
    if not lst:
        return ""
    return sep.join(str(x) for x in lst if x)


def _str_curta(texto: str, limite: int = 500) -> str:
    if len(texto) > limite:
        return texto[:limite] + "…"
    return texto


# ── Carregadores ──────────────────────────────────────────────────────────────

def _carregar_licitacoes(path: Path) -> dict[str, dict]:
    """Retorna dict {numeroControlePNCP: licitacao} de pncp_licitacoes.json."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    lista = data.get("licitacoes", data) if isinstance(data, dict) else data
    return {
        lic["numeroControlePNCP"]: lic
        for lic in lista
        if isinstance(lic, dict) and "numeroControlePNCP" in lic
    }


def _carregar_telemetria(path: Path) -> dict[str, dict]:
    """Agrega telemetria por processo: custo, tokens, nº chamadas."""
    if not path.exists():
        return {}
    acum: dict[str, dict] = defaultdict(lambda: {
        "custo_usd": 0.0, "tok_in": 0, "tok_out": 0,
        "cache_r": 0, "cache_w": 0, "lat_ms": 0,
        "n_chamadas": 0, "provedor": "", "modelo": "",
    })
    with open(path, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            try:
                t = json.loads(linha)
            except json.JSONDecodeError:
                continue
            proc = t.get("processo", "")
            if not proc:
                continue
            a = acum[proc]
            a["custo_usd"] += t.get("custo_usd") or 0.0
            a["tok_in"]    += t.get("tok_in") or 0
            a["tok_out"]   += t.get("tok_out") or 0
            a["cache_r"]   += t.get("cache_r") or 0
            a["cache_w"]   += t.get("cache_w") or 0
            a["lat_ms"]    += t.get("lat_ms") or 0
            a["n_chamadas"] += 1
            if t.get("provedor"):
                a["provedor"] = t["provedor"]
            if t.get("modelo"):
                a["modelo"] = t["modelo"]
    return dict(acum)


# ── Lógica principal ──────────────────────────────────────────────────────────

def _modo_processamento(ad: dict) -> str:
    """Classifica como 'meepp', 'pre_filtro' ou 'llm'."""
    if ad.get("_meepp"):
        return "meepp"
    if ad.get("_pre_filtrado"):
        return "pre_filtro"
    provedor = ad.get("_provedor") or ad.get("provedor") or ""
    if provedor == "pre_filtro":
        return "pre_filtro"
    return "llm"


def _extrair_campos(ad: dict) -> dict:
    """
    Normaliza os dois schemas de aderencia.json para um dict comum.

    Schema A (meepp / pre_filtro / legado):
      parecer_geral, prioridade, deve_priorizar, contexto_dominante,
      justificativa, recomendacao_motivo  — todos no nível raiz.

    Schema B (LLM atual):
      resumo_licitacao.{parecer_geral, contexto_dominante, justificativa_geral}
      recomendacao_comercial.{nivel_prioridade, deve_priorizar, motivo}
    """
    # Schema B (LLM)
    resumo  = ad.get("resumo_licitacao") or {}
    recom   = ad.get("recomendacao_comercial") or {}
    if resumo or recom:
        return {
            "parecer_geral":       resumo.get("parecer_geral", ""),
            "prioridade":          recom.get("nivel_prioridade", ""),
            "deve_priorizar":      recom.get("deve_priorizar", False),
            "contexto_dominante":  resumo.get("contexto_dominante", ""),
            "justificativa":       resumo.get("justificativa_geral", ""),
            "recomendacao_motivo": recom.get("motivo", ""),
        }
    # Schema A (meepp / pre_filtro / legado)
    return {
        "parecer_geral":       ad.get("parecer_geral", ""),
        "prioridade":          ad.get("prioridade", ""),
        "deve_priorizar":      ad.get("deve_priorizar", False),
        "contexto_dominante":  ad.get("contexto_dominante", ""),
        "justificativa":       ad.get("justificativa", ""),
        "recomendacao_motivo": ad.get("recomendacao_motivo", ""),
    }


def _carregar_itens_json(pasta: Path) -> dict[int, dict]:
    """Carrega itens.json da pasta → dict {numeroItem: item_dict}."""
    path = pasta / "itens.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # itens.json pode ser lista ou dict com chave "itens"
        if isinstance(data, dict):
            lista = data.get("itens", [])
        else:
            lista = data
        return {int(i.get("numeroItem", 0)): i for i in lista if isinstance(i, dict)}
    except Exception:
        return {}


def exportar(
    editais_dir: Path,
    licitacoes_file: Path,
    telemetria_file: Path,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Carregando dados auxiliares...")
    lic_lookup = _carregar_licitacoes(licitacoes_file)
    tel_lookup  = _carregar_telemetria(telemetria_file)
    print(f"  {len(lic_lookup):,} licitações em {licitacoes_file.name}")
    print(f"  {len(tel_lookup):,} processos na telemetria")

    pastas = sorted([
        p for p in editais_dir.iterdir()
        if p.is_dir() and (p / "aderencia.json").exists()
    ])
    print(f"  {len(pastas):,} pastas com aderencia.json em {editais_dir}/")

    linhas_lic: list[dict] = []
    linhas_itens: list[dict] = []
    erros: list[str] = []

    for pasta in pastas:
        try:
            with open(pasta / "aderencia.json", encoding="utf-8") as f:
                ad = json.load(f)
        except Exception as e:
            erros.append(f"{pasta.name}: {e}")
            continue

        uf_pasta, mun_pasta = _uf_municipio_da_pasta(pasta.name)
        processo = ad.get("_processo") or _processo_da_pasta(pasta.name)

        # Enriquecer com pncp_licitacoes.json
        lic = lic_lookup.get(processo, {})
        unidade  = (lic.get("unidadeOrgao") or {})
        orgao_d  = (lic.get("orgaoEntidade") or {})

        uf       = unidade.get("ufSigla", uf_pasta) or uf_pasta
        municipio= unidade.get("municipioNome", mun_pasta) or mun_pasta
        ibge     = unidade.get("codigoIbge", "")
        orgao    = orgao_d.get("razaoSocial", "")
        objeto   = lic.get("objetoCompra", "")
        val_est  = lic.get("valorTotalEstimado", "")
        val_hom  = lic.get("valorTotalHomologado", "")
        dt_enc   = (lic.get("dataEncerramentoProposta") or "")[:10]
        srp      = lic.get("srp", "")
        situacao = lic.get("situacaoCompraNome", "")
        link     = lic.get("linkSistemaOrigem", "")

        # Telemetria
        tel = tel_lookup.get(processo, {})

        # Modo, provedor e campos normalizados entre os dois schemas
        modo = _modo_processamento(ad)
        campos = _extrair_campos(ad)
        provedor = ad.get("_provedor") or ad.get("provedor") or tel.get("provedor", "")
        modelo   = ad.get("_modelo")   or ad.get("modelo")   or tel.get("modelo", "")
        analisado_em = (ad.get("_analisado_em") or ad.get("analisado_em") or "")[:19]

        est = ad.get("estatisticas") or {}
        total_itens = (
            ad.get("total_itens")
            or len(ad.get("itens_analisados") or [])
        )

        linhas_lic.append({
            # ── Identificação ──────────────────────────────────────────
            "processo":          processo,
            "uf":                uf,
            "municipio":         municipio,
            "codigo_ibge":       ibge,
            "orgao":             orgao,
            "objeto_compra":     _str_curta(objeto, 300),
            "link_portal":       link,
            # ── Análise ────────────────────────────────────────────────
            "parecer_geral":          campos["parecer_geral"],
            "prioridade":             campos["prioridade"],
            "deve_priorizar":         int(bool(campos["deve_priorizar"])),
            "modo_processamento":     modo,
            # ── Contexto LLM (vazio para pré-filtrados) ────────────────
            "contexto_dominante":     _str_curta(campos["contexto_dominante"], 300),
            "justificativa":          _str_curta(campos["justificativa"], 300),
            "recomendacao_motivo":    _str_curta(campos["recomendacao_motivo"], 300),
            # ── Contadores de itens ────────────────────────────────────
            "total_itens":            total_itens,
            "aderencia_direta":       est.get("aderencia_direta", 0),
            "aderencia_parcial_forte":est.get("aderencia_parcial_forte", 0),
            "aderencia_parcial_fraca":est.get("aderencia_parcial_fraca", 0),
            "falso_positivo_lexical": est.get("falso_positivo_lexical", 0),
            "nao_aderente":           est.get("nao_aderente", 0),
            # ── Dados da licitação ─────────────────────────────────────
            "valor_estimado":    val_est,
            "valor_homologado":  val_hom,
            "data_encerramento": dt_enc,
            "srp":               int(bool(srp)) if srp != "" else "",
            "situacao":          situacao,
            # ── Processamento ──────────────────────────────────────────
            "provedor":          provedor,
            "modelo":            modelo,
            "analisado_em":      analisado_em,
            # ── Custo (telemetria) ─────────────────────────────────────
            "custo_usd":         round(tel.get("custo_usd", 0), 6),
            "tok_in":            tel.get("tok_in", ""),
            "tok_out":           tel.get("tok_out", ""),
            "cache_r":           tel.get("cache_r", ""),
            "cache_w":           tel.get("cache_w", ""),
            "n_chamadas_llm":    tel.get("n_chamadas", 0),
            "lat_ms_total":      tel.get("lat_ms", ""),
        })

        # ── Itens individuais ─────────────────────────────────────────────
        # Descricao pode estar no aderencia.json (schema A) ou só no itens.json (schema B/LLM)
        itens_json = _carregar_itens_json(pasta)   # {numeroItem: item_dict}

        for item in (ad.get("itens_analisados") or []):
            num = item.get("numeroItem")
            # Tentar descricao do aderencia.json primeiro; fallback: itens.json
            descricao = (item.get("descricao") or "").strip()
            if not descricao and num is not None:
                itens_raw = itens_json.get(int(num)) or {}
                descricao = (itens_raw.get("descricao") or "").strip()

            linhas_itens.append({
                "processo":               processo,
                "uf":                     uf,
                "municipio":              municipio,
                "parecer_licitacao":      campos["parecer_geral"],
                "modo_processamento":     modo,
                "numeroItem":             num,
                "descricao":              _str_curta(descricao, 400),
                "classificacao":          item.get("classificacao", ""),
                "grau_confianca":         item.get("grau_confianca", ""),
                "produto_referencia":     item.get("produto_referencia", "") or "",
                "categoria_portfolio":    item.get("categoria_portfolio_relacionada", "") or "",
                "subcategoria_portfolio": item.get("subcategoria_portfolio_relacionada", "") or "",
                "motivos_pro":    _lista_para_str(item.get("motivos_pro_match")),
                "motivos_contra": _lista_para_str(item.get("motivos_contra_match")),
                "sinais_lexicais":_lista_para_str(item.get("sinais_lexicais_enganosos")),
                "observacao":     item.get("observacao", ""),
            })

    # Ordenar: prioridade → processo
    linhas_lic.sort(key=lambda r: (
        _PRIORIDADE_ORD.get(r["prioridade"], 9),
        r["processo"],
    ))

    # Escrever CSVs
    def _csv(nome: str, linhas: list[dict]) -> Path:
        path = out_dir / nome
        if not linhas:
            print(f"  [aviso] Nenhum dado para {nome}")
            return path
        with open(path, "w", newline="", encoding=ENCODING) as f:
            w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
            w.writeheader()
            w.writerows(linhas)
        print(f"  OK {path}  ({len(linhas):,} linhas)")
        return path

    print("\nEscrevendo arquivos...")
    path_lic   = _csv(OUT_LICITACOES, linhas_lic)
    path_itens = _csv(OUT_ITENS,      linhas_itens)

    # Resumo estatístico
    total_lic = len(linhas_lic)
    por_modo  = defaultdict(int)
    por_pare  = defaultdict(int)
    custo_tot = 0.0
    for r in linhas_lic:
        por_modo[r["modo_processamento"]] += 1
        por_pare[r["parecer_geral"]] += 1
        custo_tot += float(r["custo_usd"] or 0)

    linhas_resumo = [
        "RESUMO DA EXPORTAÇÃO POWER BI",
        "=" * 40,
        f"Licitações exportadas : {total_lic:,}",
        f"Itens exportados       : {len(linhas_itens):,}",
        f"Custo LLM total (USD)  : ${custo_tot:.4f}",
        "",
        "Por modo de processamento:",
    ]
    for modo, cnt in sorted(por_modo.items()):
        linhas_resumo.append(f"  {modo:15s} {cnt:,}")
    linhas_resumo += ["", "Por parecer geral:"]
    for pare, cnt in sorted(por_pare.items(), key=lambda x: -x[1]):
        linhas_resumo.append(f"  {pare:25s} {cnt:,}")
    if erros:
        linhas_resumo += ["", f"Erros ({len(erros)}):"]
        for e in erros[:10]:
            linhas_resumo.append(f"  {e}")

    resumo_txt = "\n".join(linhas_resumo)
    (out_dir / OUT_RESUMO).write_text(resumo_txt, encoding="utf-8")

    print(resumo_txt)
    print(f"\n  -> {path_lic}  ({len(linhas_lic):,} linhas)")
    print(f"  -> {path_itens}  ({len(linhas_itens):,} linhas)")
    print(f"  -> {out_dir / OUT_RESUMO}")
    if erros:
        print(f"  [!] {len(erros)} arquivo(s) com erro — ver {OUT_RESUMO}")


# ── Ponto de entrada ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exporta dados de aderência para Power BI")
    p.add_argument("--editais",     default=str(EDITAIS_DIR),     help="Pasta editais/")
    p.add_argument("--licitacoes",  default=str(LICITACOES_FILE), help="pncp_licitacoes.json")
    p.add_argument("--telemetria",  default=str(TELEMETRIA_FILE), help="pncp_telemetria.jsonl")
    p.add_argument("--out",         default=str(OUT_DIR),         help="Pasta de saída")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    exportar(
        editais_dir     = Path(args.editais),
        licitacoes_file = Path(args.licitacoes),
        telemetria_file = Path(args.telemetria),
        out_dir         = Path(args.out),
    )
