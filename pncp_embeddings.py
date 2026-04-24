#!/usr/bin/env python3
"""
pncp_embeddings.py — Filtro de similaridade para pré-qualificação de licitações

PARTE 1 — Estudo de viabilidade
  Simula um filtro aplicado APÓS o pré-filtro léxico e ANTES do LLM,
  usando os aderencia.json existentes como ground truth.
  Responde: em qual threshold consigo X% de economia de chamadas LLM
  sem perder nenhum item aderente?

PARTE 2 — Implementação de produção
  A classe IndicePortfolio é pronta para integração em pncp_agente.py.
  Basta instanciar uma vez e chamar score_maximo(descricao_item) por item.

Backends disponíveis:
  "semantic"  → sentence-transformers (recomendado, semântico, GPU/CPU)
  "tfidf"     → scikit-learn TF-IDF (fallback léxico, sem dependência GPU)
  "auto"      → tenta semantic, cai para tfidf se não disponível

Dependências:
  Backend semantic (recomendado):  pip install sentence-transformers
  Backend tfidf   (fallback):      pip install scikit-learn

Uso:
  python pncp_embeddings.py                          # estudo completo
  python pncp_embeddings.py --backend tfidf          # força TF-IDF
  python pncp_embeddings.py --backend semantic       # força semântico
  python pncp_embeddings.py --modelo BAAI/bge-m3     # modelo específico
  python pncp_embeddings.py --threshold 0.30         # threshold adicional
  python pncp_embeddings.py --item "colete balístico nij iiia"  # testa item

Integração em pncp_agente.py:
  from pncp_embeddings import IndicePortfolio
  _indice = IndicePortfolio.carregar_ou_construir(
      "prompts/portfolio_mestre_resgatecnica_lite_v2.json",
      backend="auto",
  )
  # Em coletar(), após pré-filtro léxico:
  textos = [item.get("descricao", "") for item in itens_elegiveis]
  score_max = max((_indice.score_maximo(t) for t in textos if t), default=0.0)
  if score_max < EMBEDDING_THRESHOLD:
      continue  # não envia ao LLM
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Garante UTF-8 no console Windows (Python 3.7+)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Detecção de dependências ──────────────────────────────────────────────────

try:
    import numpy as np
    _NUMPY_OK = True
except ImportError:
    _NUMPY_OK = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _cos_sim_sklearn
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

try:
    from sentence_transformers import SentenceTransformer
    import torch
    _ST_OK = True
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    _ST_OK = False
    _DEVICE = "cpu"

# ── Configuração ──────────────────────────────────────────────────────────────
PORTFOLIO_FILE  = Path("prompts/portfolio_mestre_resgatecnica_lite_v2.json")
EDITAIS_DIR     = Path("editais")
RELATORIO_FILE  = Path("embeddings_viabilidade.txt")
CACHE_DIR       = Path(".embeddings_cache")   # pasta para cache de vetores
CACHE_SCHEMA_VERSION = "v4_base_text_aph_aliases"

# Backend padrão: "auto" tenta semantic, cai para tfidf se não disponível
BACKEND_PADRAO  = "auto"

# Modelo semântico padrão (estado da arte multilingual, ~570 MB)
MODELO_SEMANTICO_PADRAO = "BAAI/bge-m3"

# Modelos alternativos comentados para referência:
# "paraphrase-multilingual-MiniLM-L12-v2"   # ~120 MB, mais leve, bom para CPU/XPS
# "paraphrase-multilingual-mpnet-base-v2"   # ~420 MB, qualidade intermediária
# "intfloat/multilingual-e5-large"           # ~560 MB, excelente alternativa ao bge-m3
# "rufimelo/Legal-BERTimbau-sts-large-IBAMA" # ~400 MB, PT-BR jurídico (domínio licitações)

# Threshold padrão para produção (calibrar após estudo com rodada completa)
EMBEDDING_THRESHOLD_DEFAULT = 0.30

# Thresholds avaliados no estudo
THRESHOLDS_ESTUDO = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]

# ── Bônus/penalidades de calibração (ajustáveis via config.yaml) ──────────────
# Estes valores controlam _ajustar_scores_consulta(). Edite diretamente aqui
# ou via config.yaml → seção "calibracao" (o config.yaml tem precedência).

BONUS_SERVICO_APH_CATEGORIA          = 0.20  # "atendimento pre hospitalar" em intencao_servico_aph
BONUS_SERVICO_APH_RESGATE_EVA        = 0.16  # "resgate e evacuacao" em intencao_servico_aph
BONUS_SERVICO_APH_BOLSAS             = 0.08  # "bolsas" em intencao_servico_aph
BONUS_SERVICO_APH_OXIGENIO           = 0.06  # produtos com oxigênio/resgate em intencao_servico_aph
PENALIDADE_SERVICO_APH_VEICULAR      = 0.08  # "veiculos especiais" em intencao_servico_aph
BONUS_VEICULAR_FORTE_CATEGORIA       = 0.18  # "veiculos especiais customizados" em intencao_veicular_forte
BONUS_VEICULAR_FORTE_RESGATE         = 0.07  # "resgate veicular" em intencao_veicular_forte
PENALIDADE_VEICULAR_BOLSAS_COLETES   = 0.12  # bolsas/coletes em intencao_veicular_forte
BONUS_VIATURA_ROD_MOBILIDADE         = 0.18  # "mobilidade operacional" em intencao_viatura_rodoviaria
PENALIDADE_VIATURA_ROD_MOTO          = 0.14  # motocicletas em intencao_viatura_rodoviaria


def _aplicar_config_calibracao(cfg: dict) -> None:
    """Sobrescreve constantes de calibração com valores do config.yaml."""
    global BONUS_SERVICO_APH_CATEGORIA, BONUS_SERVICO_APH_RESGATE_EVA
    global BONUS_SERVICO_APH_BOLSAS, BONUS_SERVICO_APH_OXIGENIO
    global PENALIDADE_SERVICO_APH_VEICULAR
    global BONUS_VEICULAR_FORTE_CATEGORIA, BONUS_VEICULAR_FORTE_RESGATE
    global PENALIDADE_VEICULAR_BOLSAS_COLETES
    global BONUS_VIATURA_ROD_MOBILIDADE, PENALIDADE_VIATURA_ROD_MOTO

    c = cfg.get("calibracao", {})
    if not c:
        return

    BONUS_SERVICO_APH_CATEGORIA        = float(c.get("bonus_servico_aph_categoria",        BONUS_SERVICO_APH_CATEGORIA))
    BONUS_SERVICO_APH_RESGATE_EVA      = float(c.get("bonus_servico_aph_resgate_evacuacao", BONUS_SERVICO_APH_RESGATE_EVA))
    BONUS_SERVICO_APH_BOLSAS           = float(c.get("bonus_servico_aph_bolsas",            BONUS_SERVICO_APH_BOLSAS))
    BONUS_SERVICO_APH_OXIGENIO         = float(c.get("bonus_servico_aph_oxigenio",          BONUS_SERVICO_APH_OXIGENIO))
    PENALIDADE_SERVICO_APH_VEICULAR    = float(c.get("penalidade_servico_aph_veicular",     PENALIDADE_SERVICO_APH_VEICULAR))
    BONUS_VEICULAR_FORTE_CATEGORIA     = float(c.get("bonus_veicular_forte_categoria",      BONUS_VEICULAR_FORTE_CATEGORIA))
    BONUS_VEICULAR_FORTE_RESGATE       = float(c.get("bonus_veicular_forte_resgate",        BONUS_VEICULAR_FORTE_RESGATE))
    PENALIDADE_VEICULAR_BOLSAS_COLETES = float(c.get("penalidade_veicular_bolsas_coletes",  PENALIDADE_VEICULAR_BOLSAS_COLETES))
    BONUS_VIATURA_ROD_MOBILIDADE       = float(c.get("bonus_viatura_rodoviaria_mobilidade", BONUS_VIATURA_ROD_MOBILIDADE))
    PENALIDADE_VIATURA_ROD_MOTO        = float(c.get("penalidade_viatura_rodoviaria_moto",  PENALIDADE_VIATURA_ROD_MOTO))


# Aplica config.yaml automaticamente ao importar o módulo
try:
    from pncp_config import carregar_config as _carregar_config
    _aplicar_config_calibracao(_carregar_config())
except Exception:
    pass  # config opcional — falha silenciosa


@dataclass(frozen=True)
class PortfolioEntry:
    id: str
    label: str
    texto: str
    texto_base: str
    categoria: str
    subcategoria: str
    produto: str

# ── Classes de aderência ──────────────────────────────────────────────────────
ClassLabel = Literal[
    "ADERENCIA_DIRETA", "ADERENCIA_PARCIAL_FORTE",
    "ADERENCIA_PARCIAL_FRACA", "FALSO_POSITIVO_LEXICAL", "NAO_ADERENTE",
]
# Aceita maiúsculas E minúsculas (versões antigas do agente usavam minúsculas)
CRITICOS   = {"ADERENCIA_DIRETA", "ADERENCIA_PARCIAL_FORTE",
              "aderencia_direta", "aderencia_parcial_forte"}
BORDERLINE = {"ADERENCIA_PARCIAL_FRACA", "aderencia_parcial_fraca"}
NEGATIVOS  = {"FALSO_POSITIVO_LEXICAL", "NAO_ADERENTE",
              "falso_positivo_lexical", "nao_aderente"}


# ══════════════════════════════════════════════════════════════════════════════
# Extração de textos do portfólio
# ══════════════════════════════════════════════════════════════════════════════

def _entradas_portfolio(portfolio: dict) -> list[PortfolioEntry]:
    """
    Extrai textos de todos os produtos do portfólio.
    Cada entrada: (texto_enriquecido, label_para_log).
    Enriquecimento: nome + subcategoria + categoria + contexto_operacional + domínio.
    """
    palavras_chave = " ".join(
        portfolio.get("perfil_empresa_resumido", {}).get("palavras_chave_dominio", [])
    )
    entradas: list[PortfolioEntry] = []
    idx = 0

    for cat in portfolio.get("categorias_resumidas", []):
        cat_nome  = cat.get("categoria_canonica", "")
        ctx_op    = " ".join(cat.get("contexto_operacional", []))
        tipo_forn = cat.get("tipo_fornecimento", "")

        for sub in cat.get("subcategorias_chave", []):
            sub_nome = sub.get("subcategoria", "")
            produtos = sub.get("exemplos_produtos", [])
            status = str(sub.get("status", ""))
            entradas_sub = list(produtos)

            # O portfolio lite registra algumas subcategorias confirmadas no
            # catalogo, mas sem itens estruturados. Criamos uma entrada
            # sintetica local para retrieval, preservando o JSON-fonte.
            if (not entradas_sub) and ("confirmada_no_catalogo" in status):
                entradas_sub.append(sub_nome)

            for produto in entradas_sub:
                aliases: list[str] = []
                cat_lower = cat_nome.lower()
                sub_lower = sub_nome.lower()
                prod_lower = str(produto).lower()

                if "veículos especiais" in cat_lower or "veiculos especiais" in cat_lower:
                    aliases.extend([
                        "viatura especial",
                        "veículo operacional",
                        "veiculo operacional",
                        "viatura",
                        "ambulância",
                        "ambulancia",
                        "unidade móvel",
                        "unidade movel",
                        "ambulância de simples remoção",
                        "ambulancia de simples remocao",
                        "transporte sanitário",
                        "transporte sanitario",
                    ])
                if "atendimento pré-hospitalar" in cat_lower or "atendimento pre-hospitalar" in cat_lower:
                    aliases.extend([
                        "aph",
                        "atendimento pré-hospitalar",
                        "atendimento pre-hospitalar",
                        "socorro",
                        "urgência",
                        "urgencia",
                        "emergência",
                        "emergencia",
                        "ambulância",
                        "ambulancia",
                        "resgate",
                        "remoção",
                        "remocao",
                        "uti móvel",
                        "uti movel",
                        "transporte de paciente",
                        "transporte inter-hospitalar",
                        "transporte intra-hospitalar",
                        "transporte sanitário",
                        "transporte sanitario",
                        "suporte básico",
                        "suporte basico",
                        "suporte avançado",
                        "suporte avancado",
                        "ambulância tipo a",
                        "ambulancia tipo a",
                        "ambulância tipo b",
                        "ambulancia tipo b",
                        "ambulância tipo d",
                        "ambulancia tipo d",
                    ])
                if "resgate veicular" in cat_lower:
                    aliases.extend([
                        "resgate veicular",
                        "salvamento veicular",
                        "desencarceramento",
                    ])
                if "motocicletas operacionais" in sub_lower:
                    aliases.extend(["motocicleta operacional", "motocicleta de emergência"])
                if "resgate e evacuação" in sub_lower or "resgate e evacuacao" in sub_lower:
                    aliases.extend([
                        "evacuação",
                        "evacuacao",
                        "remoção",
                        "remocao",
                        "transporte de paciente",
                        "transporte sanitário",
                        "transporte sanitario",
                    ])
                if "bolsas" in sub_lower:
                    aliases.extend([
                        "kit aph",
                        "suporte básico",
                        "suporte avançado",
                        "ambulância",
                        "ambulancia",
                    ])
                if "coletes" in sub_lower:
                    aliases.extend(["socorrista", "resgate", "aph"])
                if "motobombas" in sub_lower or "bombas" in sub_lower:
                    aliases.extend(["unidade de resgate", "equipamento embarcado"])
                if "prancha" in prod_lower or "colar" in prod_lower or "dea" in prod_lower:
                    aliases.extend(["ambulância", "ambulancia", "aph", "urgência", "emergência"])
                if "bolsa para transporte" in prod_lower:
                    aliases.extend(["ambulância", "ambulancia", "resgate"])
                if "resgate básico" in prod_lower or "resgate avançado" in prod_lower:
                    aliases.extend(["ambulância", "ambulancia", "aph", "socorro"])
                if "oxigênio" in prod_lower or "oxigenio" in prod_lower:
                    aliases.extend(["ambulância", "ambulancia", "suporte respiratório", "suporte respiratorio"])
                if "veículo" in cat_lower or "veiculo" in cat_lower:
                    aliases.extend(["ambulância", "ambulancia", "furgão", "furgoneta", "minivan"])
                if "mobilidade operacional" in sub_lower:
                    aliases.extend([
                        "ambulância",
                        "ambulancia",
                        "viatura de resgate",
                        "veículo de remoção",
                        "veiculo de remocao",
                        "uti móvel",
                        "uti movel",
                        "transporte sanitário",
                        "transporte sanitario",
                    ])
                if "combate a incêndio" in sub_lower or "combate a incendio" in sub_lower:
                    aliases.extend(["viatura operacional", "veículo especial", "veiculo especial"])
                if "todo-terreno" in sub_lower or "todo terreno" in sub_lower:
                    aliases.extend(["veículo especial", "veiculo especial", "resgate em área remota"])

                texto_base = " ".join(
                    part for part in [produto, sub_nome, " ".join(sorted(set(aliases)))] if part
                ).strip()
                texto = " ".join(
                    part for part in [texto_base, cat_nome, ctx_op, tipo_forn, palavras_chave]
                    if part
                )
                entradas.append(PortfolioEntry(
                    id=f"prod_{idx:04d}",
                    label=f"{cat_nome} > {sub_nome} > {produto[:40]}",
                    texto=texto,
                    texto_base=texto_base,
                    categoria=cat_nome,
                    subcategoria=sub_nome,
                    produto=produto,
                ))
                idx += 1

    if not entradas:
        entradas.append(PortfolioEntry(
            id="global_0000",
            label="global",
            texto=palavras_chave,
            texto_base=palavras_chave,
            categoria="",
            subcategoria="",
            produto="",
        ))

    return entradas


def _hash_portfolio(portfolio_path: Path) -> str:
    """SHA256 do arquivo de portfólio — invalida cache automaticamente se mudar."""
    return hashlib.sha256(portfolio_path.read_bytes()).hexdigest()[:16]


# Stop words portuguesas que não carregam significado no domínio de equipamentos
_STOP_WORDS_PT = {
    "a", "ao", "aos", "as", "até", "com", "da", "das", "de", "do", "dos",
    "e", "em", "entre", "essa", "esse", "esta", "este", "eu", "foi", "for",
    "há", "isso", "isto", "já", "mas", "me", "na", "nas", "nem", "no", "nos",
    "o", "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "por", "que",
    "se", "sem", "seu", "sua", "suas", "seus", "são", "também", "te", "tem",
    "ter", "tipo", "um", "uma", "uns", "umas", "à", "às",
}


def _normalizar_tfidf(texto: str) -> str:
    """Normalização para TF-IDF: minúsculas, remove stop words PT, preserva acentos."""
    texto = texto.lower()
    texto = re.sub(r'[^\w\sáàâãéèêíïóôõúüçñ]', ' ', texto)
    tokens = [t for t in texto.split() if t not in _STOP_WORDS_PT and len(t) > 1]
    return " ".join(tokens)


def _normalizar_consulta(texto: str) -> str:
    nfkd = unicodedata.normalize("NFKD", texto or "")
    texto = "".join(c for c in nfkd if not unicodedata.combining(c)).lower()
    texto = re.sub(r"[^\w\s]", " ", texto)
    return " ".join(texto.split())


# ══════════════════════════════════════════════════════════════════════════════
# IndicePortfolio — suporte a dois backends
# ══════════════════════════════════════════════════════════════════════════════

class IndicePortfolio:
    """
    Índice de similaridade do portfólio Resgatécnica.
    Suporta dois backends com a mesma interface pública:
      - "semantic": sentence-transformers (bge-m3 ou similar) — recomendado
      - "tfidf":    scikit-learn TF-IDF — fallback léxico sem GPU

    Interface pública (idêntica em ambos os backends):
      score_maximo(texto_item: str) -> float   # score ∈ [0, 1]
      top_k(texto_item, k=5) -> [(score, label), ...]
      backend: str
      modelo: str
      n_produtos: int
    """

    def __init__(self) -> None:
        self._backend: str = ""
        self._modelo: str = ""
        self._labels: list[str] = []
        self._entries: list[PortfolioEntry] = []
        self._n: int = 0

        # Backend TF-IDF
        self._vec_tfidf = None
        self._mat_tfidf = None

        # Backend semântico
        self._st_model = None
        self._mat_sem  = None   # np.ndarray shape (n_produtos, dim)

    # ── Construção ────────────────────────────────────────────────────────────

    def construir(
        self,
        portfolio: dict,
        backend: str = "auto",
        modelo: str = MODELO_SEMANTICO_PADRAO,
        portfolio_path: Path | None = None,
    ) -> "IndicePortfolio":
        """
        Constrói o índice a partir do dict de portfólio.
        backend: "auto" | "semantic" | "tfidf"
        modelo:  nome do modelo HuggingFace (só usado no backend semantic)
        portfolio_path: se fornecido, tenta carregar/salvar cache de vetores
        """
        backend_real = self._resolver_backend(backend)
        entradas     = _entradas_portfolio(portfolio)
        self._entries = entradas
        self._labels = [e.label for e in entradas]
        self._n      = len(entradas)

        if backend_real == "semantic":
            self._construir_semantico(
                [e.texto_base for e in entradas], modelo, portfolio_path
            )
        else:
            self._construir_tfidf([e.texto_base for e in entradas])

        return self

    def _resolver_backend(self, backend: str) -> str:
        if backend == "semantic":
            if not _ST_OK:
                raise ImportError(
                    "sentence-transformers não encontrado.\n"
                    "Execute: pip install sentence-transformers"
                )
            return "semantic"
        if backend == "tfidf":
            if not _SKLEARN_OK:
                raise ImportError(
                    "scikit-learn não encontrado.\n"
                    "Execute: pip install scikit-learn"
                )
            return "tfidf"
        # auto: preferir semantic
        if _ST_OK:
            return "semantic"
        if _SKLEARN_OK:
            print("[aviso] sentence-transformers não disponível — usando TF-IDF como fallback.")
            return "tfidf"
        raise ImportError(
            "Nenhum backend disponível.\n"
            "Execute:  pip install sentence-transformers\n"
            "     ou:  pip install scikit-learn"
        )

    def _construir_tfidf(self, textos: list[str]) -> None:
        self._vec_tfidf = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_features=25_000,
            sublinear_tf=True,
        )
        self._mat_tfidf = self._vec_tfidf.fit_transform(
            [_normalizar_tfidf(t) for t in textos]
        )
        self._backend = "tfidf"
        self._modelo  = "TF-IDF scikit-learn"
        print(f"  Índice TF-IDF: {self._n} produtos, "
              f"{self._mat_tfidf.shape[1]:,} features")

    def _construir_semantico(
        self,
        textos: list[str],
        modelo: str,
        portfolio_path: Path | None,
    ) -> None:
        cache_path = self._cache_path(modelo, portfolio_path)
        vecs = self._carregar_cache(cache_path, portfolio_path)

        if vecs is None:
            print(f"  Carregando modelo '{modelo}' em {_DEVICE.upper()}...")
            self._st_model = SentenceTransformer(modelo, device=_DEVICE)
            print(f"  Codificando {len(textos)} produtos do portfólio...")
            vecs = self._st_model.encode(
                textos,
                normalize_embeddings=True,   # necessário para cosine via dot product
                batch_size=64,
                show_progress_bar=False,
                device=_DEVICE,
            )
            self._salvar_cache(cache_path, vecs, portfolio_path)
        else:
            print(f"  Cache carregado: {len(textos)} vetores de portfólio ({modelo})")
            # Modelo ainda necessário para codificar itens de consulta
            self._st_model = SentenceTransformer(modelo, device=_DEVICE)

        self._mat_sem = vecs   # shape (n_produtos, dim)
        self._backend = "semantic"
        self._modelo  = modelo
        print(f"  Índice semântico: {self._n} produtos, "
              f"dim={vecs.shape[1]}, device={_DEVICE.upper()}")

    # ── Cache de vetores ──────────────────────────────────────────────────────

    def _cache_path(self, modelo: str, portfolio_path: Path | None) -> Path | None:
        if portfolio_path is None or not _NUMPY_OK:
            return None
        CACHE_DIR.mkdir(exist_ok=True)
        modelo_slug = re.sub(r'[^a-zA-Z0-9_-]', '_', modelo)
        phash = _hash_portfolio(portfolio_path)
        return CACHE_DIR / f"portfolio_{modelo_slug}_{phash}_{CACHE_SCHEMA_VERSION}.npy"

    def _carregar_cache(
        self, cache_path: Path | None, portfolio_path: Path | None
    ) -> "np.ndarray | None":
        if cache_path is None or not cache_path.exists():
            return None
        try:
            vecs = np.load(str(cache_path))
            if vecs.ndim == 2 and vecs.shape[0] == self._n:
                return vecs
        except Exception:
            pass
        return None

    def _salvar_cache(
        self, cache_path: Path | None, vecs: "np.ndarray", portfolio_path: Path | None
    ) -> None:
        if cache_path is None or not _NUMPY_OK:
            return
        try:
            np.save(str(cache_path), vecs)
            print(f"  Cache salvo: {cache_path.name}")
        except Exception as e:
            print(f"  [aviso] Não foi possível salvar cache: {e}")

    # ── Interface pública ─────────────────────────────────────────────────────

    def score_maximo(self, texto_item: str) -> float:
        """
        Similaridade máxima entre o texto do item e qualquer produto do portfólio.
        Retorna valor em [0, 1].
        """
        if not texto_item or not texto_item.strip():
            return 0.0
        if self._backend == "semantic":
            return self._score_semantico(texto_item)
        return self._score_tfidf(texto_item)

    def top_k(self, texto_item: str, k: int = 5) -> list[tuple[float, str]]:
        """Retorna os k produtos mais similares com seus scores (debug)."""
        candidatos = self.top_k_candidatos(texto_item, k=k)
        return [(c["score"], c["label"]) for c in candidatos]

    def top_k_candidatos(self, texto_item: str, k: int = 5) -> list[dict]:
        """Retorna os k produtos mais similares com metadados estruturados."""
        if not texto_item or not texto_item.strip():
            return []
        if self._backend == "semantic":
            sims = self._sims_semantico(texto_item)
        else:
            sims = self._sims_tfidf(texto_item)
        sims = self._ajustar_scores_consulta(sims, texto_item)
        idx_top = sims.argsort()[::-1][:k]
        resultados: list[dict] = []
        for i in idx_top:
            score = float(sims[i])
            if score <= 0:
                continue
            entry = self._entries[i]
            resultados.append({
                "id": entry.id,
                "label": entry.label,
                "score": score,
                "categoria": entry.categoria,
                "subcategoria": entry.subcategoria,
                "produto": entry.produto,
                "texto": entry.texto,
                "texto_base": entry.texto_base,
            })
        return resultados

    def _ajustar_scores_consulta(self, sims: "np.ndarray", texto_item: str) -> "np.ndarray":
        """
        Aplica pequenos bônus/penalidades heurísticos por domínio para consultas
        conhecidas, sem substituir o retrieval vetorial.
        """
        texto = _normalizar_consulta(texto_item)
        termos_ambul = any(t in texto for t in ["ambulancia", "furgoneta", "uti movel", "simples remocao", "remocao"])
        termos_veic = any(t in texto for t in ["veiculo", "furgao", "minivan", "viatura"])
        termos_aph = any(t in texto for t in ["aph", "urgencia", "emergencia", "socorro"])
        termos_servico = any(
            t in texto
            for t in [
                "servico",
                "contratacao",
                "prestacao",
                "locacao",
                "transporte de pacientes",
                "transporte de paciente",
                "transporte de urgencia",
                "transporte sanitario",
                "transporte inter hospitalar",
                "remocao terrestre",
            ]
        )
        intencao_veicular_forte = any(
            t in texto
            for t in [
                "ambulancia",
                "furgoneta",
                "furgao",
                "minivan",
                "viatura",
                "veiculo automotor",
                "uti movel",
            ]
        )
        intencao_viatura_rodoviaria = any(
            t in texto
            for t in [
                "furgao",
                "furgoneta",
                "pick up",
                "pickup",
                "4 x 4",
                "4x4",
                "minivan",
                "tipo a",
                "tipo b",
                "tipo c",
                "simples remocao",
                "transporte sanitario",
            ]
        )
        intencao_servico_aph = termos_servico and any(
            t in texto
            for t in [
                "uti movel",
                "remocao",
                "urgencia",
                "emergencia",
                "ambulancia",
                "minivan",       # "contratação empresa... transporte de pacientes em minivan"
                "transporte sanitario",
            ]
        )

        if not (termos_ambul or termos_veic or termos_aph):
            return sims

        ajustados = sims.copy()
        for i, entry in enumerate(self._entries):
            cat = _normalizar_consulta(entry.categoria)
            sub = _normalizar_consulta(entry.subcategoria)
            prod = _normalizar_consulta(entry.produto)
            bonus = 0.0

            if termos_ambul or termos_aph:
                if "atendimento pre hospitalar" in cat:
                    bonus += 0.10
                if "veiculos especiais customizados" in cat:
                    bonus += 0.06
                if "resgate e evacuacao" in sub:
                    bonus += 0.05
                if "bolsas" in sub or "kit" in sub:
                    bonus += 0.04
                if any(t in prod for t in ["oxigenio", "resgate", "evacuacao", "socorrista"]):
                    bonus += 0.03

            if termos_veic:
                if "veiculos especiais customizados" in cat:
                    bonus += 0.12
                if "resgate veicular" in cat:
                    bonus += 0.05
                if any(t in prod for t in ["sherp", "motocicleta", "resgate"]):
                    bonus += 0.02

            if intencao_veicular_forte:
                if "veiculos especiais customizados" in cat:
                    bonus += BONUS_VEICULAR_FORTE_CATEGORIA
                if "resgate veicular" in cat:
                    bonus += BONUS_VEICULAR_FORTE_RESGATE
                if any(t in sub for t in ["motocicletas", "veiculos", "evacuacao"]):
                    bonus += 0.04
                if any(t in prod for t in ["sherp", "search and rescue", "firefighting"]):
                    bonus += 0.04

                # Quando a consulta fala explicitamente de ambulância/viatura,
                # candidatos APH de suporte continuam úteis, mas não devem
                # dominar o topo contra itens do universo veicular.
                if "bolsas" in sub or "coletes" in sub:
                    bonus -= PENALIDADE_VEICULAR_BOLSAS_COLETES
                if any(t in prod for t in ["cadeira", "prancha", "colar", "bolsa", "colete"]):
                    bonus -= 0.10

            if intencao_viatura_rodoviaria:
                if "solucoes customizadas de mobilidade operacional" in sub:
                    bonus += BONUS_VIATURA_ROD_MOBILIDADE
                if "veiculos especiais customizados" in cat:
                    bonus += 0.04
                if "motocicletas operacionais" in sub or "motocicleta para combate a incendio" in sub:
                    bonus -= PENALIDADE_VIATURA_ROD_MOTO
                if "veiculos anfibios e todo terreno" in sub:
                    bonus -= 0.08
                if any(t in prod for t in ["sherp", "firefighting", "the ark", "motocicleta", "bmw"]):
                    bonus -= 0.06

            if intencao_servico_aph:
                if "atendimento pre hospitalar" in cat:
                    bonus += BONUS_SERVICO_APH_CATEGORIA
                if "resgate e evacuacao" in sub:
                    bonus += BONUS_SERVICO_APH_RESGATE_EVA
                if "imobilizadores" in sub or "kits prontos" in sub:
                    bonus += 0.04
                if "bolsas" in sub:
                    bonus += BONUS_SERVICO_APH_BOLSAS
                if "veiculos especiais customizados" in cat:
                    bonus -= PENALIDADE_SERVICO_APH_VEICULAR
                if "resgate e evacuacao" in sub and "cadeira" in prod:
                    bonus -= 0.14
                if any(t in prod for t in ["kit parto", "kit queimadura"]):
                    bonus -= 0.08
                if any(t in prod for t in ["oxigenio", "resgate avancado", "resgate basico"]):
                    bonus += BONUS_SERVICO_APH_OXIGENIO

            if ("resgate aquatico" in cat) and (termos_ambul or termos_veic):
                bonus -= 0.05

            ajustados[i] = max(0.0, float(ajustados[i]) + bonus)
        return ajustados

    def _score_tfidf(self, texto: str) -> float:
        vec = self._vec_tfidf.transform([_normalizar_tfidf(texto)])
        return float(_cos_sim_sklearn(vec, self._mat_tfidf).max())

    def _sims_tfidf(self, texto: str) -> "np.ndarray":
        vec = self._vec_tfidf.transform([_normalizar_tfidf(texto)])
        return _cos_sim_sklearn(vec, self._mat_tfidf)[0]

    def _score_semantico(self, texto: str) -> float:
        return float(self._sims_semantico(texto).max())

    def _sims_semantico(self, texto: str) -> "np.ndarray":
        vec = self._st_model.encode(
            [texto],
            normalize_embeddings=True,
            device=_DEVICE,
            show_progress_bar=False,
        )
        # dot product entre vetores normalizados = cosine similarity
        return (vec @ self._mat_sem.T)[0]

    @classmethod
    def carregar_ou_construir(
        cls,
        portfolio_path: Path | str,
        backend: str = "auto",
        modelo: str = MODELO_SEMANTICO_PADRAO,
    ) -> "IndicePortfolio":
        """Ponto de entrada conveniente para produção e integração."""
        path = Path(portfolio_path)
        with open(path, encoding="utf-8") as f:
            portfolio = json.load(f)
        inst = cls()
        inst.construir(portfolio, backend=backend, modelo=modelo, portfolio_path=path)
        return inst

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def modelo(self) -> str:
        return self._modelo

    @property
    def n_produtos(self) -> int:
        return self._n


# ══════════════════════════════════════════════════════════════════════════════
# Carregamento de itens LLM (ground truth para o estudo)
# ══════════════════════════════════════════════════════════════════════════════

def _carregar_itens_json(pasta: Path) -> dict[int, str]:
    """Carrega itens.json → dict {numeroItem: descricao}."""
    path = pasta / "itens.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        lista = data.get("itens", []) if isinstance(data, dict) else data
        return {
            int(i.get("numeroItem", 0)): (i.get("descricao") or "").strip()
            for i in lista if isinstance(i, dict) and i.get("numeroItem")
        }
    except Exception:
        return {}


def _carregar_itens_llm(editais_dir: Path) -> list[dict]:
    """
    Carrega itens analisados pelo LLM (exclui meepp e pre_filtro).
    Trata dois schemas de aderencia.json:
      Schema A (pré-filtro): item tem "descricao"
      Schema B (LLM atual):  item não tem "descricao" → busca em itens.json
    """
    itens: list[dict] = []
    pastas = [p for p in editais_dir.iterdir()
              if p.is_dir() and (p / "aderencia.json").exists()]

    for pasta in pastas:
        try:
            with open(pasta / "aderencia.json", encoding="utf-8") as f:
                ad = json.load(f)
        except Exception:
            continue

        if ad.get("_meepp") or ad.get("_pre_filtrado"):
            continue
        provedor = ad.get("_provedor") or ad.get("provedor") or ""
        if provedor == "pre_filtro":
            continue

        processo = ad.get("_processo", pasta.name)
        _itens_json: dict[int, str] | None = None

        for item in (ad.get("itens_analisados") or []):
            cls = item.get("classificacao", "").strip()
            num = item.get("numeroItem")
            if not cls:
                continue

            desc = (item.get("descricao") or "").strip()
            if not desc and num is not None:
                if _itens_json is None:
                    _itens_json = _carregar_itens_json(pasta)
                desc = _itens_json.get(int(num), "")

            if not desc:
                continue

            obs = (item.get("observacao") or "").lower()
            if "pré-filtro heurístico" in obs or "classificado por pré-filtro" in obs:
                continue

            itens.append({
                "descricao":      desc,
                "classificacao":  cls,
                "processo":       processo,
                "grau_confianca": item.get("grau_confianca"),
            })

    return itens


# ══════════════════════════════════════════════════════════════════════════════
# Estudo de viabilidade
# ══════════════════════════════════════════════════════════════════════════════

def _grupo(cls: str) -> str:
    if cls in CRITICOS:   return "CRITICO"
    if cls in BORDERLINE: return "BORDERLINE"
    if cls in NEGATIVOS:  return "NEGATIVO"
    return "OUTRO"


def estudo_viabilidade(indice: IndicePortfolio, editais_dir: Path) -> None:
    print("\nCarregando itens LLM como ground truth...")
    itens = _carregar_itens_llm(editais_dir)

    if not itens:
        print(
            "\n[!] Nenhum item LLM encontrado.\n"
            "    Execute o pipeline completo (pncp_agente.py) e rode novamente.\n"
            "    O estudo precisa de aderencia.json gerados pelo LLM."
        )
        return

    print(f"  {len(itens):,} itens. Calculando scores com {indice.backend} ({indice.modelo})...")
    for item in itens:
        item["score"] = indice.score_maximo(item["descricao"])

    from collections import Counter
    dist_cls = Counter(item["classificacao"] for item in itens)
    dist_grp = Counter(_grupo(item["classificacao"]) for item in itens)

    linhas: list[str] = []
    linhas += [
        "=" * 72,
        "ESTUDO DE VIABILIDADE — FILTRO POR SIMILARIDADE",
        f"Backend : {indice.backend.upper()}",
        f"Modelo  : {indice.modelo}",
        f"Device  : {_DEVICE.upper()}",
        "Posição : APÓS pré-filtro léxico, ANTES do LLM",
        "=" * 72,
        "",
        f"Total de itens analisados pelo LLM: {len(itens):,}",
        "",
        "Ground truth (classificação pelo LLM):",
    ]
    max_cnt = max(dist_cls.values()) if dist_cls else 1
    for cls, cnt in sorted(dist_cls.items(), key=lambda x: -x[1]):
        grp = _grupo(cls)
        bar = "█" * (cnt * 30 // max_cnt)
        linhas.append(f"  {cls:32s} {cnt:5,}  [{grp}]  {bar}")

    linhas += [
        "",
        f"  CRITICO    (direta + parcial_forte) : {dist_grp['CRITICO']:,}",
        f"  BORDERLINE (parcial_fraca)           : {dist_grp['BORDERLINE']:,}",
        f"  NEGATIVO   (falso_positivo + nao_ad.): {dist_grp['NEGATIVO']:,}",
    ]

    criticos    = [i for i in itens if _grupo(i["classificacao"]) == "CRITICO"]
    borderlines = [i for i in itens if _grupo(i["classificacao"]) == "BORDERLINE"]
    negativos   = [i for i in itens if _grupo(i["classificacao"]) == "NEGATIVO"]

    def _mediana(lst: list[float]) -> float:
        s = sorted(lst)
        return s[len(s) // 2] if s else 0.0

    if criticos:
        sc = [i["score"] for i in criticos]
        linhas += [
            "",
            "Distribuição de scores por grupo:",
            f"  CRITICO    min={min(sc):.3f}  med={_mediana(sc):.3f}  max={max(sc):.3f}",
        ]
    if borderlines:
        sb = [i["score"] for i in borderlines]
        linhas.append(
            f"  BORDERLINE min={min(sb):.3f}  med={_mediana(sb):.3f}  max={max(sb):.3f}"
        )
    if negativos:
        sn = [i["score"] for i in negativos]
        linhas.append(
            f"  NEGATIVO   min={min(sn):.3f}  med={_mediana(sn):.3f}  max={max(sn):.3f}"
        )

    # Tabela
    linhas += [
        "",
        "-" * 72,
        "ANÁLISE POR THRESHOLD  (score < threshold → filtrado, não chega ao LLM)",
        "-" * 72,
        f"{'Thresh':>7}  {'%Crit':>7}  {'%Bord':>7}  {'%Neg_filt':>10}  "
        f"{'Economia':>8}  {'FN_crit':>7}",
        "-" * 72,
    ]

    melhor_T        = None
    melhor_economia = 0.0

    for T in THRESHOLDS_ESTUDO:
        n_crit_pass = sum(1 for i in criticos    if i["score"] >= T)
        n_bord_pass = sum(1 for i in borderlines if i["score"] >= T)
        n_neg_filt  = sum(1 for i in negativos   if i["score"] <  T)
        n_total_pass= sum(1 for i in itens        if i["score"] >= T)
        fn_crit     = len(criticos) - n_crit_pass

        pct_crit = 100.0 * n_crit_pass / len(criticos)    if criticos    else 100.0
        pct_bord = 100.0 * n_bord_pass / len(borderlines) if borderlines else 100.0
        pct_neg  = 100.0 * n_neg_filt  / len(negativos)   if negativos   else 0.0
        economia = 100.0 * (len(itens) - n_total_pass) / len(itens)

        marcador = " <--" if fn_crit == 0 and economia > melhor_economia else ""
        if fn_crit == 0 and economia > melhor_economia:
            melhor_T        = T
            melhor_economia = economia

        linhas.append(
            f"  {T:5.2f}   {pct_crit:>6.1f}%  {pct_bord:>6.1f}%  {pct_neg:>9.1f}%  "
            f"{economia:>7.1f}%  {fn_crit:>7}{marcador}"
        )

    linhas += [
        "-" * 72,
        "",
        "LEGENDA:",
        "  %Crit     = % de aderencia_direta/parcial_forte mantidos  (meta: 100%)",
        "  %Bord     = % de aderencia_parcial_fraca mantidos",
        "  %Neg_filt = % de nao_aderente/falso_positivo filtrados    (meta: maximo)",
        "  Economia  = % de todos os itens que nao chegam ao LLM",
        "  FN_crit   = falsos negativos criticos (aderentes perdidos)",
        "  <-- melhor threshold sem FN criticos",
    ]

    # Itens críticos com menor score
    linhas += [
        "",
        "-" * 72,
        "ITENS CRITICOS COM MENOR SCORE (risco de falso negativo):",
        "-" * 72,
    ]
    for item in sorted(criticos, key=lambda x: x["score"])[:15]:
        linhas.append(
            f"  [{item['score']:.3f}]  {item['classificacao']:24s}  "
            f"{item['descricao'][:65]}"
        )
    if not criticos:
        linhas.append(
            "  (nenhum item CRITICO no dataset — rodar pipeline completo para calibrar)"
        )

    # Falsos positivos com score alto
    fpl = [i for i in itens if i["classificacao"] in
           {"FALSO_POSITIVO_LEXICAL", "falso_positivo_lexical"}]
    if fpl:
        linhas += [
            "",
            "-" * 72,
            "FALSOS POSITIVOS LEXICAIS COM MAIOR SCORE",
            "(passariam pelo filtro — LLM ainda necessario para descarta-los):",
            "-" * 72,
        ]
        for item in sorted(fpl, key=lambda x: -x["score"])[:15]:
            linhas.append(f"  [{item['score']:.3f}]  {item['descricao'][:75]}")

    # Recomendação
    linhas += ["", "=" * 72, "RECOMENDACAO", "=" * 72]

    if not criticos:
        linhas += [
            "",
            "[!] AVISO: nenhum item CRITICO no dataset atual.",
            "    O threshold recomendado abaixo baseia-se apenas em NEGATIVOS e BORDERLINES.",
            "    Execute o pipeline completo e re-rode para calibracao definitiva.",
        ]

    if melhor_T is not None:
        n_pass = sum(1 for i in itens if i["score"] >= melhor_T)
        linhas += [
            f"",
            f"Threshold recomendado : {melhor_T:.2f}",
            f"  -> {melhor_economia:.1f}% dos itens filtrados sem perder criticos",
            f"  -> {n_pass:,} de {len(itens):,} itens chegariam ao LLM",
            "",
            f"Backend utilizado : {indice.backend.upper()} ({indice.modelo})",
        ]
        if indice.backend == "tfidf":
            linhas += [
                "",
                "NOTA: TF-IDF e aproximacao lexica — nao captura similaridade semantica.",
                "  Para melhor precisao, use backend semantic:",
                "    pip install sentence-transformers",
                "    python pncp_embeddings.py --backend semantic",
            ]
        linhas += [
            "",
            "PROXIMOS PASSOS:",
            f"  1. Rodar pipeline completo (pncp_agente.py) para obter itens CRITICOS",
            f"  2. Re-executar este estudo para calibracao definitiva",
            f"  3. Integrar IndicePortfolio em pncp_agente.py com threshold={melhor_T:.2f}",
        ]
    else:
        linhas += [
            "",
            "[!] Nenhum threshold elimina 100% dos falsos negativos criticos.",
            "    Considere usar um modelo semantico de maior qualidade.",
        ]

    relatorio = "\n".join(linhas)
    RELATORIO_FILE.write_text(relatorio, encoding="utf-8")
    print(relatorio)
    print(f"\n-> Relatorio salvo em {RELATORIO_FILE}")


# ══════════════════════════════════════════════════════════════════════════════
# Modo interativo — teste de item único
# ══════════════════════════════════════════════════════════════════════════════

def _testar_item(indice: IndicePortfolio, descricao: str) -> None:
    score = indice.score_maximo(descricao)
    top   = indice.top_k(descricao, k=5)
    print(f"\nItem     : '{descricao}'")
    print(f"Backend  : {indice.backend} ({indice.modelo})")
    print(f"Device   : {_DEVICE.upper()}")
    print(f"Score max: {score:.4f}")
    print("Top 5 produtos similares:")
    for s, lbl in top:
        print(f"  [{s:.4f}]  {lbl}")


# ══════════════════════════════════════════════════════════════════════════════
# Ponto de entrada
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="Estudo de viabilidade — filtro por similaridade"
    )
    p.add_argument("--portfolio",  default=str(PORTFOLIO_FILE))
    p.add_argument("--editais",    default=str(EDITAIS_DIR))
    p.add_argument("--backend",    default=BACKEND_PADRAO,
                   choices=["auto", "semantic", "tfidf"],
                   help="Backend de similaridade (default: auto)")
    p.add_argument("--modelo",     default=MODELO_SEMANTICO_PADRAO,
                   help="Modelo HuggingFace para backend semantic")
    p.add_argument("--threshold",  type=float, default=None,
                   help="Threshold adicional para o estudo")
    p.add_argument("--item",       default=None,
                   help="Testar item específico (modo debug)")
    args = p.parse_args()

    portfolio_path = Path(args.portfolio)
    editais_dir    = Path(args.editais)

    if args.threshold is not None and args.threshold not in THRESHOLDS_ESTUDO:
        THRESHOLDS_ESTUDO.append(args.threshold)
        THRESHOLDS_ESTUDO.sort()

    print(f"Carregando portfólio de {portfolio_path}...")
    with open(portfolio_path, encoding="utf-8") as f:
        portfolio = json.load(f)

    indice = IndicePortfolio()
    print(f"Construindo índice [{args.backend}]...")
    indice.construir(
        portfolio,
        backend=args.backend,
        modelo=args.modelo,
        portfolio_path=portfolio_path,
    )

    if args.item:
        _testar_item(indice, args.item)
        return

    estudo_viabilidade(indice, editais_dir)


if __name__ == "__main__":
    main()
