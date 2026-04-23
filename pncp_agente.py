# -*- coding: utf-8 -*-
"""
pncp_agente.py
Análise semântica de aderência de licitações ao portfólio da Resgatécnica.

Melhorias em relação à v1:
  1.  Abstração multi-LLM com Protocol (backends plugáveis: Anthropic, Ollama, OpenAI)
  2.  Validação da saída contra JSON schema (jsonschema — dependência obrigatória)
  3.  Batching por orçamento de tokens (não por contagem de itens)
  4.  objetoCompra e metadados da licitação incluídos no contexto do LLM
  5.  Pré-filtro por score ponderado (termos positivos E negativos)
  6.  Telemetria por chamada em pncp_telemetria.jsonl (tokens, latência, custo)
  7.  Retries com backoff exponencial para falhas transitórias
  8.  Auditabilidade: itens pré-filtrados recebem classificação sintética (não ficam vazios)
  9.  Pipeline em etapas explícitas: coletar → pré-filtrar → analisar → consolidar
  10. Normalização de numeroControlePNCP para nomes de pasta local ("/" → "_")
      O PNCP entrega controles com "/" (ex.: 01272081000141-1-000048/2025), mas as
      pastas em editais/ usam "_" (01272081000141-1-000048_2025). Sem essa normalização
      o script não encontra os itens.json mesmo quando eles existem.

Uso:
  pip install anthropic jsonschema    # para Anthropic
  pip install openai jsonschema       # para OpenAI
  # Ollama: instalar em https://ollama.com e rodar "ollama serve"

  set ANTHROPIC_API_KEY=sk-ant-...
  python pncp_agente.py

  # Ou via variáveis de ambiente:
  set PROVEDOR=ollama
  set OLLAMA_MODELO=qwen2.5:14b
  python pncp_agente.py
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Imports opcionais — verificados em tempo de execução
# ---------------------------------------------------------------------------
try:
    import anthropic as _ant
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import openai as _oai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# jsonschema é dependência obrigatória (pip install jsonschema)
try:
    import jsonschema as _jsc
except ImportError:
    raise ImportError(
        "jsonschema é uma dependência obrigatória.\n"
        "Execute: pip install jsonschema"
    )

import requests  # já presente (usado pelo pncp_documentos.py)

# =============================================================================
# CONFIGURAÇÃO — CAMINHOS
# =============================================================================
API_DOCS_DIR       = Path("prompts")
EDITAIS_DIR        = Path("editais")
INPUT_FILE         = Path("pncp_licitacoes.json")
OUTPUT_ADERENCIAS  = Path("pncp_aderencias.json")
OUTPUT_CONSOLIDADO = Path("pncp_itens_consolidado.json")
OUTPUT_TELEMETRIA  = Path("pncp_telemetria.jsonl")
DEBUG_LLM_DIR      = Path("debug_llm")     # respostas brutas de chamadas que falharam no parsing/schema

# =============================================================================
# CONFIGURAÇÃO — PROVEDOR
# =============================================================================
PROVEDOR = os.environ.get("PROVEDOR", "anthropic")   # "anthropic" | "ollama" | "openai"

# --- Anthropic ---
ANTHROPIC_MODELO       = os.environ.get("ANTHROPIC_MODELO", "claude-haiku-4-5")
ANTHROPIC_MAX_TOKENS   = 4096
ANTHROPIC_CACHE_PROMPT = True   # Prompt Caching: reaproveita tokens fixos entre chamadas

# --- Ollama ---
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODELO   = os.environ.get("OLLAMA_MODELO",   "llama3.1:8b")
OLLAMA_NUM_CTX  = int(os.environ.get("OLLAMA_NUM_CTX", "32768"))
# Nota: llama3.1:8b suporta até 131072 tokens; qwen2.5:14b suporta 32768.
# Com 8192 (padrão antigo do Ollama), o portfólio compacto (~10K tokens) não cabe.
OLLAMA_TIMEOUT  = int(os.environ.get("OLLAMA_TIMEOUT", "180"))   # segundos; modelos locais podem ser lentos

# --- OpenAI (e endpoints compatíveis, incl. Ollama /v1) ---
OPENAI_MODELO   = os.environ.get("OPENAI_MODELO",   "gpt-4o-mini")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", None)

# =============================================================================
# CONFIGURAÇÃO — MODO DE PROMPT
# "full"     → system prompt e schema completos (modelos capazes: Claude, GPT, Llama 70B+)
#              ATENÇÃO: "full" refere-se ao PROMPT e SCHEMA, não ao portfólio.
#              O runtime usa SEMPRE portfolio_mestre_resgatecnica_lite_v2.json (lite).
#              O portfólio full (portfolio_mestre_resgatecnica_full_v2.json) é apenas para auditoria.
# "compacto" → system prompt reduzido e schema mínimo (modelos menores: 7B-14B)
#              Portfólio enviado com apenas {cat, sub, nome} por produto (~42K chars vs 161K).
# "auto"     → escolha automática por provedor (anthropic/openai → full; ollama → compacto)
# =============================================================================
MODO = "auto"
_MODO_PADRAO = {"anthropic": "full", "openai": "full", "ollama": "compacto"}

# =============================================================================
# CONFIGURAÇÃO — ORÇAMENTO DE TOKENS
# Batching dinâmico: calcula quanto cabe no contexto depois dos tokens fixos.
# Estimativa: 1 token ≈ 4 chars (heurística conservadora para PT/EN misto).
# =============================================================================
CHARS_POR_TOKEN    = 4       # estimativa de conversão chars → tokens
MARGEM_TOKENS      = 500     # margem de segurança extra
MAX_OUTPUT_TOKENS  = {"anthropic": 8192, "openai": 4096, "ollama": 3072}
# ollama: 3072 em vez de 2048 — processo 46316600000164-1-000282/2025 provou que
# licitações com descrições longas podem exceder 2048 em lotes de 6 itens,
# forçando split recursivo desnecessário. 3072 cobre lotes de até ~15 itens.
CONTEXT_WINDOW     = {"anthropic": 200_000, "openai": 128_000}  # Ollama usa OLLAMA_NUM_CTX

# =============================================================================
# CONFIGURAÇÃO — PRÉ-FILTRO MELHORADO (score ponderado + NCM)
#
# Lógica de decisão:
#   score > 0                                   → CONTINUAR (sinal positivo detectado)
#   score <= -LIMIAR_DESCARTAR                  → DESCARTAR (sinal negativo forte)
#   nenhum positivo E ao menos 1 negativo       → DESCARTAR (sem sinal a favor)
#   resto (score 0, sem positivos, sem negativos) → AMBIGUO (LLM com cautela)
#
# NCM whitelist: itens com ncmNbsCodigo iniciando em prefixo da lista recebem
# bônus +1 no score (sinal estrutural independente do texto descritivo).
# =============================================================================
USAR_PRE_FILTRO   = True
LIMIAR_DESCARTAR  = 1        # score negativo para descartar sem LLM (era 2)

# =============================================================================
# CONFIGURAÇÃO — FILTRO ESTRUTURAL ME/EPP
#
# A Resgatécnica não é ME nem EPP, portanto não pode participar de itens com:
#   tipoBeneficio = 1  →  Participação exclusiva para ME/EPP
#   tipoBeneficio = 3  →  Cota reservada para ME/EPP
#
# Apenas itens com os valores abaixo são analisados:
#   tipoBeneficio = 4  →  Sem benefício (aberto a qualquer empresa)
#   tipoBeneficio = 5  →  Não se aplica
#
# Impacto medido nos dados locais (578 licitações / 9.171 itens):
#   - 123 licitações (21%) têm 100% de itens ME/EPP → puladas inteiramente
#   -  82 licitações (14%) têm mix → itens ME/EPP removidos antes do LLM
#   - 39,1% dos itens no total são inelegíveis para a Resgatécnica
# =============================================================================
TIPOS_BENEFICIO_ELEGIVEIS = {4, 5}

# =============================================================================
# CONFIGURAÇÃO — FILTRO DE PRAZO MÍNIMO (agente)
#
# Mesmo critério usado pelo pncp_scanner.py: descarta licitações cujo
# dataEncerramentoProposta já expirou ou é inferior a DIAS_MINIMOS_PREPARO
# dias a partir de hoje. Necessário porque pncp_licitacoes.json pode ser
# executado dias/semanas após o scanner ter gerado o arquivo.
#
# Referência: tabela 5.5 do manual — Situação da Contratação:
#   1 = Divulgada (ativa) | 2 = Revogada | 3 = Anulada | 4 = Suspensa
# =============================================================================
DIAS_MINIMOS_PREPARO = 7    # dias mínimos até encerramento (espelha o scanner)
SITUACOES_INVALIDAS  = {2, 3, 4}   # revogada, anulada, suspensa

def _normalizar(texto: str) -> str:
    """Remove acentos e converte para lowercase. Usado no pré-filtro."""
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()

# Termos definidos com acentos para facilitar manutenção;
# são pré-normalizados em _TERMOS_POSITIVOS / _TERMOS_NEGATIVOS no load.
_TERMOS_POSITIVOS_RAW = {        # contextos do nicho da Resgatécnica (+1 cada)
    # ── Geral / resgate ──────────────────────────────────────────────────────
    "resgate", "salvamento", "emergência", "socorro",
    "aph", "trabalho em altura", "espaço confinado", "mergulho operacional",
    "desencarceramento", "desencarcerador",
    "descontaminação", "produtos perigosos", "hazmat",
    "equipamento de proteção individual técnico",
    "epi técnico", "equipamento tático",
    # ── Combate a incêndio ───────────────────────────────────────────────────
    "combate a incêndio", "incêndio",
    "motobomba", "motobombas", "bomba de incêndio", "bomba de combate",
    "extintor", "extintores",
    "mangote", "mangotinho", "mangueira de incêndio", "mangueiras de incêndio",
    "esguicho", "lance de mangueira",
    "roupa de aproximação", "roupa de combate a incêndio",
    "máscara de fuga", "capacete de bombeiro", "capacetes de bombeiro",
    "exaustão de fumaça", "ventilação de incêndio",
    "hidrante", "hidrantes", "chave de hidrante",
    # ── Ambulâncias / APH ────────────────────────────────────────────────────
    "ambulância", "ambulâncias",
    "uti móvel", "utm", "usv", "utm-b", "utm-a",
    "desfibrilador", "desfibriladores", "dea", "dsa", "monitor cardíaco",
    "oxímetro portátil", "capnógrafo",
    "laringoscópio", "kit de intubação",
    "prancha de imobilização", "colar cervical",
    # ── Macas e transporte de vítimas ────────────────────────────────────────
    "maca", "macas", "maca de resgate", "cadeira de evacuação",
    # ── Viaturas ─────────────────────────────────────────────────────────────
    "viatura", "viaturas", "pick-up", "caminhonete policial",
    "viatura 4x4", "viatura pickup",
    # ── Coletes balísticos ───────────────────────────────────────────────────
    "colete balístico", "coletes balísticos", "colete à prova de bala",
    "proteção balística", "blindagem pessoal",
    "nível iiia", "nível iv", "nij",
    "placa balística", "placas balísticas",
    # ── Capacetes e EPI tático ───────────────────────────────────────────────
    "capacete de resgate", "capacetes de resgate",
    "capacete tático", "capacete balístico",
    "luva de resgate", "luvas de resgate", "bota de combate",
    "óculos balístico", "joelheira tática",
    # ── Ferramentas hidráulicas de resgate ───────────────────────────────────
    "mandíbulas de vida", "ferramenta hidráulica de resgate",
    "ferramentas hidráulicas de resgate",
    "alicate hidráulico de resgate", "espalhador hidráulico",
    "cortador hidráulico de resgate", "cilindro de força",
    # ── Outras ───────────────────────────────────────────────────────────────
    "corda de resgate", "cordas de resgate", "detector de vítimas",
    "bomba hidráulica de resgate",
}

_TERMOS_NEGATIVOS_RAW = {        # contextos claramente fora do nicho (-1 cada)
    # ── Educação / material escolar ──────────────────────────────────────────
    "papelaria", "material escolar", "material de escritório", "expediente",
    "vestuário escolar", "uniforme escolar",
    "material gráfico", "artes gráficas", "livros didáticos",
    "mobiliário escolar", "carteira escolar",
    # ── Alimentação ──────────────────────────────────────────────────────────
    "gêneros alimentícios", "alimentos", "merenda", "alimentação escolar",
    "refeição", "marmita", "gênero alimentício",
    "copa e cozinha",
    # ── Saúde / farmácia ─────────────────────────────────────────────────────
    "medicamento", "fármaco", "farmacêutico",
    "gases medicinais", "oxigênio medicinal", "gás medicinal",
    "odontológic", "dental", "material odontológico",
    "material de limpeza doméstica",
    # ── Agropecuária ─────────────────────────────────────────────────────────
    "agrotóxico", "fertilizante", "insumo agrícola",
    "semente", "mudas agrícolas", "calcário agrícola",
    # ── Tecnologia da informação ─────────────────────────────────────────────
    "licença de software", "software de gestão",
    "desenvolvimento de sistema", "sistema informatizado",
    "link de dados", "fibra óptica",
    "suporte de ti", "serviço de ti", "banco de dados", "serviço de nuvem",
    # ── Obras / construção ────────────────────────────────────────────────────
    "reforma predial", "obra de engenharia", "construção civil",
    "pavimentação", "asfalto",
    # ── Higiene e limpeza ────────────────────────────────────────────────────
    "produto de higiene", "material de higiene", "saneante",
    "desinfetante", "sabonete", "serviço de limpeza",
    "higienização de ambientes", "dedetização",
    # ── Esportivo ────────────────────────────────────────────────────────────
    "equipamento esportivo", "material esportivo", "uniforme esportivo",
    "quadra esportiva", "kit esportivo",
    # ── Mobiliário genérico ──────────────────────────────────────────────────
    "mobiliário de escritório", "mesa e cadeira", "estante de aço",
}

# NCM whitelist: prefixos (4 ou 6 dígitos) de produtos Resgatécnica.
# Confere bônus +1 no score quando qualquer item da licitação os apresenta.
# Cobertura atual: ~7,7 % dos itens — sinal fraco mas livre de falsos positivos.
_NCM_POSITIVOS_PREFIXOS = {
    "8424",  # Extintores de incêndio e aparelhos de dispersão de líquidos
    "8705",  # Veículos automóveis para fins especiais (ambulâncias, viaturas)
    "8413",  # Bombas para líquidos (motobombas de combate a incêndio)
    "6211",  # Macacões e roupas de proteção (roupa de aproximação)
    "6506",  # Capacetes de proteção
    "9019",  # Aparelhos de mecanoterapia (desfibriladores portáteis)
}

# Pré-compila padrões regex com word boundary (\b) para evitar matches de
# substring (ex.: "nij" dentro de "unijato", "dea" dentro de "ideal",
# "maca" dentro de "automacao"). _normalizar remove acentos, então \b
# funciona corretamente sobre texto ASCII-lowercase.
#
# Estrutura: lista de tuplas (termo_normalizado, re.Pattern) para que
# pre_filtrar possa retornar os termos encontrados legíveis no log.
def _compilar_padroes_prefiltro(
    termos_raw: set[str],
) -> list[tuple[str, re.Pattern]]:
    resultado = []
    for t in termos_raw:
        t_norm = _normalizar(t)
        pat = re.compile(r"\b" + re.escape(t_norm) + r"\b")
        resultado.append((t_norm, pat))
    return resultado

_PADROES_POSITIVOS: list[tuple[str, re.Pattern]] = _compilar_padroes_prefiltro(_TERMOS_POSITIVOS_RAW)
_PADROES_NEGATIVOS: list[tuple[str, re.Pattern]] = _compilar_padroes_prefiltro(_TERMOS_NEGATIVOS_RAW)

# Sets mantidos apenas para retrocompatibilidade com código que os referencie
_TERMOS_POSITIVOS = {t for t, _ in _PADROES_POSITIVOS}
_TERMOS_NEGATIVOS = {t for t, _ in _PADROES_NEGATIVOS}

# =============================================================================
# CONFIGURAÇÃO — PROCESSAMENTO
# =============================================================================
INCREMENTAL           = True   # True = pular licitações que já têm aderencia.json
PAUSA_ENTRE_CHAMADAS  = 0.5   # segundos de pausa entre chamadas à API
MAX_PROCESSOS = 100 #      = int(os.environ.get("MAX_PROCESSOS", "0"))
# MAX_PROCESSOS > 0 limita o número de licitações analisadas (útil para testes).
# Exemplo: set MAX_PROCESSOS=10 && python pncp_agente.py

MAX_ITENS_POR_LOTE    = int(os.environ.get("MAX_ITENS_POR_LOTE", "0"))
# MAX_ITENS_POR_LOTE > 0 força um limite de itens por chamada, independente do budget.
# 0 = usa o default efetivo por provedor definido em _MAX_ITENS_DEFAULT_PROVEDOR.
# Anthropic/OpenAI: default 25 (protege contra truncamento — output limitado a 8192 tokens).
# Ollama: default 20 (protege contra timeout de inferência em hardware local).
# Override explícito via env var aplica-se a todos os provedores.
_MAX_ITENS_DEFAULT_PROVEDOR: dict[str, int] = {
    "ollama":    20,   # protege timeout de inferência em hardware local
    "anthropic": 25,   # protege truncamento de output (8192 tok_out)
    "openai":    25,   # idem
}

# =============================================================================
# CONFIGURAÇÃO — RETRIES
# =============================================================================
MAX_TENTATIVAS       = 3
DELAY_BASE_S         = 2.0   # delay inicial; duplica a cada tentativa
JITTER_MAX_S         = 1.0   # jitter aleatório para evitar thundering herd

# =============================================================================
# CONFIGURAÇÃO — FALLBACK DE SPLIT EM CASO DE FALHA DE PARSING
# Quando um lote falha por _extrair_json/SchemaValidationError, o pipeline tenta
# dividir o lote ao meio e reprocessar cada metade (até MAX_PROFUNDIDADE_SPLIT
# níveis). Resgata licitações grandes que estourariam max_tokens em chamada única.
# =============================================================================
USAR_SPLIT_AUTOMATICO     = True
MAX_PROFUNDIDADE_SPLIT    = 2   # 2 = até 4 sublotes a partir do lote original

# =============================================================================
# CONFIGURAÇÃO — ESTIMATIVA DE CUSTO (USD por MTok)
# =============================================================================
_CUSTO_POR_MTOK: dict[str, dict[str, float]] = {
    "claude-haiku-4-5":          {"in": 0.80,  "out": 4.00,  "cache_r": 0.08,  "cache_w": 1.00},
    "claude-3-haiku-20240307":   {"in": 0.25,  "out": 1.25,  "cache_r": 0.03,  "cache_w": 0.30},
    "claude-3-5-sonnet-20241022":{"in": 3.00,  "out": 15.00, "cache_r": 0.30,  "cache_w": 3.75},
    "gpt-4o-mini":               {"in": 0.15,  "out": 0.60},
    "gpt-4o":                    {"in": 2.50,  "out": 10.00},
}

# Arquivos do pacote de implantação
_ARQUIVOS = {
    "sys_full":        "system_prompt_resgatecnica_full.txt",
    "sys_compacto":    "system_prompt_resgatecnica_local_compacto.txt",
    "usr_full":        "user_template_resgatecnica_full.txt",
    "usr_compacto":    "user_template_resgatecnica_local_compacto.txt",
    "portfolio":       "portfolio_mestre_resgatecnica_lite_v2.json",
    "schema_full":     "output_schema_resgatecnica_full.json",
    "schema_compacto": "output_schema_resgatecnica_local_compacto.json",
}

# Campos de itens enviados ao LLM
_CAMPOS_PRINCIPAIS = ["numeroItem", "descricao"]
_CAMPOS_APOIO      = ["unidadeMedida", "informacaoComplementar", "ncmNbsCodigo", "materialOuServicoNome"]

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class LLMRequest:
    """Requisição genérica para qualquer backend LLM."""
    system_prompt: str
    user_prompt: str            # mensagem completa do usuário
    temperature: float = 0.1
    max_tokens: int = 4096
    json_schema: dict | None = None   # structured output (Ollama, OpenAI)
    timeout_s: int = 120
    # Dicas de caching (só Anthropic as usa; outros backends ignoram)
    cache_system: bool = False
    cache_user_prefix_len: int = 0    # chars do início de user_prompt a cachear

@dataclass
class LLMResponse:
    """Resposta normalizada de qualquer backend LLM."""
    text: str
    parsed_json: dict | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    latency_ms: int | None = None
    raw: Any = None
    error: str | None = None

@dataclass
class ProcessoInput:
    """Dados de uma licitação prontos para análise."""
    ctrl: str          # numeroControlePNCP cru (pode conter "/")
    objeto_compra: str
    orgao: str
    uf: str
    valor_estimado: float | None
    modalidade: str
    itens: list[dict]
    metadados: dict = field(default_factory=dict)

@dataclass
class PreFiltroResultado:
    """Resultado do pré-filtro heurístico."""
    decisao: str          # "CONTINUAR" | "AMBIGUO" | "DESCARTAR"
    score: float
    termos_positivos: list[str]
    termos_negativos: list[str]

@dataclass
class EntradaTelemetria:
    """Uma linha de telemetria (gravada em pncp_telemetria.jsonl)."""
    timestamp: str
    processo: str
    provedor: str
    modelo: str
    modo: str
    lote_idx: int
    total_lotes: int
    n_itens_lote: int
    input_tokens: int | None
    output_tokens: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    latency_ms: int | None
    custo_usd: float | None
    schema_valido: bool
    status: str           # "ok" | "schema_invalido" | "erro_llm" | "pre_filtrado"
    erro: str | None
    # path_id identifica sublotes gerados pelo split recursivo. Ex.: "1" (lote raiz),
    # "1.1" / "1.2" (split nível 1), "1.2.1" / "1.2.2" (split nível 2). Quando None,
    # o campo "lote" da telemetria cai no formato clássico "{lote_idx}/{total_lotes}".
    path_id: str | None = None

# =============================================================================
# ABSTRAÇÃO LLM — Protocol + Backends
# =============================================================================

@runtime_checkable
class LLMBackend(Protocol):
    """Interface que todo backend LLM deve implementar."""
    @property
    def provider_name(self) -> str: ...
    @property
    def model_name(self) -> str: ...
    def generate(self, req: LLMRequest) -> LLMResponse: ...


class AnthropicBackend:
    """Backend para Claude via API Anthropic, com suporte a Prompt Caching."""
    provider_name = "anthropic"

    def __init__(self):
        if not HAS_ANTHROPIC:
            raise RuntimeError("pip install anthropic")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY não definida no ambiente.")
        self._client = _ant.Anthropic(api_key=api_key)

    @property
    def model_name(self) -> str:
        return ANTHROPIC_MODELO

    def generate(self, req: LLMRequest) -> LLMResponse:
        t0 = time.monotonic()

        # System prompt — com ou sem cache
        if req.cache_system:
            system = [{"type": "text", "text": req.system_prompt,
                       "cache_control": {"type": "ephemeral"}}]
        else:
            system = req.system_prompt

        # User message — divide em prefixo cacheável e sufixo variável
        if req.cache_user_prefix_len > 0:
            prefix = req.user_prompt[:req.cache_user_prefix_len]
            suffix = req.user_prompt[req.cache_user_prefix_len:]
            user_content = [
                {"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": suffix},
            ]
        else:
            user_content = req.user_prompt

        try:
            resp = self._client.messages.create(
                model=ANTHROPIC_MODELO,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                system=system,
                messages=[{"role": "user", "content": user_content}],
                timeout=req.timeout_s,
            )
        except Exception:
            # Fallback: tenta sem cache se houve erro (ex.: modelo sem suporte)
            if req.cache_user_prefix_len > 0 or req.cache_system:
                log.warning("Prompt caching falhou, tentando sem cache...")
                resp = self._client.messages.create(
                    model=ANTHROPIC_MODELO,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    system=req.system_prompt,
                    messages=[{"role": "user", "content": req.user_prompt}],
                    timeout=req.timeout_s,
                )
            else:
                raise

        uso = resp.usage
        lat = int((time.monotonic() - t0) * 1000)
        return LLMResponse(
            text=resp.content[0].text,
            input_tokens=uso.input_tokens,
            output_tokens=uso.output_tokens,
            cache_read_tokens=getattr(uso, "cache_read_input_tokens", None),
            cache_write_tokens=getattr(uso, "cache_creation_input_tokens", None),
            latency_ms=lat,
            raw=resp,
        )


class OllamaBackend:
    """Backend para modelos locais via Ollama (/api/chat com structured output)."""
    provider_name = "ollama"

    @property
    def model_name(self) -> str:
        return OLLAMA_MODELO

    def generate(self, req: LLMRequest) -> LLMResponse:
        t0 = time.monotonic()
        url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
        payload: dict[str, Any] = {
            "model":  OLLAMA_MODELO,
            "stream": False,
            "messages": [
                {"role": "system", "content": req.system_prompt},
                {"role": "user",   "content": req.user_prompt},
            ],
            "options": {
                "temperature": req.temperature,
                "num_ctx":     OLLAMA_NUM_CTX,
                "num_predict": req.max_tokens,   # limita geração; sem isso o modelo pode gerar até encher o contexto (~21 min a 13 tok/s)
            },
        }
        if req.json_schema:
            payload["format"] = req.json_schema   # structured output

        try:
            r = requests.post(url, json=payload, timeout=req.timeout_s)
            r.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Ollama inacessível em {OLLAMA_BASE_URL}. Execute: ollama serve"
            )

        data = r.json()
        lat  = int((time.monotonic() - t0) * 1000)
        return LLMResponse(
            text=data.get("message", {}).get("content", ""),
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            latency_ms=lat,
            raw=data,
        )


class OpenAIBackend:
    """Backend para GPT via API OpenAI ou qualquer endpoint compatível."""
    provider_name = "openai"

    def __init__(self):
        if not HAS_OPENAI:
            raise RuntimeError("pip install openai")
        api_key = os.environ.get("OPENAI_API_KEY", "none")
        kwargs: dict[str, Any] = {"api_key": api_key}
        if OPENAI_BASE_URL:
            kwargs["base_url"] = OPENAI_BASE_URL
        self._client = _oai.OpenAI(**kwargs)

    @property
    def model_name(self) -> str:
        return OPENAI_MODELO

    def generate(self, req: LLMRequest) -> LLMResponse:
        t0 = time.monotonic()
        msgs = [
            {"role": "system", "content": req.system_prompt},
            {"role": "user",   "content": req.user_prompt},
        ]
        base_kwargs: dict[str, Any] = {
            "model":       OPENAI_MODELO,
            "temperature": req.temperature,
            "max_tokens":  req.max_tokens,
            "messages":    msgs,
        }

        resp = None
        # Tenta structured output com json_schema (gpt-4o, gpt-4o-mini suportam)
        # Endpoint Ollama /v1 também suporta via campo format.
        if req.json_schema:
            try:
                resp = self._client.chat.completions.create(
                    **base_kwargs,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name":   "triagem_aderencia_resgatecnica",
                            "schema": req.json_schema,
                            "strict": False,   # strict=True pode rejeitar schemas complexos
                        },
                    },
                    timeout=req.timeout_s,
                )
            except Exception:
                resp = None  # endpoint não suporta → usa fallback

        if resp is None:
            # Fallback: json_object (garante JSON, mas sem validação de schema pelo endpoint)
            resp = self._client.chat.completions.create(
                **base_kwargs,
                response_format={"type": "json_object"},
                timeout=req.timeout_s,
            )

        uso = resp.usage
        lat = int((time.monotonic() - t0) * 1000)
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            input_tokens=uso.prompt_tokens     if uso else None,
            output_tokens=uso.completion_tokens if uso else None,
            latency_ms=lat,
            raw=resp,
        )


def get_backend() -> LLMBackend:
    """Factory: retorna o backend configurado para o provedor ativo."""
    if PROVEDOR == "anthropic":
        return AnthropicBackend()
    if PROVEDOR == "ollama":
        return OllamaBackend()
    if PROVEDOR == "openai":
        return OpenAIBackend()
    raise ValueError(f"Provedor desconhecido: '{PROVEDOR}'. Use: anthropic | ollama | openai")

# =============================================================================
# ESTADO GLOBAL DO PACOTE — carregado uma vez em carregar_pacote()
# =============================================================================
_modo_efetivo: str = ""
_system_prompt: str = ""
_schema_validacao: dict = {}
_schema_ollama: dict = {}          # schema para structured output do Ollama
_portfolio_str_full: str = ""      # portfólio completo (modo full)
_portfolio_str_compacto: str = ""  # portfólio mínimo: cat + sub + nome (modo compacto)
_template_antes_itens: str = ""    # prefixo cacheável (sistema + portfólio substituído)
_template_apos_itens: str = ""     # sufixo com INSTRUÇÕES ADICIONAIS
_tokens_fixos: int = 0             # estimativa de tokens fixos por chamada (sistema + portfólio)
_tokens_variavel_fixo: int = 0     # tokens variáveis não-item por chamada ([CONTEXTO] + instruções)

# Exceção para erros de schema (não retentados)
class SchemaValidationError(Exception):
    pass

# =============================================================================
# CARREGAMENTO DO PACOTE
# =============================================================================

def _ler_txt(nome: str) -> str:
    p = API_DOCS_DIR / nome
    if not p.exists():
        raise FileNotFoundError(f"Arquivo do pacote não encontrado: {p}")
    return p.read_text(encoding="utf-8")

def _ler_json(nome: str) -> Any:
    p = API_DOCS_DIR / nome
    if not p.exists():
        raise FileNotFoundError(f"Arquivo do pacote não encontrado: {p}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def carregar_pacote():
    """Carrega prompts, portfólio e schemas do pacote de implantação (uma vez)."""
    global _modo_efetivo, _system_prompt, _schema_validacao, _schema_ollama
    global _portfolio_str_full, _portfolio_str_compacto
    global _template_antes_itens, _template_apos_itens, _tokens_fixos, _tokens_variavel_fixo

    _modo_efetivo = MODO if MODO in ("full", "compacto") else _MODO_PADRAO.get(PROVEDOR, "full")

    # System prompt e template de usuário
    _system_prompt = _ler_txt(_ARQUIVOS["sys_full" if _modo_efetivo == "full" else "sys_compacto"])
    user_template  = _ler_txt(_ARQUIVOS["usr_full" if _modo_efetivo == "full" else "usr_compacto"])

    # Schemas de validação
    # IMPORTANTE: _schema_ollama é SEMPRE o compacto porque o Ollama usa structured
    # output guiado por esse schema. Se MODO for forçado para "full" com Ollama,
    # o modelo receberá prompt full mas gerará saída no formato compacto — e a
    # validação usará o schema compacto para manter coerência.
    _schema_validacao = _ler_json(_ARQUIVOS["schema_full" if _modo_efetivo == "full" else "schema_compacto"])
    _schema_ollama    = _ler_json(_ARQUIVOS["schema_compacto"])  # Ollama sempre usa compacto

    # Portfólio: o arquivo base é sempre o LITE (portfolio_mestre_resgatecnica_lite_v2.json).
    # "full" aqui refere-se ao PROMPT completo, não ao portfólio.
    # O portfólio full (portfolio_mestre_resgatecnica_full_v2.json) é usado apenas
    # para manutenção e auditoria humana, nunca em chamadas de LLM.
    portfolio = _ler_json(_ARQUIVOS["portfolio"])
    _portfolio_str_full = json.dumps(portfolio, ensure_ascii=False)

    # Portfólio compacto: apenas categoria + subcategoria + nome do índice
    # Conforme fallback_rules_modelos_pequenos.md — reduz de ~161K para ~24K chars
    indice = portfolio.get("indice_lite_matching", []) if isinstance(portfolio, dict) else []
    perfil = portfolio.get("perfil_empresa_resumido", {}) if isinstance(portfolio, dict) else {}
    compacto = {
        "perfil": {
            "empresa": perfil.get("empresa", "Resgatécnica"),
            "tipo_fornecedor": perfil.get("tipo_fornecedor", ""),
            "macro_ramos_atuacao": perfil.get("macro_ramos_atuacao", []),
        },
        "indice": [
            {"cat": item["categoria_canonica"],
             "sub": item["subcategoria"],
             "nome": item["nome"]}
            for item in indice
        ]
    }
    _portfolio_str_compacto = json.dumps(compacto, ensure_ascii=False)

    # Portfólio a usar neste modo
    portfolio_str = _portfolio_str_full if _modo_efetivo == "full" else _portfolio_str_compacto

    # Pré-processa template: separa prefixo fixo (cacheável) dos itens (variável)
    # A divisão ocorre em {{ITENS_JSON}}; tudo antes é enviado ao cache.
    marcador = "{{ITENS_JSON}}"
    template_com_portfolio = user_template.replace("{{PORTFOLIO_MESTRE_LITE_JSON}}", portfolio_str)
    if marcador in template_com_portfolio:
        idx = template_com_portfolio.index(marcador)
        _template_antes_itens = template_com_portfolio[:idx]
        _template_apos_itens  = template_com_portfolio[idx + len(marcador):]
    else:
        _template_antes_itens = template_com_portfolio
        _template_apos_itens  = ""

    # Estima tokens fixos (sistema + portfólio + template header)
    _tokens_fixos = (len(_system_prompt) + len(_template_antes_itens)) // CHARS_POR_TOKEN

    # Estima tokens da parte variável que NÃO são os itens:
    # [CONTEXTO_PROCESSO] (~350 chars) + _template_apos_itens.
    # Esses tokens também consomem orçamento por chamada, mas não são itens.
    _tokens_variavel_fixo = (350 + len(_template_apos_itens)) // CHARS_POR_TOKEN

    qtd_produtos = len(indice)
    qtd_cat = len(portfolio.get("categorias_resumidas", [])) if isinstance(portfolio, dict) else "?"
    log.info(
        "Pacote carregado — provedor: %s | modo: %s | "
        "portfólio: %d produtos em %s categorias | tokens fixos est.: ~%d",
        PROVEDOR, _modo_efetivo, qtd_produtos, qtd_cat, _tokens_fixos,
    )

    # Avisa se o orçamento de tokens para itens for insuficiente.
    # Nota: _budget_itens() levanta RuntimeError se negativo.
    _budget = _budget_itens()
    if _budget < 500:
        window = OLLAMA_NUM_CTX if PROVEDOR == "ollama" else CONTEXT_WINDOW.get(PROVEDOR, 0)
        log.warning(
            "ATENÇÃO: orçamento de tokens para itens muito pequeno (~%d). "
            "O portfólio fixo (%d tokens) ocupa quase toda a janela de contexto (%d tokens). "
            "Aumente OLLAMA_NUM_CTX (ex.: 32768 ou 65536) para obter resultados adequados.",
            _budget, _tokens_fixos, window,
        )

# =============================================================================
# UTILITÁRIOS — NORMALIZAÇÃO DE CONTROLE PNCP
# =============================================================================

def normalizar_ctrl_para_pasta(ctrl: str) -> str:
    """
    Converte numeroControlePNCP para o padrão de nome de pasta local.

    O PNCP entrega o controle com barra, ex.: 01272081000141-1-000048/2025
    As pastas em editais/ foram criadas pelo pncp_documentos.py com "_", ex.:
      editais/01272081000141-1-000048_2025/

    Sem essa conversão o script não encontra os itens.json mesmo quando existem.
    """
    return str(ctrl).strip().replace("/", "_")

def desnormalizar_ctrl_de_pasta(ctrl_pasta: str) -> str:
    """
    Converte nome de pasta local de volta ao formato PNCP, quando aplicável.

    Usado ao reler aderencia.json já gravada, para que o campo _processo fique
    no formato original do PNCP (com "/"), mantendo consistência com o JSON de entrada.

    Exemplo: 01272081000141-1-000048_2025 → 01272081000141-1-000048/2025
    """
    return re.sub(r'_(\d{4})$', r'/\1', str(ctrl_pasta).strip())

# =============================================================================
# UTILITÁRIOS — ACESSO AO SISTEMA DE ARQUIVOS
# =============================================================================

def _ctrl_de_pasta(nome_pasta: str) -> str:
    """
    Extrai ctrl normalizado do nome de pasta, ignorando prefixo UF_CIDADE_.

    Exemplos:
      'MG_SETE_LAGOAS_97550393000149-1-000003_2026' → '97550393000149-1-000003_2026'
      '97550393000149-1-000003_2026'                 → '97550393000149-1-000003_2026'
    """
    m = re.search(r'(\d{14}-\d-\d+_\d{4})$', nome_pasta)
    return m.group(1) if m else nome_pasta


def _pasta_editais(ctrl: str) -> Path:
    """
    Localiza a pasta do processo em editais/, independente de prefixo UF_CIDADE_.

    Ordem de busca:
      1. Pasta legado sem prefixo: editais/{ctrl_norm}/
      2. Nova nomenclatura: editais/*_{ctrl_norm}/  (criada pelo pncp_documentos.py)
      3. Retorna caminho sem prefixo quando não encontrada (para criação futura).
    """
    ctrl_norm = normalizar_ctrl_para_pasta(ctrl)
    pasta = EDITAIS_DIR / ctrl_norm
    if pasta.exists():
        return pasta
    matches = list(EDITAIS_DIR.glob(f"*_{ctrl_norm}"))
    return matches[0] if matches else pasta


def caminho_itens(ctrl: str) -> Path:
    """Caminho para itens.json — localiza pasta independente de prefixo UF_CIDADE_."""
    return _pasta_editais(ctrl) / "itens.json"

def caminho_aderencia(ctrl: str) -> Path:
    """Caminho para aderencia.json — localiza pasta independente de prefixo UF_CIDADE_."""
    return _pasta_editais(ctrl) / "aderencia.json"

def ler_itens(ctrl: str) -> list[dict]:
    p = caminho_itens(ctrl)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []

def salvar_aderencia(ctrl: str, resultado: dict, provedor: str, modelo: str):
    resultado.setdefault("_processo", ctrl)
    resultado["_analisado_em"] = datetime.now(timezone.utc).isoformat()
    resultado["_modo"]         = _modo_efetivo
    resultado["_provedor"]     = provedor
    resultado["_modelo"]       = modelo
    with open(caminho_aderencia(ctrl), "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

_MAX_CHARS_DESCRICAO = 400   # descrições acima disso raramente acrescentam sinal útil

def _itens_para_llm(itens: list[dict]) -> list[dict]:
    """Filtra campos relevantes dos itens para reduzir tokens."""
    resultado = []
    for item in itens:
        d = {k: item[k] for k in _CAMPOS_PRINCIPAIS if k in item}
        # Trunca descrição longa — 5 % dos itens excedem 500 chars sem acrescentar
        # informação relevante para a classificação de aderência.
        if "descricao" in d and len(d["descricao"]) > _MAX_CHARS_DESCRICAO:
            d["descricao"] = d["descricao"][:_MAX_CHARS_DESCRICAO]
        for k in _CAMPOS_APOIO:
            v = item.get(k)
            if v and str(v).strip():
                d[k] = v
        resultado.append(d)
    return resultado

def _extrair_json(texto: str, processo: str = "", lote: int = 0) -> dict:
    """
    Extrai JSON da resposta do modelo.

    Estratégia em cascata:
    1. Parse direto (esperado quando o modelo respeita "SOMENTE JSON").
    2. Strip de markdown code-fence (```json ... ```) e novo parse.
    3. Regex sobre o texto original para extrair o maior bloco {…}.
    4. Regex após strip de fences.

    Se nenhuma estratégia funcionar, levanta ValueError (capturado pelo retry).
    """
    texto = texto.strip()

    # --- 1. Parse direto ---
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        pass

    # --- 2. Strip de markdown code-fences e novo parse ---
    texto_sem_fence = re.sub(r'^```(?:json)?\s*\n?', '', texto)
    texto_sem_fence = re.sub(r'\n?```\s*$', '', texto_sem_fence).strip()
    try:
        resultado = json.loads(texto_sem_fence)
        log.warning(
            "[%s] lote %d — modelo usou bloco markdown (```json); "
            "extraído após strip de fence.",
            processo, lote,
        )
        return resultado
    except json.JSONDecodeError:
        pass

    # --- 3 & 4. Regex: tenta nas versões original e sem-fence ---
    for candidato, origem in [(texto, "original"), (texto_sem_fence, "sem_fence")]:
        m = re.search(r'\{[\s\S]+\}', candidato)
        if m:
            bloco = m.group()
            try:
                resultado = json.loads(bloco)
                log.warning(
                    "[%s] lote %d — modelo retornou texto extra; "
                    "extraído via regex (%s). Prefixo: %r",
                    processo, lote, origem, texto[:80],
                )
                return resultado
            except json.JSONDecodeError:
                pass

    raise ValueError(
        f"[{processo}] lote {lote} — não foi possível extrair JSON válido da resposta. "
        f"Primeiros 300 chars: {texto[:300]}"
    )

# =============================================================================
# VALIDAÇÃO DE SCHEMA
# =============================================================================

def _schema_para_validar() -> dict:
    """
    Retorna o schema a usar na validação, coerente com o que o modelo recebeu.
    Ollama sempre usa o schema compacto (é ele que guia o structured output).
    Outros provedores usam o schema do modo efetivo (full ou compacto).
    """
    return _schema_ollama if PROVEDOR == "ollama" else _schema_validacao

def _saida_compacta_ativa() -> bool:
    """
    Informa se a saída esperada nesta sessão é a compacta.

    Regra:
    - Ollama sempre opera com schema compacto (structured output guiado pelo schema compacto);
    - os demais provedores seguem o schema do modo efetivo.

    Use esta função em qualquer ponto que precise construir objetos compatíveis com
    o schema esperado — inclusive itens sintéticos do pré-filtro.
    """
    return PROVEDOR == "ollama" or _modo_efetivo == "compacto"

def validar_schema(data: dict) -> tuple[bool, str]:
    """
    Valida o resultado do LLM contra o schema correspondente ao que foi pedido.
    jsonschema é dependência obrigatória; não há fallback fraco.
    """
    schema = _schema_para_validar()
    try:
        _jsc.validate(instance=data, schema=schema)
        return True, ""
    except _jsc.ValidationError as e:
        return False, e.message
    except _jsc.SchemaError as e:
        log.warning("Schema de validação malformado: %s", e)
        return True, ""   # schema com problema não bloqueia resultado válido

# =============================================================================
# ORÇAMENTO DE TOKENS E BATCHING
# =============================================================================

def _budget_itens() -> int:
    """
    Calcula quantos tokens sobram exclusivamente para os itens da licitação.
    Desconta: tokens fixos (sistema + portfólio) + tokens variáveis não-item
    ([CONTEXTO_PROCESSO] + instruções finais) + tokens de saída + margem.

    Lança RuntimeError se o orçamento for negativo (fail-fast): isso indica
    que a configuração atual não cabe na janela de contexto — a solução deve
    ser explícita, não silenciosa.
    """
    if PROVEDOR == "ollama":
        window = OLLAMA_NUM_CTX
    else:
        window = CONTEXT_WINDOW.get(PROVEDOR, 32_000)

    max_out = MAX_OUTPUT_TOKENS.get(PROVEDOR, 2048)
    budget  = window - _tokens_fixos - _tokens_variavel_fixo - max_out - MARGEM_TOKENS

    if budget <= 0:
        raise RuntimeError(
            f"Orçamento de tokens para itens negativo ou zero ({budget} tokens). "
            f"Tokens fixos ({_tokens_fixos}) + variáveis ({_tokens_variavel_fixo}) "
            f"+ saída máx ({max_out}) + margem ({MARGEM_TOKENS}) "
            f"ultrapassam a janela de contexto ({window} tokens). "
            "Soluções: (1) use MODO='compacto' para reduzir os tokens fixos; "
            "(2) aumente OLLAMA_NUM_CTX se usar Ollama; "
            "(3) reduza MARGEM_TOKENS com cautela."
        )

    if budget < 100:
        log.warning(
            "Orçamento para itens extremamente apertado (~%d tokens). "
            "A análise pode degradar; considere aumentar a janela ou reduzir o contexto fixo.",
            budget,
        )
        return 100

    return budget

def _limite_itens_efetivo() -> int:
    """
    Retorna o limite efetivo de itens por lote para o provedor ativo.

    Prioridade:
      1. MAX_ITENS_POR_LOTE > 0  → override explícito do usuário (vale para qualquer provedor)
      2. _MAX_ITENS_DEFAULT_PROVEDOR  → default por provedor (Ollama=20, Anthropic=25, OpenAI=25)
      3. 0  → sem limite (só o budget de tokens controla)
    """
    if MAX_ITENS_POR_LOTE > 0:
        return MAX_ITENS_POR_LOTE
    return _MAX_ITENS_DEFAULT_PROVEDOR.get(PROVEDOR, 0)


def _dividir_por_budget(itens_filtrados: list[dict], budget_tokens: int) -> list[list[dict]]:
    """
    Divide itens em lotes respeitando dois critérios independentes:
      1. Orçamento de tokens: evita estourar a janela de contexto do modelo.
      2. Limite de itens (_limite_itens_efetivo): protege contra truncamento de output
         em nuvem (Anthropic/OpenAI=25) e contra timeout de inferência local (Ollama=20).

    Ambos os critérios provocam corte; o que atingir primeiro divide o lote.
    """
    limite_itens = _limite_itens_efetivo()
    lotes: list[list[dict]] = []
    lote_atual: list[dict] = []
    tokens_lote = 0

    for item in itens_filtrados:
        item_tokens = len(json.dumps(item, ensure_ascii=False)) // CHARS_POR_TOKEN
        estouro_budget = lote_atual and tokens_lote + item_tokens > budget_tokens
        estouro_itens  = limite_itens > 0 and len(lote_atual) >= limite_itens
        if estouro_budget or estouro_itens:
            lotes.append(lote_atual)
            lote_atual  = [item]
            tokens_lote = item_tokens
        else:
            lote_atual.append(item)
            tokens_lote += item_tokens

    if lote_atual:
        lotes.append(lote_atual)

    return lotes

# =============================================================================
# PRÉ-FILTRO MELHORADO (score ponderado)
# =============================================================================

def pre_filtrar(processo: ProcessoInput) -> PreFiltroResultado:
    """
    Avalia a licitação por score ponderado de termos positivos e negativos,
    com bônus estrutural via NCM whitelist.

    Inclui objetoCompra e descrições dos itens na análise.
    Retorna decisão: CONTINUAR | AMBIGUO | DESCARTAR.

    Regras de decisão (em ordem de prioridade):
      1. score > 0                                   → CONTINUAR
      2. score <= -LIMIAR_DESCARTAR                  → DESCARTAR
      3. nenhum positivo E ao menos 1 negativo       → DESCARTAR
      4. demais (score 0, sem positivos, sem negativos) → AMBIGUO
    """
    if not USAR_PRE_FILTRO:
        return PreFiltroResultado("CONTINUAR", 0.0, [], [])

    # Texto normalizado (sem acentos): objetoCompra + descrições dos itens
    texto_partes = [processo.objeto_compra or ""]
    for item in processo.itens:
        texto_partes.append(item.get("descricao") or "")
        texto_partes.append(item.get("informacaoComplementar") or "")
    texto = _normalizar(" ".join(texto_partes))

    positivos_encontrados = [t for t, p in _PADROES_POSITIVOS if p.search(texto)]
    negativos_encontrados = [t for t, p in _PADROES_NEGATIVOS if p.search(texto)]

    # Bônus NCM: +1 se qualquer item tiver NCM dentro da whitelist de produtos
    # Resgatécnica. Sinal estrutural complementar ao lexical, não substituto.
    ncm_bonus = 0
    for item in processo.itens:
        ncm = item.get("ncmNbsCodigo") or ""
        if ncm and any(ncm.startswith(pref) for pref in _NCM_POSITIVOS_PREFIXOS):
            ncm_bonus = 1  # bônus único por processo (não acumula por item)
            break

    score = float(len(positivos_encontrados) - len(negativos_encontrados) + ncm_bonus)

    if score > 0:
        decisao = "CONTINUAR"
    elif score <= -LIMIAR_DESCARTAR:
        decisao = "DESCARTAR"
    elif not positivos_encontrados and negativos_encontrados:
        # Sem nenhum sinal a favor + ao menos um sinal contra → descartar
        # (nova regra: não precisa atingir LIMIAR_DESCARTAR quando não há positivos)
        decisao = "DESCARTAR"
    else:
        decisao = "AMBIGUO"

    return PreFiltroResultado(decisao, score, positivos_encontrados, negativos_encontrados)

# =============================================================================
# TELEMETRIA
# =============================================================================

def _gravar_telemetria(entrada: EntradaTelemetria):
    """Acrescenta uma linha JSON ao arquivo de telemetria."""
    linha = {
        "ts":        entrada.timestamp,
        "processo":  entrada.processo,
        "provedor":  entrada.provedor,
        "modelo":    entrada.modelo,
        "modo":      entrada.modo,
        "lote":      (f"{entrada.path_id}/{entrada.total_lotes}"
                      if entrada.path_id else
                      f"{entrada.lote_idx}/{entrada.total_lotes}"),
        "n_itens":   entrada.n_itens_lote,
        "tok_in":    entrada.input_tokens,
        "tok_out":   entrada.output_tokens,
        "cache_r":   entrada.cache_read_tokens,
        "cache_w":   entrada.cache_write_tokens,
        "lat_ms":    entrada.latency_ms,
        "custo_usd": entrada.custo_usd,
        "schema_ok": entrada.schema_valido,
        "status":    entrada.status,
        "erro":      entrada.erro,
    }
    with open(OUTPUT_TELEMETRIA, "a", encoding="utf-8") as f:
        f.write(json.dumps(linha, ensure_ascii=False) + "\n")

def _estimar_custo(modelo: str, resp: LLMResponse) -> float | None:
    """Estima custo em USD com base no uso de tokens reportado pelo backend."""
    tabela = _CUSTO_POR_MTOK.get(modelo)
    if not tabela:
        return None
    custo  = (resp.input_tokens  or 0) * tabela.get("in",  0) / 1_000_000
    custo += (resp.output_tokens or 0) * tabela.get("out", 0) / 1_000_000
    custo += (resp.cache_read_tokens  or 0) * tabela.get("cache_r", 0) / 1_000_000
    custo += (resp.cache_write_tokens or 0) * tabela.get("cache_w", 0) / 1_000_000
    return round(custo, 6)


def _salvar_resposta_debug(processo: str, lote: int, texto: str, motivo: str, n_itens: int,
                           output_tokens: int | None, path_id: str | None = None) -> None:
    """
    Persiste a resposta bruta do LLM quando ocorre falha de parsing/schema.

    Útil para distinguir rapidamente:
      - JSON com fence simples (recuperável)
      - JSON truncado (output_tokens == max_tokens)
      - JSON com texto extra fora do bloco
      - JSON semanticamente inválido contra o schema

    Arquivos gravados em DEBUG_LLM_DIR/{processo_norm}_lote_{rotulo}.txt onde rotulo
    é o path_id ("1.2.1") quando houver split, ou o índice do lote em caso contrário.
    Se já existir um arquivo com o mesmo nome, sufixos numéricos são adicionados.
    """
    try:
        DEBUG_LLM_DIR.mkdir(exist_ok=True)
        nome_base = normalizar_ctrl_para_pasta(processo)
        rotulo = path_id if path_id else str(lote)
        candidato = DEBUG_LLM_DIR / f"{nome_base}_lote_{rotulo}.txt"
        sufixo = 1
        while candidato.exists():
            candidato = DEBUG_LLM_DIR / f"{nome_base}_lote_{rotulo}_{sufixo}.txt"
            sufixo += 1
        with open(candidato, "w", encoding="utf-8") as f:
            f.write(f"# processo:      {processo}\n")
            f.write(f"# lote:          {lote}\n")
            if path_id:
                f.write(f"# path_id:       {path_id}\n")
            f.write(f"# n_itens:       {n_itens}\n")
            f.write(f"# output_tokens: {output_tokens}\n")
            f.write(f"# chars:         {len(texto)}\n")
            f.write(f"# motivo:        {motivo}\n")
            f.write(f"# {'=' * 70}\n\n")
            f.write(texto)
    except Exception as e:
        log.warning("Não foi possível gravar debug da resposta: %s", e)

# =============================================================================
# RETRIES COM BACKOFF EXPONENCIAL
# =============================================================================

def _deve_retentar(exc: Exception) -> bool:
    """
    Retorna True se a exceção é transitória e deve ser retentada.

    Usa getattr defensivo no SDK OpenAI para compatibilidade com versões variadas.
    """
    if isinstance(exc, SchemaValidationError):
        return False                # erro de schema: não adianta retentar

    # Anthropic: 429 e 5xx
    if HAS_ANTHROPIC and isinstance(exc, (_ant.RateLimitError, _ant.APIStatusError)):
        status = getattr(exc, "status_code", 0)
        return status in (429, 500, 502, 503, 529)

    # OpenAI: usa getattr para compatibilidade entre versões do SDK
    # (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError, APIStatusError)
    if HAS_OPENAI:
        openai_retryables = tuple(
            exc_type for exc_type in (
                getattr(_oai, "RateLimitError",      None),
                getattr(_oai, "APIConnectionError",  None),
                getattr(_oai, "APITimeoutError",     None),
                getattr(_oai, "InternalServerError", None),
                getattr(_oai, "APIStatusError",      None),
            ) if exc_type is not None
        )
        if openai_retryables and isinstance(exc, openai_retryables):
            status = getattr(exc, "status_code", None)
            # status None = sem HTTP (conexão/timeout) → sempre retentar
            # 408/409/429 = rate limit ou conflito → retentar
            # >= 500 = erro de servidor → retentar
            return status is None or status in (408, 409, 429) or status >= 500

    # Requests (Ollama)
    if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True
    # RuntimeError genérico (ex.: Ollama fora do ar) → retentar
    if isinstance(exc, RuntimeError) and "inacessível" in str(exc):
        return True

    return False

def com_retry(fn, processo: str = "", lote: int = 0):
    """Executa fn() com até MAX_TENTATIVAS tentativas e backoff exponencial."""
    for tentativa in range(1, MAX_TENTATIVAS + 1):
        try:
            return fn()
        except Exception as exc:
            if tentativa == MAX_TENTATIVAS or not _deve_retentar(exc):
                raise
            delay = DELAY_BASE_S * (2 ** (tentativa - 1)) + random.uniform(0, JITTER_MAX_S)
            log.warning(
                "[%s] Lote %d — tentativa %d/%d falhou: %s. Aguardando %.1fs...",
                processo, lote, tentativa, MAX_TENTATIVAS, exc, delay,
            )
            time.sleep(delay)

# =============================================================================
# MONTAGEM DA MENSAGEM DO USUÁRIO
# =============================================================================

def montar_user_message(processo: ProcessoInput, itens_filtrados: list[dict]) -> tuple[str, int]:
    """
    Monta a mensagem completa do usuário e retorna (mensagem, len_prefixo_cacheavel).
    Estrutura:
      [Prefixo cacheável] = header template + portfólio substituído
      [Variável]          = [CONTEXTO_PROCESSO] + itens + INSTRUÇÕES ADICIONAIS
    O objetoCompra fica na parte variável (fora do cache, pois muda por licitação).
    """
    contexto = {
        "numeroControlePNCP": processo.ctrl,
        "objetoCompra":       processo.objeto_compra,
        "orgao":              processo.orgao,
        "uf":                 processo.uf,
        "valorTotalEstimado": processo.valor_estimado,
        "modalidade":         processo.modalidade,
    }
    contexto_str = json.dumps(contexto, ensure_ascii=False)
    itens_str    = json.dumps(itens_filtrados, ensure_ascii=False)

    variavel = (
        f"\n[CONTEXTO_PROCESSO]\n{contexto_str}\n\n"
        + itens_str
        + _template_apos_itens
    )
    user_msg = _template_antes_itens + variavel
    return user_msg, len(_template_antes_itens)

# =============================================================================
# CHAMADA PRINCIPAL AO LLM (com telemetria, validação e retry)
# =============================================================================

def _chamar_com_telemetria(
    backend: LLMBackend,
    processo: ProcessoInput,
    itens_lote: list[dict],
    lote_idx: int,
    total_lotes: int,
    path_id: str | None = None,
) -> dict:
    """
    Chama o backend, valida o schema, grava telemetria e retorna o resultado.
    Levanta SchemaValidationError se a saída não passar na validação.

    path_id: identificador hierárquico do lote (ex.: "1.2.1") para sublotes
    gerados pelo split recursivo. Quando None, registra apenas lote_idx/total_lotes.
    """
    itens_filtrados = _itens_para_llm(itens_lote)
    user_msg, prefix_len = montar_user_message(processo, itens_filtrados)

    req = LLMRequest(
        system_prompt=_system_prompt,
        user_prompt=user_msg,
        temperature=0.1,
        max_tokens=MAX_OUTPUT_TOKENS.get(PROVEDOR, 4096),
        # OpenAI tenta structured output com o schema; Ollama usa para format=.
        # Anthropic não usa json_schema no campo (caching gerencia o formato).
        json_schema=_schema_para_validar() if PROVEDOR in ("ollama", "openai") else None,
        timeout_s=OLLAMA_TIMEOUT if PROVEDOR == "ollama" else 120,
        cache_system=(PROVEDOR == "anthropic" and ANTHROPIC_CACHE_PROMPT),
        cache_user_prefix_len=(prefix_len if PROVEDOR == "anthropic" and ANTHROPIC_CACHE_PROMPT else 0),
    )

    ts = datetime.now(timezone.utc).isoformat()
    resp: LLMResponse | None = None
    status = "ok"
    erro_str: str | None = None
    schema_ok = False
    resultado: dict = {}

    try:
        resp = backend.generate(req)
        resultado = _extrair_json(resp.text, processo=processo.ctrl, lote=lote_idx)

        valido, motivo = validar_schema(resultado)
        if not valido:
            status = "schema_invalido"
            log.warning("[%s] Lote %s — schema inválido: %s",
                        processo.ctrl, path_id or lote_idx, motivo)
            _salvar_resposta_debug(
                processo.ctrl, lote_idx, resp.text,
                f"schema_invalido: {motivo}", len(itens_lote),
                resp.output_tokens, path_id=path_id,
            )
            raise SchemaValidationError(motivo)
        schema_ok = True

    except SchemaValidationError:
        raise
    except Exception as e:
        status = "erro_llm"
        erro_str = str(e)
        # Resposta bruta só existe se backend.generate() retornou; ValueError
        # do _extrair_json acontece após resp ser preenchido.
        if resp is not None and resp.text:
            _salvar_resposta_debug(
                processo.ctrl, lote_idx, resp.text,
                f"erro_llm: {e}", len(itens_lote),
                resp.output_tokens, path_id=path_id,
            )
        raise

    finally:
        _gravar_telemetria(EntradaTelemetria(
            timestamp=ts,
            processo=processo.ctrl,
            provedor=backend.provider_name,
            modelo=backend.model_name,
            modo=_modo_efetivo,
            lote_idx=lote_idx,
            total_lotes=total_lotes,
            n_itens_lote=len(itens_lote),
            input_tokens=resp.input_tokens if resp else None,
            output_tokens=resp.output_tokens if resp else None,
            cache_read_tokens=resp.cache_read_tokens if resp else None,
            cache_write_tokens=resp.cache_write_tokens if resp else None,
            latency_ms=resp.latency_ms if resp else None,
            custo_usd=_estimar_custo(backend.model_name, resp) if resp else None,
            schema_valido=schema_ok,
            status=status,
            erro=erro_str,
            path_id=path_id,
        ))

    return resultado

# =============================================================================
# PIPELINE — ETAPA 1: COLETAR
# =============================================================================

def coletar(licitacoes: list[dict]) -> list[ProcessoInput]:
    """
    Constrói ProcessoInput para cada licitação que possui itens.json.
    Aplica strip() ao ctrl para eliminar espaços ou quebras vindos do JSON.
    O ctrl é armazenado no formato cru do PNCP (com "/"); a normalização para
    pasta ("_") ocorre apenas em caminho_itens() e caminho_aderencia().

    Filtro estrutural ME/EPP (tipoBeneficio):
      Itens com tipoBeneficio ∉ TIPOS_BENEFICIO_ELEGIVEIS são removidos antes
      de qualquer análise — a Resgatécnica não pode participar deles.
      Se todos os itens forem inelegíveis, a licitação é descartada com
      aderencia.json sintética (status "meepp") sem chamar o LLM.
    """
    processos: list[ProcessoInput] = []
    sem_itens         = 0
    excluidos_meepp   = 0   # licitações 100% ME/EPP
    itens_filtrados   = 0   # itens ME/EPP removidos de licitações parcialmente abertas
    expiradas         = 0   # prazo já encerrado ou insuficiente
    situacao_invalida = 0   # revogadas / anuladas / suspensas

    _TZ_BR   = ZoneInfo("America/Sao_Paulo")
    agora_br = datetime.now(_TZ_BR)

    for l in licitacoes:
        ctrl = str(l.get("numeroControlePNCP") or "").strip()
        if not ctrl:
            continue

        # ── Filtro de situação administrativa ────────────────────────────────
        sit_id = l.get("situacaoCompraId")
        if sit_id in SITUACOES_INVALIDAS:
            situacao_invalida += 1
            log.info("  [SITUAÇÃO] %s — situacaoCompraId=%s (%s), ignorando",
                     ctrl, sit_id, l.get("situacaoCompraNome", "?"))
            continue

        # ── Filtro de prazo mínimo ────────────────────────────────────────────
        # Reavalia dataEncerramentoProposta para detectar licitações que
        # expiraram após a geração do pncp_licitacoes.json (arquivo pode ser
        # de dias/semanas atrás).
        # Formato da API: "2026-05-10T09:00:00" (horário de Brasília, sem fuso).
        # Comparamos com datetime de Brasília para honrar a hora exata de
        # encerramento (ex.: 08:55), não apenas a data.
        if DIAS_MINIMOS_PREPARO > 0:
            enc_str = l.get("dataEncerramentoProposta")
            if enc_str:
                try:
                    # Parseia o datetime sem fuso e localiza em Brasília
                    enc_dt = datetime.fromisoformat(str(enc_str)[:19]).replace(
                        tzinfo=_TZ_BR
                    )
                    limite = agora_br + timedelta(days=DIAS_MINIMOS_PREPARO)
                    if enc_dt < limite:
                        expiradas += 1
                        prazo_h = (enc_dt - agora_br).total_seconds() / 3600
                        log.info(
                            "  [PRAZO] %s — encerramento %s (%.1fh), ignorando",
                            ctrl, enc_str[:16], prazo_h,
                        )
                        continue
                except ValueError:
                    pass  # data inválida → manter (sem descarte por precaução)

        itens = ler_itens(ctrl)
        if not itens:
            sem_itens += 1
            continue

        # ── Filtro estrutural ME/EPP ──────────────────────────────────────────
        itens_elegiveis = [
            i for i in itens
            if i.get("tipoBeneficio") in TIPOS_BENEFICIO_ELEGIVEIS
        ]
        n_excluidos = len(itens) - len(itens_elegiveis)

        if not itens_elegiveis:
            # Nenhum item elegível — Resgatécnica não pode participar
            excluidos_meepp += 1
            log.info("  [ME/EPP] %s — todos os %d itens exclusivos ME/EPP, descartando",
                     ctrl, len(itens))
            _gravar_aderencia_meepp(ctrl, itens)
            continue

        if n_excluidos:
            itens_filtrados += n_excluidos
            log.info("  [ME/EPP] %s — %d/%d itens ME/EPP removidos, analisando %d",
                     ctrl, n_excluidos, len(itens), len(itens_elegiveis))

        # ── Montar ProcessoInput com apenas os itens elegíveis ────────────────
        orgao_ent = l.get("orgaoEntidade") or {}
        if isinstance(orgao_ent, str):
            try: orgao_ent = json.loads(orgao_ent.replace("'", '"'))
            except Exception: orgao_ent = {}

        unidade = l.get("unidadeOrgao") or {}
        if isinstance(unidade, str):
            try: unidade = json.loads(unidade.replace("'", '"'))
            except Exception: unidade = {}

        try:
            valor = float(l.get("valorTotalEstimado") or 0) or None
        except (TypeError, ValueError):
            valor = None

        processos.append(ProcessoInput(
            ctrl=ctrl,
            objeto_compra=str(l.get("objetoCompra") or ""),
            orgao=str(orgao_ent.get("razaoSocial") or ""),
            uf=str(unidade.get("ufSigla") or unidade.get("ufNome") or ""),
            valor_estimado=valor,
            modalidade=str(l.get("modalidadeNome") or ""),
            itens=itens_elegiveis,
        ))

    log.info(
        "Etapa COLETAR — elegíveis: %d | sem itens: %d | "
        "expiradas: %d | situação inválida: %d | "
        "descartados ME/EPP: %d | itens ME/EPP filtrados: %d",
        len(processos), sem_itens, expiradas, situacao_invalida,
        excluidos_meepp, itens_filtrados,
    )
    return processos

# =============================================================================
# PIPELINE — ETAPA 2: PRÉ-FILTRAR
# =============================================================================

def pre_filtrar_lote(
    processos: list[ProcessoInput],
    ja_analisados: set[str] | dict[str, str],
) -> tuple[list[ProcessoInput], list[ProcessoInput], list[ProcessoInput]]:
    """
    Classifica processos em:
      - pendentes:   analisar com LLM (CONTINUAR ou AMBIGUO)
      - descartados: NAO_ADERENTE por heurística (sem LLM)
      - ignorados:   já têm aderencia.json (INCREMENTAL)

    ja_analisados contém nomes de pasta (com "_"), por isso compara contra
    normalizar_ctrl_para_pasta(p.ctrl).
    """
    pendentes:   list[ProcessoInput] = []
    descartados: list[ProcessoInput] = []
    ignorados:   list[ProcessoInput] = []

    for p in processos:
        if INCREMENTAL and normalizar_ctrl_para_pasta(p.ctrl) in ja_analisados:
            ignorados.append(p)
            continue
        resultado_pf = pre_filtrar(p)
        if resultado_pf.decisao == "DESCARTAR":
            descartados.append(p)
            _gravar_aderencia_pre_filtrado(p, resultado_pf)
        else:
            pendentes.append(p)

    log.info(
        "Etapa PRÉ-FILTRAR — pendentes: %d | descartados: %d | ignorados (cache): %d",
        len(pendentes), len(descartados), len(ignorados),
    )
    return pendentes, descartados, ignorados

def _construir_item_sintetico(item: dict, justif: str) -> dict:
    """
    Constrói item sintético compatível com o schema ATIVO nesta sessão.

    Usa _saida_compacta_ativa() — mesma lógica de _schema_para_validar() —
    para decidir o formato. Isso garante que itens sintéticos nunca causem
    falha de validação de schema, inclusive no caso Ollama com MODO='full'.
    """
    base: dict[str, Any] = {
        "numeroItem":         item.get("numeroItem"),
        "descricao":          item.get("descricao", ""),
        "classificacao":      "NAO_ADERENTE",
        "grau_confianca":     0.85,
        "produto_referencia": None,
        "justificativa":      justif,
    }
    if not _saida_compacta_ativa():
        # Campos obrigatórios pelo schema full que o pré-filtro não produz
        base["categoria_portfolio_relacionada"]    = None
        base["subcategoria_portfolio_relacionada"] = None
        base["motivos_pro_match"]                  = []
        base["motivos_contra_match"]               = [justif]
        base["sinais_lexicais_enganosos"]          = []
        base["observacao"]                         = (
            "Classificado por pré-filtro heurístico sem análise LLM."
        )
    return base

def _gravar_aderencia_pre_filtrado(processo: ProcessoInput, pf: PreFiltroResultado):
    """
    Grava aderencia.json com classificação sintética por item.
    Itens recebem campos compatíveis com o schema ativo (não ficam em lista vazia).
    """
    justif = (
        f"Pré-filtro heurístico: score={pf.score:.0f}. "
        f"Termos negativos: {pf.termos_negativos[:5]}. "
        "Contexto dominante claramente não-aderente ao portfólio da Resgatécnica."
    )
    itens_sinteticos = [
        _construir_item_sintetico(item, justif)
        for item in processo.itens
    ]
    resultado = {
        "parecer_geral": "NAO_ADERENTE",
        "itens_analisados": itens_sinteticos,
        "estatisticas": {
            "aderencia_direta": 0, "aderencia_parcial_forte": 0,
            "aderencia_parcial_fraca": 0, "falso_positivo_lexical": 0,
            "nao_aderente": len(processo.itens),
        },
        "_pre_filtrado": True,
        "_score_pre_filtro": pf.score,
        "_termos_negativos": pf.termos_negativos,
        "_processo": processo.ctrl,
    }
    salvar_aderencia(processo.ctrl, resultado, "pre_filtro", "heuristica")

    _gravar_telemetria(EntradaTelemetria(
        timestamp=datetime.now(timezone.utc).isoformat(),
        processo=processo.ctrl,
        provedor="pre_filtro",
        modelo="heuristica",
        modo=_modo_efetivo,
        lote_idx=0, total_lotes=0,
        n_itens_lote=len(processo.itens),
        input_tokens=None, output_tokens=None,
        cache_read_tokens=None, cache_write_tokens=None,
        latency_ms=None, custo_usd=0.0,
        schema_valido=True,
        status="pre_filtrado",
        erro=None,
    ))

def _gravar_aderencia_meepp(ctrl: str, itens: list[dict]):
    """
    Grava aderencia.json sintética para licitações cujos itens são 100%
    de participação exclusiva ou cota reservada para ME/EPP.

    A Resgatécnica não pode participar desses itens (tipoBeneficio 1 ou 3),
    portanto não é necessário chamar o LLM — o descarte é estrutural.
    """
    justif = (
        "Item de participação exclusiva ou cota reservada para ME/EPP "
        "(tipoBeneficio 1 ou 3). A Resgatécnica não é elegível para este item."
    )
    itens_sinteticos = [_construir_item_sintetico(item, justif) for item in itens]
    resultado = {
        "parecer_geral": "NAO_ADERENTE",
        "itens_analisados": itens_sinteticos,
        "estatisticas": {
            "aderencia_direta": 0, "aderencia_parcial_forte": 0,
            "aderencia_parcial_fraca": 0, "falso_positivo_lexical": 0,
            "nao_aderente": len(itens),
        },
        "_meepp": True,
        "_motivo": "Todos os itens são exclusivos ou cota reservada para ME/EPP",
        "_processo": ctrl,
    }
    salvar_aderencia(ctrl, resultado, "pre_filtro", "meepp")

    _gravar_telemetria(EntradaTelemetria(
        timestamp=datetime.now(timezone.utc).isoformat(),
        processo=ctrl,
        provedor="pre_filtro",
        modelo="meepp",
        modo=_modo_efetivo,
        lote_idx=0, total_lotes=0,
        n_itens_lote=len(itens),
        input_tokens=None, output_tokens=None,
        cache_read_tokens=None, cache_write_tokens=None,
        latency_ms=None, custo_usd=0.0,
        schema_valido=True,
        status="meepp",
        erro=None,
    ))

# =============================================================================
# PIPELINE — ETAPA 3: ANALISAR (LLM)
# =============================================================================

def _agregar_lotes(lotes: list[dict]) -> dict:
    """Agrega resultados de múltiplos lotes num resultado consolidado."""
    if len(lotes) == 1:
        return lotes[0]
    base = lotes[0].copy()
    todos_itens: list[dict] = []
    stats: dict[str, int] = {
        "aderencia_direta": 0, "aderencia_parcial_forte": 0,
        "aderencia_parcial_fraca": 0, "falso_positivo_lexical": 0, "nao_aderente": 0,
    }
    for lote in lotes:
        todos_itens.extend(lote.get("itens_analisados", []))
        for k in stats:
            stats[k] += lote.get("estatisticas", {}).get(k, 0)
    base["itens_analisados"] = todos_itens
    base["estatisticas"] = stats
    # Recalcula parecer geral
    total = len(todos_itens)
    if total > 0:
        ader = stats["aderencia_direta"] + stats["aderencia_parcial_forte"]
        parc = stats["aderencia_parcial_fraca"]
        p = ("ADERENTE" if ader/total >= 0.30
             else "PARCIALMENTE_ADERENTE" if (ader+parc)/total >= 0.20
             else "NAO_ADERENTE")
        if "resumo_licitacao" in base:
            base["resumo_licitacao"]["parecer_geral"] = p
            base["resumo_licitacao"]["justificativa_geral"] = (
                f"Agregado de {len(lotes)} lotes. Aderentes: {ader}/{total} ({ader/total:.0%}).")
        elif "parecer_geral" in base:
            base["parecer_geral"] = p
    return base

def _processar_lote_recursivo(
    backend: LLMBackend,
    processo: ProcessoInput,
    lote_orig: list[dict],
    idx: int,
    total_lotes: int,
    profundidade: int = 0,
    path_id: str | None = None,
) -> list[dict]:
    """
    Processa um lote com fallback de split automático.

    Fluxo:
      1. Tenta processar o lote inteiro via _chamar_com_telemetria + com_retry.
      2. Se falhar por parsing (ValueError) ou schema (SchemaValidationError) e
         o lote tiver >1 item e profundidade < MAX_PROFUNDIDADE_SPLIT,
         divide ao meio e reprocessa cada metade recursivamente.
      3. Erros não-recuperáveis (API, timeout etc.) não disparam split; o lote
         inteiro é marcado como falho.

    path_id rastreia a hierarquia do split na telemetria/logs/debug:
      "1"        → lote raiz
      "1.1"/"1.2" → primeiro nível de split
      "1.2.1"…   → segundo nível
    Permite reconstruir auditavelmente o que aconteceu com cada sublote.

    Retorna lista de resultados parciais (vazia se tudo falhou).
    O split é importante porque, em modo full com Anthropic/OpenAI, lotes muito
    grandes podem estourar max_tokens — dividir resgata os itens em vez de
    perder a licitação inteira.
    """
    if path_id is None:
        path_id = str(idx)

    try:
        resultado = com_retry(
            lambda: _chamar_com_telemetria(
                backend, processo, lote_orig, idx, total_lotes, path_id=path_id,
            ),
            processo=processo.ctrl,
            lote=idx,
        )
        return [resultado]

    except (ValueError, SchemaValidationError) as e:
        n = len(lote_orig)
        pode_dividir = (
            USAR_SPLIT_AUTOMATICO
            and n > 1
            and profundidade < MAX_PROFUNDIDADE_SPLIT
        )
        if not pode_dividir:
            motivo = "schema" if isinstance(e, SchemaValidationError) else "parsing"
            log.error(
                "[%s] Lote %s (n=%d, prof=%d) falhou por %s e não pode mais ser dividido: %s",
                processo.ctrl, path_id, n, profundidade, motivo, e,
            )
            return []

        meio = n // 2
        path_a = f"{path_id}.1"
        path_b = f"{path_id}.2"
        log.warning(
            "[%s] Lote %s falhou no parsing/schema (n=%d, prof=%d). "
            "Aplicando split: %s (%d itens) + %s (%d itens).",
            processo.ctrl, path_id, n, profundidade,
            path_a, meio, path_b, n - meio,
        )
        time.sleep(PAUSA_ENTRE_CHAMADAS)
        a = _processar_lote_recursivo(
            backend, processo, lote_orig[:meio], idx, total_lotes,
            profundidade + 1, path_id=path_a,
        )
        time.sleep(PAUSA_ENTRE_CHAMADAS)
        b = _processar_lote_recursivo(
            backend, processo, lote_orig[meio:], idx, total_lotes,
            profundidade + 1, path_id=path_b,
        )
        return a + b

    except Exception as e:
        # Erro não-recuperável (API/timeout/outros) — split não ajudaria.
        log.error("[%s] Lote %s falhou (não-recuperável por split): %s",
                  processo.ctrl, path_id, e)
        return []


def analisar(backend: LLMBackend, processos: list[ProcessoInput]) -> list[dict]:
    """Analisa cada processo via LLM, com batching por orçamento de tokens."""
    budget = _budget_itens()
    log.info(
        "Etapa ANALISAR — %d processos | budget tokens/lote: ~%d | modelo: %s",
        len(processos), budget, backend.model_name,
    )

    resultados: list[dict] = []
    for i, processo in enumerate(processos, 1):
        itens_filtrados = _itens_para_llm(processo.itens)
        lotes = _dividir_por_budget(itens_filtrados, budget)
        total_lotes = len(lotes)

        log.info(
            "[%d/%d] %s — %d itens → %d lote(s) | objeto: %.60s",
            i, len(processos), processo.ctrl,
            len(processo.itens), total_lotes,
            processo.objeto_compra,
        )

        resultados_lotes: list[dict] = []
        for idx, lote_filtrado in enumerate(lotes, 1):
            # Recupera itens originais correspondentes ao lote
            lote_orig = processo.itens[
                (idx-1)*len(lote_filtrado) : idx*len(lote_filtrado)
            ]
            resultados_lotes.extend(
                _processar_lote_recursivo(backend, processo, lote_orig, idx, total_lotes)
            )

            if idx < total_lotes:
                time.sleep(PAUSA_ENTRE_CHAMADAS)

        if not resultados_lotes:
            log.error("[%s] Nenhum lote bem-sucedido, licitação pulada", processo.ctrl)
            continue

        resultado = _agregar_lotes(resultados_lotes)
        resultado["_processo"] = processo.ctrl
        salvar_aderencia(processo.ctrl, resultado, backend.provider_name, backend.model_name)
        resultados.append(resultado)

        time.sleep(PAUSA_ENTRE_CHAMADAS)

    return resultados

# =============================================================================
# PIPELINE — ETAPA 4: CONSOLIDAR
# =============================================================================

_ORDEM_PRIORIDADE = {"ALTA": 0, "MEDIA": 1, "BAIXA": 2, "DESCARTAR": 3, "?": 4}

def _resumo_processo(ctrl: str, r: dict) -> dict:
    if "resumo_licitacao" in r:
        res      = r["resumo_licitacao"]
        parecer  = res.get("parecer_geral", "?")
        contexto = res.get("contexto_dominante", "")
        justif   = res.get("justificativa_geral", "")
    else:
        parecer  = r.get("parecer_geral", "?")
        contexto = justif = ""
    rec        = r.get("recomendacao_comercial", {})
    prioridade = rec.get("nivel_prioridade", "?")
    return {
        "processo":          ctrl,
        "parecer_geral":     parecer,
        "prioridade":        prioridade,
        "deve_priorizar":    rec.get("deve_priorizar", False),
        "contexto_dominante": contexto,
        "justificativa":     justif,
        "recomendacao_motivo": rec.get("motivo", ""),
        "estatisticas":      r.get("estatisticas", {}),
        "total_itens":       len(r.get("itens_analisados", [])),
        "pre_filtrado":      r.get("_pre_filtrado", False),
        "analisado_em":      r.get("_analisado_em", ""),
        "provedor":          r.get("_provedor", "?"),
        "modelo":            r.get("_modelo", "?"),
    }

def consolidar(todos: list[dict]):
    """Gera pncp_aderencias.json e pncp_itens_consolidado.json."""
    resumos = [_resumo_processo(r["_processo"], r) for r in todos if "_processo" in r]
    resumos.sort(key=lambda x: (_ORDEM_PRIORIDADE.get(x["prioridade"], 4), x["processo"]))

    stats_glob: dict[str, int] = {
        k: 0 for k in ["aderencia_direta", "aderencia_parcial_forte",
                        "aderencia_parcial_fraca", "falso_positivo_lexical", "nao_aderente"]}
    pareceres:   dict[str, int] = {}
    prioridades: dict[str, int] = {}

    for r in resumos:
        for k in stats_glob: stats_glob[k] += r.get("estatisticas", {}).get(k, 0)
        p = r["parecer_geral"];  pareceres[p]   = pareceres.get(p, 0) + 1
        p = r["prioridade"];     prioridades[p] = prioridades.get(p, 0) + 1

    sumario = {
        "gerado_em":            datetime.now(timezone.utc).isoformat(),
        "total_analisadas":     len(resumos),
        "pareceres":            pareceres,
        "prioridades":          prioridades,
        "estatisticas_globais": stats_glob,
        "licitacoes":           resumos,
    }
    with open(OUTPUT_ADERENCIAS, "w", encoding="utf-8") as f:
        json.dump(sumario, f, ensure_ascii=False, indent=2)

    consolidado = [
        {"processo": r.get("_processo", "?"), **item}
        for r in todos
        for item in r.get("itens_analisados", [])
    ]
    with open(OUTPUT_CONSOLIDADO, "w", encoding="utf-8") as f:
        json.dump(consolidado, f, ensure_ascii=False, indent=2)

    log.info(
        "CONSOLIDAR — %s (%d licitações) | %s (%d itens)",
        OUTPUT_ADERENCIAS, len(resumos), OUTPUT_CONSOLIDADO, len(consolidado),
    )

    log.info("Por prioridade: %s", prioridades)
    altas = [r for r in resumos if r["prioridade"] == "ALTA" and not r.get("pre_filtrado")]
    if altas:
        log.info(">>> ALTA PRIORIDADE (%d) <<<", len(altas))
        for r in altas:
            log.info("  %s | %s", r["processo"], r.get("contexto_dominante", ""))

    return sumario

# =============================================================================
# VERIFICAÇÃO DE PRÉ-REQUISITOS
# =============================================================================

def verificar_prerequisitos() -> bool:
    ok = True
    if PROVEDOR == "anthropic":
        if not HAS_ANTHROPIC:
            log.error("pip install anthropic"); ok = False
        elif not os.environ.get("ANTHROPIC_API_KEY"):
            log.error("ANTHROPIC_API_KEY não definida"); ok = False
    elif PROVEDOR == "ollama":
        try:
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            modelos = [m["name"] for m in r.json().get("models", [])]
            base_modelo = OLLAMA_MODELO.split(":")[0]
            if not any(base_modelo in m for m in modelos):
                log.warning("Modelo '%s' pode não estar disponível. Instalados: %s",
                             OLLAMA_MODELO, modelos[:5])
        except Exception as e:
            log.error("Ollama inacessível em %s: %s — Execute: ollama serve", OLLAMA_BASE_URL, e)
            ok = False
    elif PROVEDOR == "openai":
        if not HAS_OPENAI:
            log.error("pip install openai"); ok = False
        elif not OPENAI_BASE_URL and not os.environ.get("OPENAI_API_KEY"):
            log.error("OPENAI_API_KEY não definida"); ok = False
    else:
        log.error("Provedor '%s' desconhecido", PROVEDOR); ok = False
    return ok

# =============================================================================
# MAIN
# =============================================================================

def main():
    log.info("=== pncp_agente.py — Análise de Aderência ===")
    log.info("Provedor: %s", PROVEDOR)

    if not verificar_prerequisitos():
        return
    if not API_DOCS_DIR.exists():
        log.error("Pacote de implantação não encontrado: %s", API_DOCS_DIR); return
    if not INPUT_FILE.exists():
        log.error("Arquivo não encontrado: %s", INPUT_FILE); return

    try:
        carregar_pacote()
    except FileNotFoundError as e:
        log.error("Erro ao carregar pacote: %s", e); return
    except RuntimeError as e:
        log.error("Configuração inviável: %s", e); return

    with open(INPUT_FILE, encoding="utf-8") as f:
        data = json.load(f)
    licitacoes = data.get("licitacoes", data) if isinstance(data, dict) else data
    log.info("Licitações carregadas: %d", len(licitacoes))

    # Descobre quais já foram analisadas.
    # ja_analisados: dict[ctrl_norm → nome_pasta_real]
    #   chave = ctrl normalizado (sem prefixo UF_CIDADE_) → usado para lookup rápido
    #   valor = nome da pasta real no disco → usado para ler aderencia.json
    # A comparação no pre_filtrar_lote() usa normalizar_ctrl_para_pasta() como chave,
    # o que funciona tanto para pastas legado quanto para pastas com prefixo.
    ja_analisados: dict[str, str] = {
        _ctrl_de_pasta(d.name): d.name
        for d in EDITAIS_DIR.iterdir()
        if d.is_dir() and (d / "aderencia.json").exists()
    } if INCREMENTAL else {}

    # Instancia backend (valida conexão e credenciais)
    try:
        backend = get_backend()
        log.info("Backend: %s (%s)", backend.provider_name, backend.model_name)
    except Exception as e:
        log.error("Falha ao instanciar backend: %s", e); return

    # ---- Pipeline em 4 etapas ----

    # 1. Coletar
    processos = coletar(licitacoes)
    if MAX_PROCESSOS > 0:
        log.info("MAX_PROCESSOS=%d — limitando análise (modo teste)", MAX_PROCESSOS)
        processos = processos[:MAX_PROCESSOS]

    # Log de resumption (licitações já analisadas)
    if ja_analisados:
        log.info("✓ Resumption ativado: %d licitação(ões) já analisada(s) — serão puladas",
                 len(ja_analisados))

    # 2. Pré-filtrar
    pendentes, descartados, ignorados = pre_filtrar_lote(processos, ja_analisados)

    # 3. Analisar (LLM)
    resultados_novos = analisar(backend, pendentes) if pendentes else []

    # 4. Consolidar todos os resultados
    todos_resultados: list[dict] = []

    # Já analisados anteriormente (lidos do disco)
    # ctrl_norm é a chave (ctrl sem prefixo UF_CIDADE_); pasta_nome é o nome real no disco.
    for ctrl_norm, pasta_nome in ja_analisados.items():
        p = EDITAIS_DIR / pasta_nome / "aderencia.json"
        try:
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
            r.setdefault("_processo", desnormalizar_ctrl_de_pasta(ctrl_norm))
            todos_resultados.append(r)
        except Exception as e:
            log.warning("[%s] Erro ao ler aderencia.json: %s", pasta_nome, e)

    # Descartados pelo pré-filtro (gravados acima em _gravar_aderencia_pre_filtrado)
    for p in descartados:
        ap = caminho_aderencia(p.ctrl)
        if ap.exists():
            try:
                with open(ap, encoding="utf-8") as f:
                    r = json.load(f)
                r.setdefault("_processo", p.ctrl)
                todos_resultados.append(r)
            except Exception:
                pass

    todos_resultados.extend(resultados_novos)

    if todos_resultados:
        consolidar(todos_resultados)

    # Sumário final
    log.info("=" * 60)
    log.info(
        "CONCLUÍDO — processadas NOVAS: %d | descartadas (pré-filtro): %d | "
        "puladas do resumption: %d | falhas: %d",
        len(resultados_novos), len(descartados),
        len(ja_analisados), len(pendentes) - len(resultados_novos),
    )
    log.info("Total em pncp_aderencias.json: %d licitações", len(todos_resultados))
    log.info("Telemetria gravada em: %s", OUTPUT_TELEMETRIA)


if __name__ == "__main__":
    main()
