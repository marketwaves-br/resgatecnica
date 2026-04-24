#!/usr/bin/env python3
"""
pncp_filtros.py — Lógica contextual compartilhada entre pipelines

Módulo criado no Sprint 1 para isolar as travas contextuais originalmente
definidas em pncp_agente.py e torná-las reutilizáveis pelo pipeline barato
(pncp_similaridade.py) sem acoplamento à biblioteca anthropic/openai.

Conteúdo:
  - Termos positivos/negativos do domínio Resgatécnica (cópia sincronizada
    com pncp_agente.py — quando um mudar, o outro deve acompanhar).
  - NCM whitelist (_NCM_POSITIVOS_PREFIXOS).
  - pre_filtrar_licitacao() — porta do pre_filtrar() do agente, sem dependência
    de dataclass ProcessoInput.
  - peso_contexto_licitacao() — NOVO. Multiplica score_bid conforme âncoras
    fortes / contextos incompatíveis / contextos impossíveis no objetoCompra.
  - calcular_score_bid() — NOVO. Substitui score_max por agregação top-3
    ponderada + densidade + penalização por variância.

Regra de sincronização:
  Se _TERMOS_POSITIVOS_RAW, _TERMOS_NEGATIVOS_RAW ou _NCM_POSITIVOS_PREFIXOS
  forem alterados em pncp_agente.py, copie para cá. A duplicação é intencional:
  o pipeline barato deve rodar sem anthropic/openai instalados.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


# =============================================================================
# Normalização de texto (cópia de pncp_agente.py)
# =============================================================================

def _normalizar(texto: str) -> str:
    """Remove acentos e converte para lowercase. Usado no pré-filtro."""
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


# =============================================================================
# Termos positivos/negativos — cópia sincronizada com pncp_agente.py
# =============================================================================

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

# NCM whitelist — cópia de pncp_agente.py. Prefixos de 4 dígitos suficientes
# para o escopo atual; cobertura baixa (~7,7% dos itens) mas sem falso positivo.
_NCM_POSITIVOS_PREFIXOS = {
    "8424",  # Extintores de incêndio e aparelhos de dispersão de líquidos
    "8705",  # Veículos automóveis para fins especiais (ambulâncias, viaturas)
    "8413",  # Bombas para líquidos (motobombas de combate a incêndio)
    "6211",  # Macacões e roupas de proteção (roupa de aproximação)
    "6506",  # Capacetes de proteção
    "9019",  # Aparelhos de mecanoterapia (desfibriladores portáteis)
}


def _compilar_padroes_prefiltro(
    termos_raw: set[str],
) -> list[tuple[str, re.Pattern]]:
    """Compila regex com \\b word boundary sobre texto normalizado (ASCII)."""
    resultado = []
    for t in termos_raw:
        t_norm = _normalizar(t)
        pat = re.compile(r"\b" + re.escape(t_norm) + r"\b")
        resultado.append((t_norm, pat))
    return resultado


_PADROES_POSITIVOS: list[tuple[str, re.Pattern]] = _compilar_padroes_prefiltro(
    _TERMOS_POSITIVOS_RAW
)
_PADROES_NEGATIVOS: list[tuple[str, re.Pattern]] = _compilar_padroes_prefiltro(
    _TERMOS_NEGATIVOS_RAW
)


# =============================================================================
# Pré-filtro contextual (porta do pre_filtrar() do pncp_agente.py)
# =============================================================================

LIMIAR_DESCARTAR = 1   # idêntico ao agente (2026-04-22)


@dataclass
class PreFiltroResultado:
    """
    Resultado do pré-filtro heurístico sobre uma licitação inteira.

    Campos:
      decisao     — "CONTINUAR" | "AMBIGUO" | "DESCARTAR"
      score       — pontuação líquida (n_positivos - n_negativos + ncm_bonus)
      positivos   — lista de termos positivos encontrados
      negativos   — lista de termos negativos encontrados
      ncm_hit     — True se algum item tem NCM na whitelist
    """
    decisao: str
    score: float
    positivos: list[str] = field(default_factory=list)
    negativos: list[str] = field(default_factory=list)
    ncm_hit: bool = False


def pre_filtrar_licitacao(
    objeto_compra: str,
    itens: list[dict],
) -> PreFiltroResultado:
    """
    Versão standalone do pre_filtrar() do pncp_agente.py, sem dependência de
    ProcessoInput. Opera sobre objetoCompra + descrições + informacaoComplementar.

    Regras (idênticas ao agente):
      1. score > 0                                   → CONTINUAR
      2. score <= -LIMIAR_DESCARTAR                  → DESCARTAR
      3. nenhum positivo E ao menos 1 negativo       → DESCARTAR
      4. resto (score 0, sem positivos, sem negativos) → AMBIGUO
    """
    partes = [objeto_compra or ""]
    for item in itens:
        partes.append(item.get("descricao") or "")
        partes.append(item.get("informacaoComplementar") or "")
    texto = _normalizar(" ".join(partes))

    positivos = [t for t, p in _PADROES_POSITIVOS if p.search(texto)]
    negativos = [t for t, p in _PADROES_NEGATIVOS if p.search(texto)]

    ncm_hit = False
    for item in itens:
        ncm = item.get("ncmNbsCodigo") or ""
        if ncm and any(ncm.startswith(pref) for pref in _NCM_POSITIVOS_PREFIXOS):
            ncm_hit = True
            break

    ncm_bonus = 1 if ncm_hit else 0
    score = float(len(positivos) - len(negativos) + ncm_bonus)

    if score > 0:
        decisao = "CONTINUAR"
    elif score <= -LIMIAR_DESCARTAR:
        decisao = "DESCARTAR"
    elif not positivos and negativos:
        decisao = "DESCARTAR"
    else:
        decisao = "AMBIGUO"

    return PreFiltroResultado(decisao, score, positivos, negativos, ncm_hit)


# =============================================================================
# Peso de contexto da licitação (NOVO no Sprint 1)
# =============================================================================

# Âncoras fortes — objeto contém evidência inequívoca do domínio Resgatécnica
_ANCORAS_FORTES_RAW = [
    "resgate", "salvamento", "emergência", "socorro", "aph",
    "ambulância", "ambulancia",
    "incêndio", "incendio", "combate a incêndio", "combate a incendio",
    "hidrante", "desencarceramento", "bombeiro",
    "balístico", "balistico", "descontaminação", "descontaminacao",
    "espaço confinado", "espaco confinado",
    "mergulho operacional", "maca de resgate",
    "viatura especial", "viaturas especiais",
    "uti móvel", "uti movel",
    "combate a sinistros",
]

# Contextos incompatíveis — objeto é de outro domínio, peso reduzido
_CONTEXTOS_INCOMPATIVEIS_RAW = [
    "mobiliário", "mobiliario", "papelaria",
    "merenda", "alimentação escolar", "alimentacao escolar",
    "gêneros alimentícios", "generos alimenticios",
    "medicamento", "medicamentos", "farmacêutic", "farmaceutic",
    "odontológic", "odontologic",
    "material escolar", "material didático", "material didatico",
    "material esportivo", "equipamento esportivo",
    "material de informática", "material de informatica",
    "licença de software", "licenca de software",
    "limpeza predial", "serviço de limpeza", "servico de limpeza",
    "obra civil", "obras civis", "reforma predial",
    "seguro veicular",
    "seguro total", "seguro da frota",
    "locação de veículos", "locacao de veiculos",
    "veículos automotores", "veiculos automotores",
    "veículo de passeio", "veiculo de passeio",
    "minivan",
    "transporte escolar",
    "pneus", "peças automotivas", "pecas automotivas",
    "troca de óleo", "troca de oleo",
    "ar condicionado",
    "medicina do trabalho",
    "engenharia de segurança", "engenharia de seguranca",
    "pcmso", "ltcat", "pgr", "aso",
    "stent", "materiais medico hospitalares", "materiais médico hospitalares",
    "gerador de hipoclorito", "condutivimetro", "condutivímetro",
    "adaptacao veicular", "adaptação veicular",
]

# Contextos impossíveis — objeto é tão claramente fora do escopo que merece 0.00.
# Subset muito restrito do incompatível: usado apenas quando não há âncora.
_CONTEXTOS_IMPOSSIVEIS_RAW = [
    "merenda escolar", "alimentação escolar", "alimentacao escolar",
    "gêneros alimentícios", "generos alimenticios",
    "material didático", "material didatico",
    "livros didáticos", "livros didaticos",
    "ar condicionado",
    "seguro total", "seguro da frota", "seguro veicular",
    "medicina do trabalho", "engenharia de segurança", "engenharia de seguranca",
    "pcmso", "ltcat", "pgr", "aso",
    "stent", "materiais medico hospitalares", "materiais médico hospitalares",
    "gerador de hipoclorito", "condutivimetro", "condutivímetro",
]


def _compilar_lista(termos: list[str]) -> re.Pattern:
    termos_norm = sorted({_normalizar(t) for t in termos}, key=len, reverse=True)
    padrao = r"\b(?:" + "|".join(re.escape(t) for t in termos_norm) + r")\b"
    return re.compile(padrao)


_RE_ANCORAS_FORTES = _compilar_lista(_ANCORAS_FORTES_RAW)
_RE_INCOMPATIVEIS = _compilar_lista(_CONTEXTOS_INCOMPATIVEIS_RAW)
_RE_IMPOSSIVEIS = _compilar_lista(_CONTEXTOS_IMPOSSIVEIS_RAW)


def peso_contexto_licitacao(objeto_compra: str) -> float:
    """
    Retorna multiplicador aplicado ao score da licitação conforme contexto
    dominante no objetoCompra.

    Escala (decisão do Sprint 1):
      1.00  âncora forte presente (vence sempre)
      0.70  objeto neutro — sem âncora, sem contexto incompatível
      0.35  contexto incompatível (mobiliário, merenda, medicamentos, TI…)
      0.00  contexto impossível (merenda escolar pura, material didático puro)

    Regra de precedência: âncora forte vence qualquer contexto negativo.
    A ausência de âncora não é fatal — objetos genéricos continuam com 0.70
    e dependem dos itens para pontuar.
    """
    if not objeto_compra:
        return 0.70
    texto = _normalizar(objeto_compra)

    if _RE_ANCORAS_FORTES.search(texto):
        return 1.00
    if _RE_IMPOSSIVEIS.search(texto):
        return 0.00
    if _RE_INCOMPATIVEIS.search(texto):
        return 0.35
    return 0.70


# =============================================================================
# score_bid — agregação top-3 + densidade + penalização por variância
# =============================================================================

def calcular_score_bid(
    scores_itens: list[float],
    n_elegiveis: int,
    peso_contexto: float,
) -> dict:
    """
    Substitui score_max como métrica da licitação. Implementa o desenho
    proposto nos textos do Gemini/GPT v2:

      1. top-3 ponderado: 0.50 * top1 + 0.30 * top2 + 0.20 * top3
         (pesos adaptados quando há <3 itens)
      2. fator de contexto global aplicado multiplicativamente
      3. fator de densidade: proporção de itens com score válido
      4. penalização por variância: item solitário em licitação grande
         com cauda próxima de zero → score * 0.55

    Parâmetros:
      scores_itens   — scores individuais de cada item elegível (após
                       _NEGATIVOS_ITEM e filtros de item). Inclui itens
                       de score zero (importantes para densidade).
      n_elegiveis    — total de itens elegíveis da licitação (tipoBeneficio
                       ∈ {4,5}), mesmo os de score zero.
      peso_contexto  — saída de peso_contexto_licitacao(objetoCompra).

    Retorna dict com:
      score_bid, score_top1, score_top3_medio,
      densidade_relevancia, n_itens_relevantes,
      penalizacao_variancia, peso_contexto
    """
    # Itens sem pontuação alguma não contam para agregação, mas contam para
    # densidade. "Relevante" = score > 0 (exclui zerados pelo _NEGATIVOS_ITEM
    # e equivalentes).
    scores_validos = [s for s in scores_itens if s > 0.0]
    n_validos = len(scores_validos)

    resultado = {
        "score_bid": 0.0,
        "score_top1": 0.0,
        "score_top3_medio": 0.0,
        "densidade_relevancia": 0.0,
        "n_itens_relevantes": n_validos,
        "penalizacao_variancia": False,
        "peso_contexto": peso_contexto,
    }

    if n_validos == 0:
        return resultado

    top = sorted(scores_validos, reverse=True)[:3]
    resultado["score_top1"] = top[0]
    resultado["score_top3_medio"] = sum(top) / len(top)

    # Agregação top-k ponderada
    if len(top) == 1:
        agregado = top[0] * 0.55
    elif len(top) == 2:
        agregado = 0.65 * top[0] + 0.35 * top[1]
    else:
        agregado = 0.50 * top[0] + 0.30 * top[1] + 0.20 * top[2]

    # Peso de contexto global
    agregado *= peso_contexto

    # Fator de densidade: penaliza licitação com muitos itens e poucos relevantes.
    # Fórmula da spec v2: densidade = min(1.0, n_validos / max(2, min(5, n_elegiveis)))
    denominador = max(2, min(5, n_elegiveis)) if n_elegiveis > 0 else 1
    densidade = min(1.0, n_validos / denominador)
    agregado *= (0.70 + 0.30 * densidade)
    resultado["densidade_relevancia"] = densidade

    # Penalização por variância: item solitário em licitação grande de cauda zero.
    # Objetivo: impedir que um único token contaminante "incendeie" edital de
    # 50 itens estruturalmente irrelevantes.
    if n_elegiveis >= 20 and n_validos == 1:
        # Média dos demais itens (inclui zerados): se próxima de zero, penalizar.
        demais = [s for s in scores_itens if s != scores_validos[0]]
        media_restante = sum(demais) / len(demais) if demais else 0.0
        if media_restante < 0.08:
            agregado *= 0.55
            resultado["penalizacao_variancia"] = True

    resultado["score_bid"] = min(agregado, 1.0)
    return resultado
