# -*- coding: utf-8 -*-
"""
pncp_scanner.py
Coleta licitações do Portal Nacional de Contratações Públicas (PNCP)
via API de Consultas e grava os resultados em pncp_licitacoes.json.

API base: https://pncp.gov.br/api/consulta
Endpoint: GET /v1/contratacoes/proposta
Documentação: PNCP_API_Consultas.pdf (Manual de APIs de Consultas — v1.0)
"""

import json
import logging
import re
import time
import unicodedata
from datetime import date, timedelta

import requests
import yaml

# =============================================================================
# CONFIGURAÇÃO — MODALIDADES
# Referência: tabela 5.2 do manual (codigoModalidadeContratacao)
# True  = consultar esta modalidade
# False = ignorar
# =============================================================================
MODALIDADES = {
    4: False,   # Concorrência - Eletrônica
    6: True,   # Pregão - Eletrônico
    # --------------- desabilitadas por padrão --------------------------------
    # 1:  False,  # Leilão - Eletrônico
    # 2:  False,  # Diálogo Competitivo
    # 3:  False,  # Concurso
    # 5:  False,  # Concorrência - Presencial
    # 7:  False,  # Pregão - Presencial
    # 8:  False,  # Dispensa de Licitação
    # 9:  False,  # Inexigibilidade
    # 10: False,  # Manifestação de Interesse
    # 11: False,  # Pré-qualificação
    # 12: False,  # Credenciamento
    # 13: False,  # Leilão - Presencial
}

# =============================================================================
# CONFIGURAÇÃO — UNIDADES FEDERATIVAS
# True  = incluir na busca
# False = ignorar
# Padrão de teste: apenas Sudeste habilitado
# =============================================================================
UFS = {
    # ── Sudeste ──────────────────────────────────────────────────────────────
    "ES": False,   # Espírito Santo
    "MG": True,   # Minas Gerais
    "RJ": False,   # Rio de Janeiro
    "SP": False,   # São Paulo
    # ── Sul ──────────────────────────────────────────────────────────────────
    "PR": False,  # Paraná
    "RS": False,  # Rio Grande do Sul
    "SC": False,  # Santa Catarina
    # ── Centro-Oeste ─────────────────────────────────────────────────────────
    "DF": False,  # Distrito Federal
    "GO": False,  # Goiás
    "MS": False,  # Mato Grosso do Sul
    "MT": False,  # Mato Grosso
    # ── Nordeste ─────────────────────────────────────────────────────────────
    "AL": False,  # Alagoas
    "BA": False,  # Bahia
    "CE": False,  # Ceará
    "MA": False,  # Maranhão
    "PB": False,  # Paraíba
    "PE": False,  # Pernambuco
    "PI": False,  # Piauí
    "RN": False,  # Rio Grande do Norte
    "SE": False,  # Sergipe
    # ── Norte ────────────────────────────────────────────────────────────────
    "AC": False,  # Acre
    "AM": False,  # Amazonas
    "AP": False,  # Amapá
    "PA": False,  # Pará
    "RO": False,  # Rondônia
    "RR": False,  # Roraima
    "TO": False,  # Tocantins
}

# =============================================================================
# CONFIGURAÇÃO — FILTRO SRP
# SRP = Sistema de Registro de Preços (campo booleano na resposta da API).
# Não é uma modalidade — é um atributo da contratação.
#
#   None  → incluir TODAS as contratações (SRP e não-SRP)   ← padrão
#   True  → incluir APENAS contratações SRP
#   False → incluir APENAS contratações não-SRP
# =============================================================================
FILTRO_SRP = None

# =============================================================================
# CONFIGURAÇÃO — FILTRO DE AMPARO LEGAL
# Filtra por Lei 14.133/2021 — Art. 28: modalidade de licitação obrigatória.
#
#   None  → incluir TODAS as contratações (indiferente do amparo legal)
#   [1]   → incluir APENAS bens e serviços COMUNS (Art. 28, I — pregão obrigatório)
#   [2]   → incluir APENAS bens/serviços especiais + obras/engenharia (Art. 28, II)
#   [1,2] → incluir ambas (padrão: apenas comuns recomendado)
#
# RECOMENDADO: [1] — elimina obras e serviços de engenharia estruturalmente.
# =============================================================================
FILTRO_AMPARO_LEGAL = [1]  # apenas bens e serviços comuns

# =============================================================================
# CONFIGURAÇÃO — JANELA DE TEMPO
# dataFinal = data de hoje + JANELA_DIAS
# A API retorna licitações com período de recebimento de propostas em aberto
# até essa data. Não há parâmetro dataInicial neste endpoint — o filtro de
# prazo mínimo é aplicado localmente sobre dataEncerramentoProposta.
# =============================================================================
JANELA_DIAS = 30

# =============================================================================
# CONFIGURAÇÃO — PRAZO MÍNIMO PARA PREPARAÇÃO DE PROPOSTA
#
# Descarta licitações cuja dataEncerramentoProposta seja inferior a
# hoje + DIAS_MINIMOS_PREPARO. Evita trazer licitações que abrem/fecham
# rápido demais para cotação, documentação e elaboração da proposta.
#
# Exceção SRP: quando srp=True, aplica DIAS_MINIMOS_PREPARO_SRP (menor),
# pois a empresa pode já estar cadastrada no registro de preços.
#
# Use 0 para desabilitar o filtro.
# =============================================================================
DIAS_MINIMOS_PREPARO     = 7   # dias mínimos até encerramento — licitações comuns
DIAS_MINIMOS_PREPARO_SRP =  7   # dias mínimos até encerramento — licitações SRP

# =============================================================================
# CONFIGURAÇÃO — CAMPOS DO JSON DE SAÍDA
# True  = gravar o campo no arquivo de saída
# False = descartar o campo
#
# Campos de nível raiz da resposta da API (endpoint /v1/contratacoes/proposta).
# Campos aninhados (orgaoEntidade, unidadeOrgao, etc.) são controlados
# pelo campo pai: True grava o objeto inteiro, False descarta tudo.
# =============================================================================
CAMPOS = {
    # -- Identificação da contratação --
    "numeroControlePNCP":              True,
    "numeroCompra":                    True,
    "anoCompra":                       True,
    "processo":                        True,
    "sequencialCompra":                True,

    # -- Instrumento convocatório --
    "tipoInstrumentoConvocatorioId":   True,
    "tipoInstrumentoConvocatorioNome": True,

    # -- Modalidade e disputa --
    "modalidadeId":                    True,
    "modalidadeNome":                  True,
    "modoDisputaId":                   True,
    "modoDisputaNome":                 True,

    # -- Situação --
    "situacaoCompraId":                True,
    "situacaoCompraNome":              True,

    # -- Objeto --
    "objetoCompra":                    True,
    "informacaoComplementar":          True,

    # -- SRP --
    "srp":                             True,

    # -- Amparo legal (objeto aninhado) --
    "amparoLegal":                     True,

    # -- Valores --
    "valorTotalEstimado":              True,
    "valorTotalHomologado":            True,

    # -- Datas --
    "dataAberturaProposta":            True,
    "dataEncerramentoProposta":        True,
    "dataPublicacaoPncp":              True,
    "dataInclusao":                    True,
    "dataAtualizacao":                 True,

    # -- Órgão (objeto aninhado: cnpj, razaosocial, poderId, esferaId) --
    "orgaoEntidade":                   True,

    # -- Unidade administrativa (objeto aninhado) --
    "unidadeOrgao":                    True,

    # -- Órgão sub-rogado (objeto aninhado, pode ser null) --
    "orgaoSubRogado":                  True,

    # -- Unidade sub-rogada (objeto aninhado, pode ser null) --
    "unidadeSubRogada":                True,

    # -- Portal de origem --
    "usuarioNome":                     True,
    "linkSistemaOrigem":               True,
    "linkProcessoEletronico":          True,

    # -- Outros --
    "justificativaPresencial":         True,

    # -- Campos calculados (não vêm da API — gerados pelo scanner) --
    # Documentos (6.3.8): retorna lista com url, tipoDocumentoId, titulo.
    #   Tipos de interesse: 2=Edital | 4=TR | 7=ETP | 6=Projeto Básico
    #   Sem autenticação — documentos hospedados no próprio PNCP.
    "url_documentos_pncp":             True,

    # Itens (6.3.13): retorna lista paginada de itens com descrição, quantidade,
    #   valorUnitarioEstimado, NCM, tipoBenefício, etc.
    #   Útil para análise de aderência mesmo quando o edital PDF não está disponível.
    #   Sem autenticação.
    "url_itens_pncp":                  True,
}

# =============================================================================
# CONFIGURAÇÃO — API
# =============================================================================
BASE_URL      = "https://pncp.gov.br/api/consulta"
BASE_URL_PNCP = "https://pncp.gov.br/api/pncp"     # API de Integração (documentos/itens)
ENDPOINT      = "/v1/contratacoes/proposta"
TAMANHO_PAG   = 50 #50          # padrão da API; testar 100/200 se estável
TIMEOUT       = (10, 30)     # (connect, read) em segundos
PAUSA_PAG     = 0.5 #0.3          # segundos entre páginas da mesma modalidade
PAUSA_MOD     = 1.0          # segundos entre modalidades
MAX_TENTATIVAS = 5           # retentativas em caso de erro HTTP 5xx

HEADERS = {
    "Accept":     "application/json",
    "User-Agent": "PNCP-Scanner/1.0",
}

# =============================================================================
# CONFIGURAÇÃO — SAÍDA
# =============================================================================
OUTPUT_FILE    = "pncp_licitacoes.json"
EXCLUSOES_FILE = "exclusoes.yaml"   # lista de termos para exclusão de objetos irrelevantes

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def calcular_data_final() -> str:
    """Retorna dataFinal = hoje + JANELA_DIAS no formato AAAAMMDD."""
    data = date.today() + timedelta(days=JANELA_DIAS)
    return data.strftime("%Y%m%d")


def filtrar_campos(registro: dict) -> dict:
    """Retorna apenas os campos configurados como True em CAMPOS."""
    return {k: v for k, v in registro.items() if CAMPOS.get(k, True)}


def calcular_url_documentos(registro: dict) -> str | None:
    """
    Monta a URL do endpoint de documentos do PNCP a partir dos campos
    orgaoEntidade.cnpj, anoCompra e sequencialCompra.

    Endpoint: GET /v1/orgaos/{cnpj}/compras/{ano}/{sequencial}/arquivos
    Retorna lista de documentos com tipo e URL de download direto.
    Tipos de interesse: 2=Edital | 4=Termo de Referência | 7=ETP

    Não requer autenticação — documentos hospedados no próprio PNCP.
    """
    try:
        cnpj       = registro["orgaoEntidade"]["cnpj"]
        ano        = registro["anoCompra"]
        sequencial = registro["sequencialCompra"]
        if cnpj and ano and sequencial:
            return f"{BASE_URL_PNCP}/v1/orgaos/{cnpj}/compras/{ano}/{sequencial}/arquivos"
    except (KeyError, TypeError):
        pass
    return None


def calcular_url_itens(registro: dict) -> str | None:
    """
    Monta a URL do endpoint de itens do PNCP a partir dos campos
    orgaoEntidade.cnpj, anoCompra e sequencialCompra.

    Endpoint: GET /v1/orgaos/{cnpj}/compras/{ano}/{sequencial}/itens
    Retorna lista paginada de itens com descrição, quantidade, valor estimado,
    NCM, tipo de benefício, etc.  Suporta ?pagina=&tamanhoPagina=.

    Não requer autenticação.
    """
    try:
        cnpj       = registro["orgaoEntidade"]["cnpj"]
        ano        = registro["anoCompra"]
        sequencial = registro["sequencialCompra"]
        if cnpj and ano and sequencial:
            return f"{BASE_URL_PNCP}/v1/orgaos/{cnpj}/compras/{ano}/{sequencial}/itens"
    except (KeyError, TypeError):
        pass
    return None


def _normalizar(texto: str) -> str:
    """Remove acentos e aplica casefold para comparação normalizada."""
    return (
        unicodedata.normalize("NFD", texto)
        .encode("ascii", "ignore")
        .decode()
        .casefold()
    )


# Patterns compilados uma vez ao carregar o arquivo de exclusões.
_EXCLUSOES_PATTERNS: list[tuple[str, re.Pattern]] = []


def carregar_exclusoes() -> int:
    """
    Lê EXCLUSOES_FILE e compila os padrões de exclusão.
    Cada termo é normalizado e compilado como prefixo de palavra:
      "obra"      → \\bobra\\w*  captura "obra", "obras"
      "construca" → \\bconstruca\\w*  captura "construção", "construções"
    Frases multi-palavra também funcionam: "servico de engenharia".
    Retorna o número de termos carregados (0 = filtro desabilitado).
    """
    global _EXCLUSOES_PATTERNS
    _EXCLUSOES_PATTERNS = []

    try:
        with open(EXCLUSOES_FILE, encoding="utf-8") as f:
            dados = yaml.safe_load(f)
    except FileNotFoundError:
        log.warning("'%s' não encontrado — filtro de exclusões desabilitado.", EXCLUSOES_FILE)
        return 0
    except Exception as exc:
        log.error("Erro ao ler '%s': %s — filtro desabilitado.", EXCLUSOES_FILE, exc)
        return 0

    termos = dados.get("termos_exclusao") or []
    for termo in termos:
        if not termo:
            continue
        termo_norm = _normalizar(str(termo))
        # Prefixo de palavra: \bTERMO\w*
        # Cobre singular/plural e flexões sem necessidade de stemmer.
        pattern = re.compile(r"\b" + re.escape(termo_norm) + r"\w*")
        _EXCLUSOES_PATTERNS.append((termo, pattern))

    log.info("Exclusões carregadas: %d termos de '%s'", len(_EXCLUSOES_PATTERNS), EXCLUSOES_FILE)
    return len(_EXCLUSOES_PATTERNS)


def aplicar_filtro_exclusoes(registros: list) -> list:
    """
    Descarta registros cujo objetoCompra ou informacaoComplementar contenha
    algum dos termos de exclusão definidos em EXCLUSOES_FILE.
    Se nenhum padrão estiver carregado, retorna a lista sem alterações.
    """
    if not _EXCLUSOES_PATTERNS:
        return registros

    aprovados = []
    descartados = 0

    for reg in registros:
        texto = " ".join(filter(None, [
            reg.get("objetoCompra") or "",
            reg.get("informacaoComplementar") or "",
        ]))
        texto_norm = _normalizar(texto)

        termo_encontrado = next(
            (termo for termo, pat in _EXCLUSOES_PATTERNS if pat.search(texto_norm)),
            None,
        )

        if termo_encontrado:
            descartados += 1
            log.debug(
                "Excluído [%s] (termo: '%s'): %.80s",
                reg.get("numeroControlePNCP", "?"),
                termo_encontrado,
                reg.get("objetoCompra", ""),
            )
        else:
            aprovados.append(reg)

    if descartados:
        log.info(
            "Filtro de exclusões: %d descartados | %d mantidos",
            descartados, len(aprovados),
        )

    return aprovados


def aplicar_filtro_amparo_legal(registros: list) -> list:
    """
    Filtra registros pelo campo amparoLegal.codigo conforme FILTRO_AMPARO_LEGAL.
    Lei 14.133/2021 — Art. 28:
      codigo = 1: bens e serviços comuns (pregão obrigatório)
      codigo = 2: bens/serviços especiais + obras/serviços de engenharia
    """
    if FILTRO_AMPARO_LEGAL is None:
        return registros

    aprovados = []
    descartados = 0

    for reg in registros:
        codigo = None
        try:
            codigo = reg.get("amparoLegal", {}).get("codigo")
        except (TypeError, AttributeError):
            pass

        if codigo in FILTRO_AMPARO_LEGAL:
            aprovados.append(reg)
        else:
            descartados += 1
            log.debug(
                "Excluído por amparo legal [%s]: codigo=%s (%s)",
                reg.get("numeroControlePNCP", "?"),
                codigo,
                reg.get("amparoLegal", {}).get("nome", "desconhecido"),
            )

    if descartados:
        log.info(
            "Filtro de amparo legal: %d descartados (fora de %s) | %d mantidos",
            descartados, FILTRO_AMPARO_LEGAL, len(aprovados),
        )

    return aprovados


def aplicar_filtro_srp(registros: list) -> list:
    """Filtra registros pelo campo srp conforme FILTRO_SRP."""
    if FILTRO_SRP is None:
        return registros
    return [r for r in registros if r.get("srp") is FILTRO_SRP]


def aplicar_filtro_prazo(registros: list) -> list:
    """
    Descarta licitações cujo dataEncerramentoProposta seja inferior ao
    prazo mínimo de preparo configurado.

    - Licitações comuns: exige DIAS_MINIMOS_PREPARO dias até o encerramento.
    - Licitações SRP   : exige DIAS_MINIMOS_PREPARO_SRP dias (geralmente menor).
    - Se ambos os valores forem 0, o filtro é ignorado.
    - Registros sem dataEncerramentoProposta são mantidos (sem info = sem descarte).
    """
    if DIAS_MINIMOS_PREPARO == 0 and DIAS_MINIMOS_PREPARO_SRP == 0:
        return registros

    hoje = date.today()
    aprovados = []
    descartados = 0

    for reg in registros:
        encerramento_str = reg.get("dataEncerramentoProposta")

        if not encerramento_str:
            aprovados.append(reg)
            continue

        try:
            # Formato da API: "2026-05-10T09:00:00" ou "2026-05-10"
            encerramento = date.fromisoformat(encerramento_str[:10])
        except ValueError:
            aprovados.append(reg)
            continue

        is_srp  = reg.get("srp", False)
        minimo  = DIAS_MINIMOS_PREPARO_SRP if is_srp else DIAS_MINIMOS_PREPARO
        prazo   = (encerramento - hoje).days

        if prazo >= minimo:
            aprovados.append(reg)
        else:
            descartados += 1

    if descartados:
        log.info(
            "Filtro de prazo: %d descartados (< %d dias SRP / %d dias comum) | %d mantidos",
            descartados, DIAS_MINIMOS_PREPARO_SRP, DIAS_MINIMOS_PREPARO, len(aprovados),
        )

    return aprovados


def nome_modalidade(codigo: int) -> str:
    nomes = {
        1: "Leilão Eletrônico",
        2: "Diálogo Competitivo",
        3: "Concurso",
        4: "Concorrência Eletrônica",
        5: "Concorrência Presencial",
        6: "Pregão Eletrônico",
        7: "Pregão Presencial",
        8: "Dispensa de Licitação",
        9: "Inexigibilidade",
        10: "Manifestação de Interesse",
        11: "Pré-qualificação",
        12: "Credenciamento",
        13: "Leilão Presencial",
    }
    return nomes.get(codigo, f"Modalidade {codigo}")


# =============================================================================
# FUNÇÕES DE CONSULTA À API
# =============================================================================

def buscar_pagina(data_final: str, cod_modalidade: int, pagina: int, uf: str | None = None) -> dict | None:
    """
    Consulta uma página da API /v1/contratacoes/proposta.
    Retorna o dict JSON da resposta ou None em caso de falha permanente.
    """
    url    = BASE_URL + ENDPOINT
    params = {
        "dataFinal":                    data_final,
        "codigoModalidadeContratacao":  cod_modalidade,
        "pagina":                       pagina,
        "tamanhoPagina":                TAMANHO_PAG,
    }
    if uf:
        params["uf"] = uf

    for tentativa in range(1, MAX_TENTATIVAS + 1):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code == 204:
                # Sem conteúdo — resultado vazio, não é erro
                return {"data": [], "totalRegistros": 0, "totalPaginas": 0,
                        "numeroPagina": pagina, "paginasRestantes": 0, "empty": True}

            if resp.status_code in (400, 422):
                log.error("Parâmetros inválidos (HTTP %s): %s", resp.status_code, resp.text[:200])
                return None  # erro permanente — não tentar novamente

            if resp.status_code >= 500:
                log.warning(
                    "HTTP %s na tentativa %d/%d (modalidade=%d, página=%d)",
                    resp.status_code, tentativa, MAX_TENTATIVAS, cod_modalidade, pagina,
                )
                if tentativa < MAX_TENTATIVAS:
                    time.sleep(2 ** tentativa)  # backoff exponencial
                    continue
                return None

            log.warning("HTTP inesperado %s: %s", resp.status_code, resp.text[:200])
            return None

        except requests.exceptions.Timeout:
            log.warning("Timeout na tentativa %d/%d (modalidade=%d, página=%d)",
                        tentativa, MAX_TENTATIVAS, cod_modalidade, pagina)
            if tentativa < MAX_TENTATIVAS:
                time.sleep(2 ** tentativa)
        except requests.exceptions.RequestException as exc:
            log.error("Erro de conexão: %s", exc)
            return None

    return None


def coletar_combinacao(data_final: str, cod_modalidade: int, uf: str | None) -> list:
    """
    Pagina completamente a API para uma combinação modalidade+UF e retorna
    a lista de todos os registros encontrados.
    """
    label_uf = uf if uf else "todos os estados"
    log.info("  [%s] paginando...", label_uf)

    todos  = []
    pagina = 1

    while True:
        resultado = buscar_pagina(data_final, cod_modalidade, pagina, uf)

        if resultado is None:
            log.error("  [%s] Falha permanente na página %d — interrompendo.", label_uf, pagina)
            break

        registros      = resultado.get("data") or []
        total_pags     = resultado.get("totalPaginas", 1) or 1
        pags_restantes = resultado.get("paginasRestantes", 0) or 0
        total_regs     = resultado.get("totalRegistros", 0) or 0

        todos.extend(registros)
        log.info("  [%s] pág. %d/%d — %d registros (total: %d)",
                 label_uf, pagina, total_pags, len(registros), total_regs)

        if pags_restantes == 0 or pagina >= total_pags or not registros:
            break

        pagina += 1
        time.sleep(PAUSA_PAG)

    return todos


def coletar_modalidade(data_final: str, cod_modalidade: int, ufs_ativas: list) -> list:
    """
    Coleta todos os registros de uma modalidade, iterando sobre as UFs ativas.
    Se ufs_ativas estiver vazio, faz uma única consulta sem filtro de UF
    (retorna todo o Brasil).
    """
    label = nome_modalidade(cod_modalidade)
    log.info("▶ Modalidade %d — %s | UFs: %s", cod_modalidade, label,
             ufs_ativas if ufs_ativas else "todas")

    todos = []

    if ufs_ativas:
        for uf in ufs_ativas:
            registros = coletar_combinacao(data_final, cod_modalidade, uf)
            todos.extend(registros)
            time.sleep(PAUSA_PAG)
    else:
        todos = coletar_combinacao(data_final, cod_modalidade, uf=None)

    log.info("✔ Modalidade %d concluída: %d registros coletados.", cod_modalidade, len(todos))
    return todos


# =============================================================================
# MAIN
# =============================================================================

def main():
    data_final         = calcular_data_final()
    modalidades_ativas = [cod for cod, ativo in MODALIDADES.items() if ativo]
    ufs_ativas         = [uf  for uf,  ativo in UFS.items()       if ativo]

    n_exclusoes = carregar_exclusoes()

    log.info("=" * 60)
    log.info("PNCP Scanner — início")
    log.info("dataFinal    : %s (hoje + %d dias)", data_final, JANELA_DIAS)
    log.info("Modalidades  : %s", modalidades_ativas)
    log.info("UFs ativas   : %s", ufs_ativas if ufs_ativas else "todas (sem filtro)")
    log.info("Amparo legal : %s", FILTRO_AMPARO_LEGAL if FILTRO_AMPARO_LEGAL is not None else "todos")
    log.info("Filtro SRP   : %s", FILTRO_SRP)
    log.info("Prazo mínimo : %d dias (comum) | %d dias (SRP)", DIAS_MINIMOS_PREPARO, DIAS_MINIMOS_PREPARO_SRP)
    log.info("Exclusões    : %d termos (camada extra)", n_exclusoes)
    log.info("Saída        : %s", OUTPUT_FILE)
    log.info("=" * 60)

    if not modalidades_ativas:
        log.error("Nenhuma modalidade habilitada em MODALIDADES. Encerrando.")
        return

    # -------------------------------------------------------------------------
    # Coleta por modalidade × UF
    # -------------------------------------------------------------------------
    todos_brutos: list = []

    for i, cod in enumerate(modalidades_ativas):
        registros = coletar_modalidade(data_final, cod, ufs_ativas)
        todos_brutos.extend(registros)

        if i < len(modalidades_ativas) - 1:
            time.sleep(PAUSA_MOD)

    log.info("Total bruto coletado (todas as modalidades): %d registros", len(todos_brutos))

    # -------------------------------------------------------------------------
    # Deduplicação por numeroControlePNCP
    # (mesmo registro pode aparecer em mais de uma consulta futuramente)
    # -------------------------------------------------------------------------
    vistos: set = set()
    sem_duplicatas: list = []
    for reg in todos_brutos:
        chave = reg.get("numeroControlePNCP") or id(reg)
        if chave not in vistos:
            vistos.add(chave)
            sem_duplicatas.append(reg)

    duplicatas = len(todos_brutos) - len(sem_duplicatas)
    if duplicatas:
        log.info("Duplicatas removidas: %d", duplicatas)

    # -------------------------------------------------------------------------
    # Filtro de amparo legal (1ª camada — estrutural, baseado em lei)
    # -------------------------------------------------------------------------
    filtrados = aplicar_filtro_amparo_legal(sem_duplicatas)

    # -------------------------------------------------------------------------
    # Filtro SRP
    # -------------------------------------------------------------------------
    filtrados = aplicar_filtro_srp(filtrados)
    if FILTRO_SRP is not None:
        log.info("Após filtro SRP=%s: %d registros", FILTRO_SRP, len(filtrados))

    # -------------------------------------------------------------------------
    # Filtro de prazo mínimo de preparo
    # -------------------------------------------------------------------------
    filtrados = aplicar_filtro_prazo(filtrados)

    # -------------------------------------------------------------------------
    # Filtro de exclusões por objeto (2ª camada — textual, camada extra de segurança)
    # -------------------------------------------------------------------------
    filtrados = aplicar_filtro_exclusoes(filtrados)

    # -------------------------------------------------------------------------
    # Aplicar seleção de campos
    # -------------------------------------------------------------------------
    campos_ativos = sum(1 for v in CAMPOS.values() if v)
    campos_totais = len(CAMPOS)
    log.info("Campos configurados: %d/%d ativos", campos_ativos, campos_totais)

    # Enriquecer com URL calculada ANTES de filtrar campos,
    # para que url_documentos_pncp passe pelo filtro de CAMPOS normalmente.
    for r in filtrados:
        r["url_documentos_pncp"] = calcular_url_documentos(r)
        r["url_itens_pncp"]      = calcular_url_itens(r)

    resultado_final = [filtrar_campos(r) for r in filtrados]

    # -------------------------------------------------------------------------
    # Gravar JSON de saída
    # -------------------------------------------------------------------------
    saida = {
        "_meta": {
            "gerado_em":         date.today().isoformat(),
            "data_final_busca":  data_final,
            "janela_dias":             JANELA_DIAS,
            "modalidades":             modalidades_ativas,
            "ufs":                     ufs_ativas if ufs_ativas else "todas",
            "filtro_amparo_legal":      FILTRO_AMPARO_LEGAL,
            "filtro_srp":              FILTRO_SRP,
            "dias_minimos_preparo":    DIAS_MINIMOS_PREPARO,
            "dias_minimos_preparo_srp": DIAS_MINIMOS_PREPARO_SRP,
            "termos_exclusao":   n_exclusoes,
            "total_registros":   len(resultado_final),
            "campos_ativos":     campos_ativos,
        },
        "licitacoes": resultado_final,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(saida, f, ensure_ascii=False, indent=2)

    log.info("=" * 60)
    log.info("Arquivo gravado: %s", OUTPUT_FILE)
    log.info("Total de licitações: %d", len(resultado_final))
    log.info("=" * 60)


if __name__ == "__main__":
    main()
