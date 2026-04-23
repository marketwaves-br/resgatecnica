# -*- coding: utf-8 -*-
"""
pncp_documentos.py
Baixa editais, TRs e ETPs do PNCP para as licitações listadas em
pncp_licitacoes.json, seguindo uma estratégia de fallback de 4 estágios.

Estratégia de fallback (por licitação):
  A → /arquivos PNCP   (download direto de PDF hospedado no PNCP)
  B → /itens PNCP      (coleta estruturada de itens — sem PDF)
  C → Playwright       (portal externo via linkSistemaOrigem)
  D → Alerta manual    (nenhuma fonte funcionou — revisão humana)

Estrutura de saída por licitação:
  editais/{ctrl}/
    arquivos.json       ← resposta bruta do endpoint /arquivos
    itens.json          ← itens coletados do endpoint /itens
    docs/
      {ctrl}_Edital.pdf
      {ctrl}_Termo_Referencia.docx
      {ctrl}_Planilha_precos.xlsx
      ...

APIs usadas (base diferente do scanner!):
  Base: https://pncp.gov.br/api/pncp
  GET /v1/orgaos/{cnpj}/compras/{ano}/{seq}/arquivos  (seção 6.3.8)
  GET /v1/orgaos/{cnpj}/compras/{ano}/{seq}/itens     (seção 6.3.13)

Dependências opcionais (instalar uma vez):
  pip install py7zr rarfile
  RAR também requer UnRAR.exe no PATH (https://www.rarlab.com/rar_add.htm)
"""

import json
import logging
import re
import shutil
import tarfile
import time
import unicodedata
import zipfile
from datetime import datetime
from pathlib import Path

import requests

# =============================================================================
# CONFIGURAÇÃO — ENTRADA/SAÍDA
# =============================================================================
INPUT_FILE     = "pncp_licitacoes.json"  # gerado pelo pncp_scanner.py
OUTPUT_DIR     = "editais"               # pasta raiz; subpastas por licitação
MANIFESTO_FILE = "pncp_documentos.json"  # manifesto de status e metadados

# =============================================================================
# CONFIGURAÇÃO — TIPOS DE DOCUMENTO DE INTERESSE
# Referência: PNCP API — seção 6.3.8 (tipoDocumentoId)
#   2 = Edital  |  4 = Termo de Referência  |  6 = Projeto Básico  |  7 = ETP
# =============================================================================
TIPOS_DOCUMENTO = [2, 4, 7]

# =============================================================================
# CONFIGURAÇÃO — EXTENSÕES LEGÍVEIS PELO USUÁRIO
# Arquivos com essas extensões são considerados prontos para uso.
# Qualquer outro formato recebe detecção por magic bytes e/ou extração.
# =============================================================================
EXTENSOES_LEITURA = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".odt", ".ods", ".odp",
    ".html", ".htm", ".txt",
}

# =============================================================================
# CONFIGURAÇÃO — FORMATOS COMPACTADOS (desencadeiam extração recursiva)
# =============================================================================
FORMATOS_COMPACTADOS = {".zip", ".gz", ".bz2", ".tar", ".tgz", ".7z", ".rar"}

# Profundidade máxima de extração recursiva (proteção contra zip bombs)
MAX_NIVEL_EXTRACAO = 5

# =============================================================================
# CONFIGURAÇÃO — MANTER COMPACTADO APÓS EXTRAÇÃO
#   True  = após extrair com sucesso, move o arquivo compactado original
#           (já com extensão correta) para docs/ junto com os extraídos.
#           Útil para verificar se todos os arquivos internos foram copiados.
#   False = apaga o compactado após extração bem-sucedida (comportamento padrão)
# =============================================================================
MANTER_COMPACTADO = True

# =============================================================================
# CONFIGURAÇÃO — LIMITE DE PATH NO WINDOWS
# MAX_PATH = 260. Base estimada: D:\Edward\Resgatecnica\editais\{ctrl}\docs\
# ≈ 90 caracteres. Sobram 170 para prefixo+stem+extensão.
# =============================================================================
_MAX_FILENAME = 170   # caracteres disponíveis para o nome do arquivo
_BASE_PATH_LEN = 90   # estimativa do caminho base (conservadora)

# =============================================================================
# CONFIGURAÇÃO — ESTRATÉGIA C (PLAYWRIGHT)
# Requer: pip install playwright && playwright install chromium
#   False = desabilitado (padrão)
#   True  = tentar portal externo se A falhar sem documentos
# =============================================================================
USAR_PLAYWRIGHT = False

# =============================================================================
# CONFIGURAÇÃO — FASE DE DOWNLOAD
#   False (padrão) = fase 1: coleta apenas itens.json via Estratégia B.
#                    Rápido. Execute ANTES do pncp_agente.py.
#   True           = fase 2: baixa também PDFs/editais (Estratégias A e C).
#                    Execute DEPOIS do pncp_agente.py, quando já souber quais
#                    licitações são aderentes.
# =============================================================================
BAIXAR_DOCUMENTOS = False

# =============================================================================
# CONFIGURAÇÃO — PROCESSAMENTO INCREMENTAL
#   True  = pular licitações já presentes no manifesto
#   False = reprocessar metadados (mas nunca re-baixar arquivos existentes)
#
# Cache de arquivos é independente: apague a pasta do item para forçar
# novo download.
# =============================================================================
INCREMENTAL = True

# =============================================================================
# CONFIGURAÇÃO — LIMITE DE ITENS NO MANIFESTO
# itens.json sempre completo; manifesto salva apenas amostra.
# None = salvar todos no manifesto.
# =============================================================================
MAX_ITENS_MANIFESTO = 10

# =============================================================================
# CONFIGURAÇÃO — API
# =============================================================================
BASE_URL_PNCP  = "https://pncp.gov.br/api/pncp"
TIMEOUT        = (10, 60)
PAUSA_LICITA   = 1.0
PAUSA_DOC      = 0.5
MAX_TENTATIVAS = 3

HEADERS = {
    "Accept":     "application/json",
    "User-Agent": "PNCP-Scanner/1.0",
}
HEADERS_DOWNLOAD = {
    "User-Agent": "PNCP-Scanner/1.0",
}

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
# UTILITÁRIOS DE CAMINHO E NOME
# =============================================================================

def sanitizar_nome(texto: str) -> str:
    """Remove acentos e caracteres inválidos para nomes de pasta/arquivo."""
    texto = unicodedata.normalize("NFD", texto).encode("ascii", "ignore").decode()
    texto = re.sub(r'[\\/:*?"<>|]', "_", texto)
    return texto.strip()[:100]


def _prefixo_pasta(licitacao: dict | None) -> str:
    """
    Retorna '{UF}_{CIDADE}_' quando os dados estão disponíveis, '' caso contrário.
    Exemplo: 'MG_SETE_LAGOAS_'
    """
    if not licitacao:
        return ""
    unidade = licitacao.get("unidadeOrgao") or {}
    uf  = (unidade.get("ufSigla")      or "").strip().upper()
    mun = (unidade.get("municipioNome") or "").strip().upper()
    if not uf and not mun:
        return ""
    uf_norm  = sanitizar_nome(uf)
    mun_norm = sanitizar_nome(mun).replace(" ", "_")
    return f"{uf_norm}_{mun_norm}_"


def pasta_licitacao(ctrl: str, licitacao: dict | None = None) -> Path:
    """
    Pasta raiz da licitação.
    Com licitacao fornecida, inclui prefixo '{UF}_{CIDADE}_'.
    Exemplo: editais/MG_SETE_LAGOAS_97550393000149-1-000003_2026/
    """
    ctrl_norm = sanitizar_nome(ctrl.replace("/", "_"))
    nome = _prefixo_pasta(licitacao) + ctrl_norm
    return Path(OUTPUT_DIR) / nome


def pasta_docs(ctrl: str, licitacao: dict | None = None) -> Path:
    """Subpasta docs/ onde ficam os documentos legíveis."""
    return pasta_licitacao(ctrl, licitacao) / "docs"


def _nome_doc(ctrl: str, nome_arquivo: str) -> str:
    """
    Gera '{ctrl}_{nome}' respeitando o limite de PATH do Windows.

    Exemplo: '18715391000196-1-000020_2026_Edital.pdf'
    """
    prefixo = sanitizar_nome(ctrl) + "_"
    p       = Path(nome_arquivo)
    ext     = p.suffix          # mantém extensão original
    stem    = sanitizar_nome(p.stem)

    max_stem = max(_MAX_FILENAME - len(prefixo) - len(ext), 20)
    return prefixo + stem[:max_stem] + ext


def arquivos_legiveis(pasta: Path) -> list[Path]:
    """Retorna arquivos com extensão legível existentes em pasta/docs/."""
    docs = pasta / "docs"
    if not docs.exists():
        return []
    return [f for f in docs.iterdir() if f.is_file() and f.suffix.lower() in EXTENSOES_LEITURA]


# =============================================================================
# HTTP
# =============================================================================

def get_com_retry(url: str, headers: dict | None = None, stream: bool = False) -> requests.Response | None:
    """GET com retry exponencial em erros 5xx."""
    h = headers or HEADERS
    for tentativa in range(1, MAX_TENTATIVAS + 1):
        try:
            resp = requests.get(url, headers=h, timeout=TIMEOUT, stream=stream)

            if resp.status_code == 200:
                return resp
            if resp.status_code == 204:
                return resp
            if resp.status_code in (400, 404, 422):
                log.debug("HTTP %s (permanente): %s", resp.status_code, url)
                return None
            if resp.status_code >= 500:
                log.warning("HTTP %s tentativa %d/%d: %s",
                            resp.status_code, tentativa, MAX_TENTATIVAS, url)
                if tentativa < MAX_TENTATIVAS:
                    time.sleep(2 ** tentativa)
                    continue
                return None
            log.warning("HTTP %s inesperado: %s", resp.status_code, url)
            return None

        except requests.exceptions.Timeout:
            log.warning("Timeout tentativa %d/%d: %s", tentativa, MAX_TENTATIVAS, url)
            if tentativa < MAX_TENTATIVAS:
                time.sleep(2 ** tentativa)
        except requests.exceptions.RequestException as exc:
            log.error("Erro de conexao: %s -- %s", exc, url)
            return None

    return None


# =============================================================================
# DETECÇÃO DE TIPO DE ARQUIVO
# =============================================================================

def _extensao_do_content_type(content_type: str) -> str:
    """
    Infere extensão pelo Content-Type.
    Retorna "" quando ambíguo (octet-stream) — será resolvido por magic bytes.
    """
    ct = (content_type or "").split(";")[0].strip().lower()
    mapa = {
        "application/pdf":   ".pdf",
        "application/msword": ".doc",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/vnd.ms-excel": ".xls",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        "application/zip":              ".zip",
        "application/x-zip-compressed": ".zip",
        "application/x-rar-compressed": ".rar",
        "application/vnd.rar":          ".rar",
        "application/x-7z-compressed":  ".7z",
        "application/octet-stream":     "",    # ambíguo → magic bytes
        "text/html":  ".html",
        "text/plain": ".txt",
    }
    return mapa.get(ct, "")


_MAGIC: list[tuple[bytes, str]] = [
    (b"%PDF",              ".pdf"),
    (b"PK\x03\x04",       ".zip"),
    (b"PK\x05\x06",       ".zip"),
    (b"\xd0\xcf\x11\xe0", ".doc"),
    (b"Rar!\x1a\x07\x01", ".rar"),   # RAR5
    (b"Rar!\x1a\x07\x00", ".rar"),   # RAR4
    (b"\x1f\x8b",         ".gz"),
    (b"7z\xbc\xaf'\"",    ".7z"),
    (b"BZh",              ".bz2"),
]


def _detectar_extensao_por_magic(caminho: Path) -> str:
    """Lê os primeiros bytes do arquivo e retorna a extensão correspondente."""
    try:
        with open(caminho, "rb") as fh:
            header = fh.read(10)
        for magic, ext in _MAGIC:
            if header[: len(magic)] == magic:
                return ext
    except OSError:
        pass
    return ".bin"


def _extensao_office_em_zip(z: zipfile.ZipFile) -> str | None:
    """
    Detecta se um ZIP é na verdade um documento Office (OOXML ou ODF).
    Retorna a extensão correta ou None se for ZIP genérico.
    """
    nomes = set(z.namelist())
    if "word/document.xml" in nomes:
        return ".docx"
    if "xl/workbook.xml" in nomes:
        return ".xlsx"
    if "ppt/presentation.xml" in nomes:
        return ".pptx"
    if "mimetype" in nomes:
        try:
            mime = z.read("mimetype").decode("ascii", errors="ignore").strip()
            odf = {
                "application/vnd.oasis.opendocument.text":         ".odt",
                "application/vnd.oasis.opendocument.spreadsheet":  ".ods",
                "application/vnd.oasis.opendocument.presentation": ".odp",
            }
            return odf.get(mime)
        except Exception:
            pass
    return None


# =============================================================================
# EXTRATORES POR FORMATO
# =============================================================================

def _tentar_zip(arquivo: Path, destino: Path) -> bool:
    try:
        with zipfile.ZipFile(arquivo, "r") as z:
            z.extractall(destino)
        return True
    except Exception as exc:
        log.warning("    ZIP invalido '%s': %s", arquivo.name, exc)
        return False


def _tentar_tar(arquivo: Path, destino: Path) -> bool:
    try:
        with tarfile.open(arquivo) as t:
            t.extractall(destino)
        return True
    except Exception as exc:
        log.warning("    TAR invalido '%s': %s", arquivo.name, exc)
        return False


def _tentar_7z(arquivo: Path, destino: Path) -> bool:
    try:
        import py7zr  # type: ignore
        with py7zr.SevenZipFile(arquivo, "r") as z:
            z.extractall(destino)
        return True
    except ImportError:
        log.warning("    py7zr nao instalado — instale: pip install py7zr")
        return False
    except Exception as exc:
        log.warning("    7z invalido '%s': %s", arquivo.name, exc)
        return False


def _tentar_rar(arquivo: Path, destino: Path) -> bool:
    try:
        import rarfile  # type: ignore
        with rarfile.RarFile(arquivo) as r:
            r.extractall(destino)
        return True
    except ImportError:
        log.warning("    rarfile nao instalado — instale: pip install rarfile + UnRAR no PATH")
        return False
    except Exception as exc:
        log.warning("    RAR invalido '%s': %s", arquivo.name, exc)
        return False


# =============================================================================
# EXTRAÇÃO RECURSIVA
# =============================================================================

def _extrair_para_docs(arquivo: Path, docs: Path, ctrl: str, nivel: int = 0) -> list[Path]:
    """
    Processa um arquivo para a pasta docs/ de forma recursiva:
      • Sem extensão / .bin  → detecta tipo por magic bytes antes de tudo
      • ZIP Office (DOCX/XLSX…) → renomeia com extensão correta → docs/
      • Compactado (ZIP/RAR/7z/TAR…) → extrai para área temporária →
            cada arquivo extraído passa por _extrair_para_docs (recursão)
            documentos finais → docs/
            compactados intermediários → docs/compactados/ (se MANTER_COMPACTADO)
      • Arquivo legível / desconhecido → docs/ com prefixo ctrl

    Limite de recursão: MAX_NIVEL_EXTRACAO (proteção contra zip bombs).
    Retorna lista dos arquivos finais em docs/.
    """
    docs.mkdir(parents=True, exist_ok=True)

    # ── Detectar tipo quando extensão ausente ou desconhecida ────────────────
    # Cobre arquivos sem extensão extraídos de ZIPs e ".bin" residuais.
    if arquivo.suffix.lower() in ("", ".bin"):
        ext_detectada = _detectar_extensao_por_magic(arquivo)
        if ext_detectada != arquivo.suffix.lower():
            novo = arquivo.with_suffix(ext_detectada)
            arquivo.rename(novo)
            arquivo = novo
            log.info("    Extensao detectada: %s -> %s", arquivo.with_suffix("").name, ext_detectada)

    ext = arquivo.suffix.lower()

    # ── ZIP: detectar Office antes de extrair ────────────────────────────────
    if ext == ".zip":
        try:
            with zipfile.ZipFile(arquivo, "r") as z:
                ext_office = _extensao_office_em_zip(z)
            if ext_office:
                destino = docs / _nome_doc(ctrl, arquivo.stem + ext_office)
                arquivo.rename(destino)
                log.info("    Office detectado: %s", destino.name)
                return [destino]
        except Exception:
            pass  # corrompido ou falsa detecção → extrai normalmente

    # ── Arquivo não-compactado: mover para docs/ ─────────────────────────────
    if ext not in FORMATOS_COMPACTADOS:
        destino = docs / _nome_doc(ctrl, arquivo.name)
        arquivo.rename(destino)
        return [destino]

    # ── Limite de recursão atingido ───────────────────────────────────────────
    if nivel >= MAX_NIVEL_EXTRACAO:
        log.warning("    Nivel maximo de extracao (%d) atingido: %s — movido sem extrair",
                    MAX_NIVEL_EXTRACAO, arquivo.name)
        destino = docs / _nome_doc(ctrl, arquivo.name)
        arquivo.rename(destino)
        return [destino]

    # ── Extrair para área temporária ──────────────────────────────────────────
    tmp = arquivo.parent / f"_tmp_{arquivo.stem}_{nivel}"
    tmp.mkdir(exist_ok=True)

    if ext == ".zip":
        ok = _tentar_zip(arquivo, tmp)
    elif ext in (".tar", ".gz", ".bz2", ".tgz"):
        ok = _tentar_tar(arquivo, tmp)
    elif ext == ".7z":
        ok = _tentar_7z(arquivo, tmp)
    elif ext == ".rar":
        ok = _tentar_rar(arquivo, tmp)
    else:
        ok = False

    if not ok:
        # Falha na extração: mover como está para docs/
        shutil.rmtree(tmp, ignore_errors=True)
        destino = docs / _nome_doc(ctrl, arquivo.name)
        arquivo.rename(destino)
        log.info("    Nao extraido — movido: %s", destino.name)
        return [destino]

    # ── Manter ou apagar o compactado original ───────────────────────────────
    if MANTER_COMPACTADO:
        compactados = docs / "compactados"
        compactados.mkdir(exist_ok=True)
        destino_original = compactados / _nome_doc(ctrl, arquivo.name)
        arquivo.rename(destino_original)
        log.info("    Original em compactados/: %s", destino_original.name)
    else:
        arquivo.unlink()

    # ── Processar arquivos extraídos (recursão) ───────────────────────────────
    finais: list[Path] = []
    for f in sorted(tmp.rglob("*")):
        if f.is_file():
            sub = _extrair_para_docs(f, docs, ctrl, nivel + 1)
            finais.extend(sub)

    shutil.rmtree(tmp, ignore_errors=True)

    if finais:
        log.info("    Extraidos %d arquivo(s) (nivel %d)", len(finais), nivel)
    return finais


# =============================================================================
# DOWNLOAD
# =============================================================================

def baixar_arquivo(url: str, caminho_tmp: Path) -> Path | None:
    """
    Baixa url e salva em caminho_tmp (sem extensão = inferida automaticamente).

    Resolução de extensão:
      1. Content-Type do servidor
      2. Magic bytes quando CT é ambíguo ou ausente

    Retorna o Path do arquivo gravado ou None em caso de falha.
    A extração de compactados é responsabilidade do chamador (_extrair_para_docs).
    """
    resp = get_com_retry(url, headers=HEADERS_DOWNLOAD, stream=True)
    if resp is None or resp.status_code != 200:
        return None

    destino = caminho_tmp
    if not destino.suffix:
        ext = _extensao_do_content_type(resp.headers.get("Content-Type", ""))
        destino = destino.with_suffix(ext or ".tmp")

    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    # Corrigir extensão se ficou provisória ou inválida
    if destino.suffix in (".tmp", ".bin", ""):
        ext_real = _detectar_extensao_por_magic(destino)
        novo = destino.with_suffix(ext_real)
        destino.rename(novo)
        destino = novo

    tamanho_kb = destino.stat().st_size // 1024
    log.info("    Baixado: %s (%d KB)", destino.name, tamanho_kb)
    return destino


# =============================================================================
# ESTRATÉGIA A — Download direto de arquivos hospedados no PNCP
# =============================================================================

def estrategia_a(licitacao: dict, pasta: Path, ctrl: str) -> dict:
    """
    Estratégia A: consulta /arquivos, baixa os documentos relevantes e os
    processa (extração recursiva + prefixação) para pasta/docs/.

    Cache: se docs/ já contiver arquivos legíveis, o download é ignorado.
    Para forçar novo download, apague a pasta do processo em editais/.
    """
    # ── Cache ─────────────────────────────────────────────────────────────────
    existentes = arquivos_legiveis(pasta)
    if existentes:
        log.info("  [A] %d arquivo(s) em docs/ — download ignorado", len(existentes))
        return {
            "sucesso": True,
            "cache":   True,
            "documentos": [
                {"titulo": f.stem, "tipo_id": None, "arquivo": f.name, "origem": "cache"}
                for f in existentes
            ],
        }

    # ── Consulta /arquivos ────────────────────────────────────────────────────
    url = licitacao.get("url_documentos_pncp")
    if not url:
        return {"sucesso": False, "documentos": [], "erro": "sem url_documentos_pncp"}

    log.info("  [A] GET %s", url)
    resp = get_com_retry(url)

    if resp is None:
        return {"sucesso": False, "documentos": [], "erro": "falha na requisicao"}
    if resp.status_code == 204:
        return {"sucesso": False, "documentos": [], "erro": "sem documentos (204)"}

    try:
        payload = resp.json()
    except Exception as exc:
        return {"sucesso": False, "documentos": [], "erro": f"JSON invalido: {exc}"}

    arquivos = payload if isinstance(payload, list) else (payload.get("data") or [])

    # Salvar resposta bruta na raiz do processo
    pasta.mkdir(parents=True, exist_ok=True)
    with open(pasta / "arquivos.json", "w", encoding="utf-8") as f:
        json.dump(arquivos, f, ensure_ascii=False, indent=2)

    relevantes = [a for a in arquivos if a.get("tipoDocumentoId") in TIPOS_DOCUMENTO]
    log.info("  [A] %d arquivo(s) no PNCP, %d relevante(s) (tipos %s)",
             len(arquivos), len(relevantes), TIPOS_DOCUMENTO)

    docs    = pasta / "docs"
    tmp_dir = pasta / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    documentos_baixados: list[dict] = []

    for arq in relevantes:
        titulo  = arq.get("titulo") or arq.get("nome") or f"doc_{arq.get('sequencialDocumento', 'x')}"
        tipo_id = arq.get("tipoDocumentoId")
        url_arq = (
            arq.get("url") or arq.get("urlArquivo") or
            arq.get("link") or arq.get("urlDownload")
        )
        if url_arq and url_arq.startswith("/"):
            url_arq = "https://pncp.gov.br" + url_arq
        if not url_arq:
            log.warning("  [A] Sem URL para: %s (tipo %s)", titulo, tipo_id)
            continue

        # Baixar para área temporária usando o titulo como nome base
        caminho_tmp = tmp_dir / sanitizar_nome(titulo)
        time.sleep(PAUSA_DOC)
        arquivo = baixar_arquivo(url_arq, caminho_tmp)

        if not arquivo:
            log.warning("  [A] Falha ao baixar: %s", titulo)
            continue

        # Extrair/processar para docs/
        finais = _extrair_para_docs(arquivo, docs, ctrl)
        for f in finais:
            documentos_baixados.append({
                "titulo":  titulo,
                "tipo_id": tipo_id,
                "arquivo": f.name,
                "origem":  "download",
            })

    # Limpar área temporária
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        "sucesso":          len(documentos_baixados) > 0,
        "cache":            False,
        "documentos":       documentos_baixados,
        "total_no_pncp":    len(arquivos),
        "total_relevantes": len(relevantes),
    }


# =============================================================================
# ESTRATÉGIA B — Coleta estruturada de itens via /itens
# =============================================================================

def estrategia_b(licitacao: dict, pasta: Path) -> dict:
    """
    Estratégia B: coleta todos os itens via /itens (paginado).

    Cache: se itens.json já existir, é reutilizado sem nova requisição.
    itens.json fica na raiz do processo (não em docs/).
    """
    # ── Cache ─────────────────────────────────────────────────────────────────
    itens_json = pasta / "itens.json"
    if itens_json.exists():
        try:
            with open(itens_json, encoding="utf-8") as f:
                todos_itens = json.load(f)
            log.info("  [B] itens.json reutilizado (%d itens)", len(todos_itens))
            return {"sucesso": True, "cache": True, "itens": todos_itens}
        except Exception:
            pass  # corrompido → coletar novamente

    # ── Coleta paginada ───────────────────────────────────────────────────────
    url_base = licitacao.get("url_itens_pncp")
    if not url_base:
        return {"sucesso": False, "itens": [], "erro": "sem url_itens_pncp"}

    todos_itens: list = []
    pagina = 1

    while True:
        url = f"{url_base}?pagina={pagina}&tamanhoPagina=50"
        log.info("  [B] GET %s", url)
        resp = get_com_retry(url)

        if resp is None or resp.status_code == 204:
            break

        try:
            dados = resp.json()
        except Exception:
            break

        if isinstance(dados, list):
            todos_itens.extend(dados)
            break

        itens      = dados.get("data") or []
        total_pags = dados.get("totalPaginas", 1) or 1
        pags_rest  = dados.get("paginasRestantes", 0) or 0

        todos_itens.extend(itens)
        log.info("  [B] pag. %d/%d — %d itens", pagina, total_pags, len(itens))

        if pags_rest == 0 or pagina >= total_pags or not itens:
            break

        pagina += 1
        time.sleep(PAUSA_DOC)

    if not todos_itens:
        return {"sucesso": False, "itens": [], "erro": "sem itens retornados"}

    pasta.mkdir(parents=True, exist_ok=True)
    with open(itens_json, "w", encoding="utf-8") as f:
        json.dump(todos_itens, f, ensure_ascii=False, indent=2)

    log.info("  [B] %d itens coletados → itens.json", len(todos_itens))
    return {"sucesso": True, "cache": False, "itens": todos_itens}


# =============================================================================
# ESTRATÉGIA C — Playwright (portal externo via linkSistemaOrigem)
# =============================================================================

def estrategia_c(licitacao: dict, pasta: Path, ctrl: str) -> dict:
    """
    Estratégia C: navega até o portal externo e tenta baixar o edital.
    Abordagem genérica — busca links de PDF/edital na página carregada.

    Requer USAR_PLAYWRIGHT = True e:
      pip install playwright && playwright install chromium
    """
    if not USAR_PLAYWRIGHT:
        return {"sucesso": False, "erro": "Playwright desabilitado (USAR_PLAYWRIGHT=False)"}

    link = licitacao.get("linkSistemaOrigem")
    if not link:
        return {"sucesso": False, "erro": "sem linkSistemaOrigem"}

    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except ImportError:
        log.warning("  [C] Playwright nao instalado. "
                    "Execute: pip install playwright && playwright install chromium")
        return {"sucesso": False, "erro": "playwright nao instalado"}

    log.info("  [C] Navegando: %s", link)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page    = browser.new_page()
            page.goto(link, timeout=30_000)
            page.wait_for_load_state("networkidle", timeout=15_000)
            candidatos: list[str] = page.eval_on_selector_all(
                "a[href]",
                """els => els
                    .map(e => ({ href: e.href, text: (e.textContent || '').toLowerCase() }))
                    .filter(o =>
                        o.href.toLowerCase().includes('.pdf') ||
                        o.href.toLowerCase().includes('edital') ||
                        o.text.includes('edital') ||
                        o.text.includes('termo de referencia') ||
                        o.text.includes('documento')
                    )
                    .map(o => o.href)
                """,
            )
            browser.close()

        if not candidatos:
            return {"sucesso": False, "erro": "sem candidatos na pagina"}

        docs    = pasta / "docs"
        tmp_dir = pasta / "_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        for pdf_url in candidatos[:3]:
            arquivo = baixar_arquivo(pdf_url, tmp_dir / "edital_externo")
            if arquivo:
                finais = _extrair_para_docs(arquivo, docs, ctrl)
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return {
                    "sucesso":  True,
                    "arquivos": [f.name for f in finais],
                    "fonte":    pdf_url,
                }
            time.sleep(PAUSA_DOC)

        shutil.rmtree(tmp_dir, ignore_errors=True)
        return {"sucesso": False, "erro": "falha ao baixar candidatos"}

    except Exception as exc:
        log.error("  [C] Erro Playwright: %s", exc)
        return {"sucesso": False, "erro": str(exc)}


# =============================================================================
# ORQUESTRADOR — cadeia A → B → C → D
# =============================================================================

def processar_licitacao(licitacao: dict) -> dict:
    """
    Executa a cadeia de fallback e retorna o registro do manifesto.

    BAIXAR_DOCUMENTOS=False  → apenas Estratégia B (itens.json). Rápido.
    BAIXAR_DOCUMENTOS=True   → cadeia completa A → B → C → D.
    """
    ctrl = licitacao.get("numeroControlePNCP", "desconhecido")
    obj  = (licitacao.get("objetoCompra") or "")[:120]
    log.info("-- %s", ctrl)
    log.info("   %s", obj)

    pasta = pasta_licitacao(ctrl, licitacao)

    manifesto: dict = {
        "numeroControlePNCP":       ctrl,
        "objetoCompra":             licitacao.get("objetoCompra", ""),
        "valorTotalEstimado":       licitacao.get("valorTotalEstimado"),
        "dataEncerramentoProposta": licitacao.get("dataEncerramentoProposta"),
        "linkSistemaOrigem":        licitacao.get("linkSistemaOrigem"),
        "srp":                      licitacao.get("srp"),
        "pasta":                    str(pasta),
        "estrategia_utilizada":     None,
        "status":                   None,
        "documentos_baixados":      [],
        "itens_total":              0,
        "itens_amostra":            [],
        "erros":                    {},
        "processado_em":            datetime.now().isoformat(timespec="seconds"),
    }

    # ── Fase 1 apenas (BAIXAR_DOCUMENTOS=False) ──────────────────────────────
    # Coleta somente itens.json. PDFs/editais são baixados numa etapa posterior,
    # após a análise de aderência, apenas para licitações relevantes.
    if not BAIXAR_DOCUMENTOS:
        res_b = estrategia_b(licitacao, pasta)
        if res_b["sucesso"]:
            itens = res_b["itens"]
            manifesto["estrategia_utilizada"] = "B"
            manifesto["status"]               = "parcial"
            manifesto["itens_total"]          = len(itens)
            manifesto["itens_amostra"]        = itens[:MAX_ITENS_MANIFESTO] if MAX_ITENS_MANIFESTO else itens
        else:
            manifesto["estrategia_utilizada"] = "B"
            manifesto["status"]               = "manual"
            manifesto["erros"]["B"]           = res_b.get("erro", "falhou")
            log.warning("  [B] Sem itens para: %s", ctrl)
        return manifesto

    # ── Fase 2 completa: A → B → C → D ───────────────────────────────────────
    # ── A ────────────────────────────────────────────────────────────────────
    res_a = estrategia_a(licitacao, pasta, ctrl)

    if res_a["sucesso"]:
        manifesto["estrategia_utilizada"] = "A"
        manifesto["status"]               = "ok"
        manifesto["documentos_baixados"]  = res_a["documentos"]

        res_b = estrategia_b(licitacao, pasta)
        if res_b["sucesso"]:
            itens = res_b["itens"]
            manifesto["itens_total"]          = len(itens)
            manifesto["itens_amostra"]        = itens[:MAX_ITENS_MANIFESTO] if MAX_ITENS_MANIFESTO else itens
            manifesto["estrategia_utilizada"] = "A+B"

        return manifesto

    manifesto["erros"]["A"] = res_a.get("erro", "falhou")

    # ── B ────────────────────────────────────────────────────────────────────
    res_b = estrategia_b(licitacao, pasta)

    if res_b["sucesso"]:
        itens = res_b["itens"]
        manifesto["estrategia_utilizada"] = "B"
        manifesto["status"]               = "parcial"
        manifesto["itens_total"]          = len(itens)
        manifesto["itens_amostra"]        = itens[:MAX_ITENS_MANIFESTO] if MAX_ITENS_MANIFESTO else itens
        return manifesto

    manifesto["erros"]["B"] = res_b.get("erro", "falhou")

    # ── C ────────────────────────────────────────────────────────────────────
    if USAR_PLAYWRIGHT:
        res_c = estrategia_c(licitacao, pasta, ctrl)
        if res_c["sucesso"]:
            manifesto["estrategia_utilizada"] = "C"
            manifesto["status"]               = "parcial"
            manifesto["documentos_baixados"]  = [
                {"arquivo": nome, "fonte": res_c.get("fonte"), "origem": "playwright"}
                for nome in res_c.get("arquivos", [])
            ]
            return manifesto
        manifesto["erros"]["C"] = res_c.get("erro", "falhou")

    # ── D — Alerta manual ────────────────────────────────────────────────────
    manifesto["estrategia_utilizada"] = "D"
    manifesto["status"]               = "manual"
    log.warning("  [D] Revisao manual necessaria: %s", ctrl)
    log.warning("      %s", licitacao.get("linkSistemaOrigem", "(sem link externo)"))
    return manifesto


# =============================================================================
# I/O
# =============================================================================

def carregar_licitacoes() -> list:
    try:
        with open(INPUT_FILE, encoding="utf-8") as f:
            dados = json.load(f)
        licitacoes = dados.get("licitacoes", [])
        log.info("Licitacoes carregadas: %d (de '%s')", len(licitacoes), INPUT_FILE)
        return licitacoes
    except FileNotFoundError:
        log.error("'%s' nao encontrado. Execute pncp_scanner.py primeiro.", INPUT_FILE)
        return []
    except Exception as exc:
        log.error("Erro ao ler '%s': %s", INPUT_FILE, exc)
        return []


def carregar_manifesto_anterior() -> tuple[set, list]:
    """
    Carrega o manifesto existente e valida cada entrada contra o disco.
    Se a pasta do processo foi apagada, o registro é descartado do manifesto
    e a licitação volta à fila de processamento — sem precisar apagar o JSON.
    """
    try:
        with open(MANIFESTO_FILE, encoding="utf-8") as f:
            dados = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return set(), []

    registros = dados.get("licitacoes", [])
    ja_feitos: set      = set()
    validos:   list     = []
    revertidos: int     = 0

    for r in registros:
        ctrl = r.get("numeroControlePNCP")
        if not ctrl:
            continue
        # Usa o caminho salvo no manifesto (inclui prefixo UF_CIDADE se existir).
        # Isso evita recalcular o nome da pasta, que pode ter mudado de formato.
        pasta_str = r.get("pasta", "")
        pasta_ok  = Path(pasta_str).exists() if pasta_str else pasta_licitacao(ctrl).exists()
        if pasta_ok:
            ja_feitos.add(ctrl)
            validos.append(r)
        else:
            revertidos += 1

    if revertidos:
        log.info("Pastas ausentes — %d registro(s) revertidos para reprocessamento", revertidos)

    return ja_feitos, validos


def salvar_manifesto(registros: list, counters: dict):
    saida = {
        "_meta": {
            "gerado_em":       datetime.now().isoformat(timespec="seconds"),
            "input_file":      INPUT_FILE,
            "output_dir":      OUTPUT_DIR,
            "tipos_documento": TIPOS_DOCUMENTO,
            "playwright":      USAR_PLAYWRIGHT,
            "incremental":     INCREMENTAL,
            "total_registros": len(registros),
            "status_ok":       counters.get("ok", 0),
            "status_parcial":  counters.get("parcial", 0),
            "status_manual":   counters.get("manual", 0),
            "status_erro":     counters.get("erro", 0),
        },
        "licitacoes": registros,
    }
    with open(MANIFESTO_FILE, "w", encoding="utf-8") as f:
        json.dump(saida, f, ensure_ascii=False, indent=2)
    log.info("Manifesto gravado: %s (%d registros)", MANIFESTO_FILE, len(registros))


# =============================================================================
# MAIN
# =============================================================================

def main():
    log.info("=" * 60)
    log.info("PNCP Documentos -- inicio")
    log.info("Entrada    : %s", INPUT_FILE)
    log.info("Saida      : %s/ + %s", OUTPUT_DIR, MANIFESTO_FILE)
    if BAIXAR_DOCUMENTOS:
        log.info("Modo       : FASE 2 — itens + PDFs/editais (A+B+C)")
        log.info("Tipos doc  : %s (2=Edital 4=TR 7=ETP 6=PB)", TIPOS_DOCUMENTO)
        log.info("Playwright : %s", "habilitado" if USAR_PLAYWRIGHT else "desabilitado")
    else:
        log.info("Modo       : FASE 1 — somente itens.json (BAIXAR_DOCUMENTOS=False)")
    log.info("Incremental: %s", INCREMENTAL)
    log.info("Pastas     : {UF}_{CIDADE}_{ctrl} (ex: MG_SETE_LAGOAS_97550393000149-1-000003_2026)")
    log.info("Cache      : ativo — apague a pasta do item para reforcar download")
    log.info("=" * 60)

    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    licitacoes = carregar_licitacoes()
    if not licitacoes:
        return

    ja_processados: set   = set()
    registros_anteriores: list = []

    if INCREMENTAL:
        ja_processados, registros_anteriores = carregar_manifesto_anterior()
        if ja_processados:
            log.info("Incremental: %d ja processados (mantidos)", len(ja_processados))

    pendentes = [l for l in licitacoes if l.get("numeroControlePNCP") not in ja_processados]
    log.info("Pendentes  : %d licitacoes a processar", len(pendentes))

    if not pendentes:
        log.info("Nada a fazer. Use INCREMENTAL=False para reprocessar metadados.")
        return

    novos_registros: list = []
    counters: dict = {"ok": 0, "parcial": 0, "manual": 0, "erro": 0}

    for i, licitacao in enumerate(pendentes, 1):
        log.info("[%d/%d]", i, len(pendentes))
        try:
            registro = processar_licitacao(licitacao)
        except Exception as exc:
            ctrl = licitacao.get("numeroControlePNCP", "?")
            log.error("Erro inesperado em %s: %s", ctrl, exc)
            registro = {
                "numeroControlePNCP":   ctrl,
                "status":               "erro",
                "estrategia_utilizada": None,
                "erros":                {"geral": str(exc)},
                "processado_em":        datetime.now().isoformat(timespec="seconds"),
            }

        status = registro.get("status", "erro")
        counters[status if status in counters else "erro"] += 1
        novos_registros.append(registro)

        if i % 10 == 0:
            salvar_manifesto(registros_anteriores + novos_registros, counters)

        if i < len(pendentes):
            time.sleep(PAUSA_LICITA)

    for reg in registros_anteriores:
        s = reg.get("status", "erro")
        counters[s if s in counters else "erro"] += 1

    todos = registros_anteriores + novos_registros
    salvar_manifesto(todos, counters)

    log.info("=" * 60)
    log.info("Concluido.")
    log.info("  ok      : %d (documentos baixados)", counters["ok"])
    log.info("  parcial : %d (so itens ou portal externo)", counters["parcial"])
    log.info("  manual  : %d (revisao humana necessaria)", counters["manual"])
    log.info("  erro    : %d (erro inesperado)", counters["erro"])
    log.info("  Total   : %d registros", len(todos))
    log.info("=" * 60)


if __name__ == "__main__":
    main()
