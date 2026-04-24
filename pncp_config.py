#!/usr/bin/env python3
"""
pncp_config.py — Carregador de configuração compartilhado.

Suporta config.yaml, config.json ou config.py (dict CONFIG) no diretório atual.
Prioridade de parâmetros: CLI > config.yaml/json > defaults do script.

Uso em qualquer script:
    from pncp_config import carregar_config
    cfg = carregar_config()                      # lê config.yaml se existir
    cfg = carregar_config("meu_config.yaml")     # arquivo específico

Retorna dict vazio {} se nenhum arquivo de configuração for encontrado —
o script usa seus próprios defaults normalmente.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Arquivos de config procurados automaticamente (em ordem de prioridade)
_CANDIDATOS = ["config.yaml", "config.yml", "config.json"]


def carregar_config(caminho: str | Path | None = None) -> dict:
    """
    Carrega arquivo de configuração e retorna dict.

    Se `caminho` não for fornecido, procura automaticamente por
    config.yaml → config.yml → config.json no diretório atual.

    Retorna {} se nenhum arquivo for encontrado (não é erro).
    """
    if caminho is not None:
        return _ler_arquivo(Path(caminho))

    for nome in _CANDIDATOS:
        p = Path(nome)
        if p.exists():
            return _ler_arquivo(p)

    return {}


def _ler_arquivo(path: Path) -> dict:
    if not path.exists():
        print(f"[config] Arquivo não encontrado: {path}", file=sys.stderr)
        return {}

    sufixo = path.suffix.lower()

    if sufixo in (".yaml", ".yml"):
        try:
            import yaml
            with open(path, encoding="utf-8") as f:
                dados = yaml.safe_load(f) or {}
            print(f"[config] Carregado: {path}")
            return dados
        except ImportError:
            print(
                f"[config] PyYAML não instalado. Execute: pip install pyyaml\n"
                f"         Usando apenas defaults do script.",
                file=sys.stderr,
            )
            return {}
        except Exception as e:
            print(f"[config] Erro ao ler {path}: {e}", file=sys.stderr)
            return {}

    if sufixo == ".json":
        try:
            dados = json.loads(path.read_text(encoding="utf-8"))
            print(f"[config] Carregado: {path}")
            return dados
        except Exception as e:
            print(f"[config] Erro ao ler {path}: {e}", file=sys.stderr)
            return {}

    print(f"[config] Formato não suportado: {sufixo} (use .yaml ou .json)", file=sys.stderr)
    return {}


def obter(cfg: dict, *chaves, default=None):
    """
    Acessa valor aninhado com segurança.

    Exemplo:
        threshold = obter(cfg, "pipeline", "threshold", default=0.10)
        patterns  = obter(cfg, "benchmark", "patterns", default=[])
    """
    atual = cfg
    for chave in chaves:
        if not isinstance(atual, dict):
            return default
        atual = atual.get(chave, None)
        if atual is None:
            return default
    return atual
