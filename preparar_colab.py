# -*- coding: utf-8 -*-
"""
preparar_colab.py
Gera resgatecnica_colab.zip pronto para upload no Google Colab.

Inclui:
  - pncp_agente.py
  - pncp_licitacoes.json
  - exclusoes.yaml
  - prompts/  (todos os arquivos)
  - editais/*/itens.json  (SEM aderencia.json — análise limpa no Colab)

NÃO inclui:
  - aderencia.json (para não pular licitações no Colab)
  - pncp_telemetria.jsonl / pncp_aderencias.json (outputs)
  - debug_llm/, backup/, api_docs/ (desnecessários)
  - PDF do catálogo (upload separado se precisar do RAG)

Uso:
  python preparar_colab.py
"""

import zipfile
import pathlib
import os

BASE = pathlib.Path(__file__).parent
SAIDA = BASE / "resgatecnica_colab.zip"

# Arquivos raiz a incluir
ARQUIVOS_RAIZ = [
    "pncp_agente.py",
    "pncp_licitacoes.json",
    "exclusoes.yaml",
]

def main():
    total_bytes = 0
    total_arquivos = 0

    with zipfile.ZipFile(SAIDA, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:

        # 1. Arquivos raiz
        for nome in ARQUIVOS_RAIZ:
            p = BASE / nome
            if p.exists():
                zf.write(p, nome)
                total_bytes += p.stat().st_size
                total_arquivos += 1
                print(f"  + {nome}  ({p.stat().st_size / 1024:.0f} KB)")
            else:
                print(f"  ! AVISO: {nome} não encontrado, pulando")

        # 2. Pasta prompts/ (todos os arquivos)
        prompts_dir = BASE / "prompts"
        if prompts_dir.exists():
            for f in sorted(prompts_dir.iterdir()):
                if f.is_file():
                    arcname = f"prompts/{f.name}"
                    zf.write(f, arcname)
                    total_bytes += f.stat().st_size
                    total_arquivos += 1
                    print(f"  + {arcname}  ({f.stat().st_size / 1024:.0f} KB)")
        else:
            print("  ! AVISO: pasta prompts/ não encontrada")

        # 3. editais/*/itens.json (sem aderencia.json)
        editais_dir = BASE / "editais"
        n_itens = 0
        sz_itens = 0
        if editais_dir.exists():
            for pasta in sorted(editais_dir.iterdir()):
                if pasta.is_dir():
                    itens = pasta / "itens.json"
                    if itens.exists():
                        arcname = f"editais/{pasta.name}/itens.json"
                        zf.write(itens, arcname)
                        n_itens += 1
                        sz_itens += itens.stat().st_size
                        total_bytes += itens.stat().st_size
                        total_arquivos += 1
            print(f"  + editais/*/itens.json  ({n_itens} arquivos, {sz_itens / 1024:.0f} KB)")
        else:
            print("  ! AVISO: pasta editais/ não encontrada")

    zip_size = SAIDA.stat().st_size
    print()
    print(f"{'=' * 50}")
    print(f"ZIP gerado: {SAIDA.name}")
    print(f"Arquivos incluídos:  {total_arquivos}")
    print(f"Tamanho original:    {total_bytes / 1_048_576:.2f} MB")
    print(f"Tamanho comprimido:  {zip_size / 1_048_576:.2f} MB")
    print()
    print("Próximo passo:")
    print("  1. Abra o notebook resgatecnica_colab.ipynb no Google Colab")
    print("  2. Faça upload de resgatecnica_colab.zip quando solicitado")
    print("  3. Siga as células em ordem")


if __name__ == "__main__":
    print(f"Gerando {SAIDA.name}...")
    main()
