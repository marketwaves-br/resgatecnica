import os
import json

# Configuração do diretório onde estão as pastas com os arquivos "itens.json"
# Você pode alterar este valor conforme necessário para mudar o local de execução
ITENS_DIR = "../editais/"
# Exemplos alternativos:
# ITENS_DIR = "Resgatecnica/editais/"
# ITENS_DIR = "d:/Edward/Resgatecnica/editais/"

ARQUIVO_SAIDA = "itens_geral.json"

# Lista de valores permitidos para o campo "tipoBeneficio"
# Apenas os itens que tiverem "tipoBeneficio" com um destes valores serão importados.
# Exemplo: se não interessam os tipos 1, 2 e 3, você pode definir apenas [4, 5, etc]
TIPO_BENEFICIO_PERMITIDO = [4, 5]

def agrupar_itens():
    dados_agrupados = {}
    itens_processados = 0
    
    print(f"Iniciando o agrupamento de itens do diretório: {ITENS_DIR}")
    
    # Verifica se o diretório configurado existe
    if not os.path.exists(ITENS_DIR):
        print(f"Erro: O diretório '{ITENS_DIR}' não foi encontrado.")
        return

    # Percorre todos os itens no diretório configurado
    for nome_pasta in os.listdir(ITENS_DIR):
        caminho_pasta = os.path.join(ITENS_DIR, nome_pasta)
        
        # Certifica-se de que é realmente um diretório (uma pasta de processo)
        if os.path.isdir(caminho_pasta):
            caminho_arquivo_json = os.path.join(caminho_pasta, "itens.json")
            
            # Verifica se o arquivo itens.json existe dentro dessa pasta
            if os.path.exists(caminho_arquivo_json):
                try:
                    # Lê o conteúdo do arquivo itens.json
                    with open(caminho_arquivo_json, 'r', encoding='utf-8') as f:
                        itens = json.load(f)
                        # Filtra os itens mantendo apenas aqueles com tipoBeneficio permitido
                        itens_filtrados = [
                            item for item in itens 
                            if item.get("tipoBeneficio") in TIPO_BENEFICIO_PERMITIDO
                        ]
                        
                        # Se ainda houver itens após o filtro, adicionamos ao dicionário, 
                        # usando o nome da pasta (processo) como chave
                        if itens_filtrados:
                            dados_agrupados[nome_pasta] = itens_filtrados
                            itens_processados += 1
                        
                except json.JSONDecodeError:
                    print(f"Aviso: O arquivo '{caminho_arquivo_json}' não é um JSON válido e foi ignorado.")
                except Exception as e:
                    print(f"Erro ao processar '{caminho_arquivo_json}': {e}")
                    
    # Salva todos os dados agrupados no arquivo de saída
    try:
        with open(ARQUIVO_SAIDA, 'w', encoding='utf-8') as f:
            json.dump(dados_agrupados, f, ensure_ascii=False, indent=4)
        
        print(f"\nSucesso! {itens_processados} pastas processadas.")
        print(f"Os dados foram agrupados e salvos no arquivo '{ARQUIVO_SAIDA}'.")
    except Exception as e:
        print(f"Erro ao salvar o arquivo de saída '{ARQUIVO_SAIDA}': {e}")

if __name__ == "__main__":
    agrupar_itens()
