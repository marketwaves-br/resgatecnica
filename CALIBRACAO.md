# Manual de Calibração do Pipeline — Resgatécnica

## O que é calibração e quando fazer

Calibrar é ajustar parâmetros do pipeline para que ele **encontre mais licitações certas e menos erradas**.

Você calibra quando:
- Edward descarta licitações que o pipeline aprovou ("isso não é pra nós")
- Edward encontra oportunidades que o pipeline perdeu ("cadê essa licitação?")
- Após atualizar o portfólio de produtos
- Após adicionar novos padrões de busca (novos "produtos" a prospectar)

---

## Arquitetura de parâmetros

```
config.yaml          ← você edita aqui
     ↓
pncp_config.py       ← carregador compartilhado (lê yaml/json)
     ↓
┌────────────────────────────────────────────┐
│ pncp_similaridade.py   (pipeline completo) │
│ benchmark_licitacoes.py (validação)        │
│ pncp_embeddings.py      (retrieval/bônus)  │
└────────────────────────────────────────────┘
```

**Prioridade:** argumentos CLI > `config.yaml` > defaults internos do script.

---

## As 3 camadas de controle

### Camada 1 — Threshold (portão de saída)

**Onde:** `config.yaml` → `pipeline.threshold`
**O que faz:** score_bid mínimo para uma licitação aparecer no resultado final.

```yaml
pipeline:
  threshold: 0.10   # padrão conservador
```

| Valor | Efeito |
|-------|--------|
| 0.08  | Mais resultados, inclui casos duvidosos |
| 0.10  | Padrão atual — equilibrado |
| 0.15  | Mais restrito, só casos com boa evidência |
| 0.20  | Muito restrito, apenas casos fortes |

**Quando subir:** Edward está recebendo muitas licitações irrelevantes.  
**Quando descer:** Edward está perdendo oportunidades reais.

---

### Camada 2 — Contexto do objeto (multiplicador global)

**Onde:** `pncp_filtros.py` → listas `_ANCORAS_FORTES_RAW`, `_CONTEXTOS_INCOMPATIVEIS_RAW`, `_CONTEXTOS_IMPOSSIVEIS_RAW`

O `objetoCompra` da licitação define um multiplicador antes de checar os itens:

| Situação | Multiplicador | Exemplo |
|----------|---------------|---------|
| Âncora forte encontrada | 1.00 | "ambulância", "resgate", "incêndio" |
| Objeto neutro | 0.70 | "registro de preços para aquisição de materiais" |
| Contexto incompatível | 0.35 | "mobiliário", "ar condicionado", "merenda" |
| Contexto impossível | 0.00 | "merenda escolar", "material didático" |

**Como adicionar um novo contexto impossível** (licitação que nunca deve aparecer):
```python
# pncp_filtros.py — _CONTEXTOS_IMPOSSIVEIS_RAW
_CONTEXTOS_IMPOSSIVEIS_RAW = [
    "merenda escolar",
    "material didático",
    "seu novo termo aqui",   # ← adicione aqui
    ...
]
```

**Como adicionar uma nova âncora forte** (objeto que sempre deve passar):
```python
# pncp_filtros.py — _ANCORAS_FORTES_RAW
_ANCORAS_FORTES_RAW = [
    "ambulância",
    "resgate",
    "seu novo termo aqui",   # ← adicione aqui
    ...
]
```

> ⚠️ Após editar `pncp_filtros.py`, não precisa recompilar cache.

---

### Camada 3 — Bônus de retrieval por intenção

**Onde:** `config.yaml` → seção `calibracao`  
**O que faz:** ajusta quanto o retrieval semântico favorece ou penaliza cada categoria do portfólio conforme o tipo de intenção detectada na descrição do item.

```yaml
calibracao:
  bonus_servico_aph_categoria: 0.20        # quanto favorecer APH em serviços de transporte
  bonus_veicular_forte_categoria: 0.18     # quanto favorecer veicular em aquisições
  penalidade_veicular_bolsas_coletes: 0.12 # quanto penalizar bolsas/coletes em busca de veículo
```

**Regra:** ajuste de **+0.02 a +0.05 por iteração**. Mudanças grandes causam efeitos colaterais.

---

## O ciclo de calibração

```
1. Rodar benchmark
2. Gerar relatório
3. Identificar padrão dos erros
4. Ajustar UM parâmetro no config.yaml
5. Rodar benchmark de novo
6. Comparar
7. Repetir até estabilizar
```

### Passo a passo prático

**Rodar benchmark:**
```bash
python benchmark_licitacoes.py
```

**Gerar relatório:**
```bash
python resumo_resultados.py benchmark
```

**Ler o output — o que importa:**
```
MISMATCHES:
  ⚠  servico_aph->veicular_customizado: 3   ← serviço APH indo para veicular
  ⚠  aquisicao_veicular->aph: 1             ← veículo indo para APH

DETALHE DOS MISMATCHES:
  scenario : servico_aph → expected: aph | got: veicular_customizado
  desc     : CONTRATACAO DE EMPRESA... UTI MOVEL...
  rerank#1 : [0.22 raw=-1.24] Veículos Especiais > Soluções customizadas...
```

**Tabela de diagnóstico:**

| Mismatch observado | Causa provável | O que ajustar |
|--------------------|---------------|---------------|
| APH → veicular | Bônus APH insuficiente | Subir `bonus_servico_aph_categoria` |
| veicular → APH | Bônus APH excessivo | Descer `bonus_servico_aph_categoria` |
| Licitação irrelevante aprovada | Threshold baixo ou falta contexto impossível | Subir `threshold` ou adicionar termo em `_CONTEXTOS_IMPOSSIVEIS_RAW` |
| Licitação relevante rejeitada | Threshold alto ou objeto mal classificado | Descer `threshold` ou adicionar termo em `_ANCORAS_FORTES_RAW` |
| Score alto em item claramente errado | Falta no `_NEGATIVOS_ITEM` | Adicionar regex em `pncp_similaridade.py` → `_NEGATIVOS_ITEM` |

---

## Como testar novos produtos ("patterns")

O benchmark valida o pipeline contra licitações reais que contêm referência a produtos específicos.
Para cada produto que a Resgatécnica quer prospectar, você define um **pattern** (padrão regex).

### Patterns disponíveis

| Chave | O que detecta |
|-------|---------------|
| `ambulancia` | Ambulância, ambulância |
| `uti_movel` | UTI móvel, UTI móvel |
| `simples_remocao` | Simples remoção |
| `furgoneta` | Furgoneta, furgão |
| `minivan` | Minivan |
| `viatura` | Viatura |
| `transporte_sanitario` | Transporte sanitário |
| `remocao_terrestre` | Remoção terrestre |

### Como adicionar um novo produto para testar

**Exemplo: quero testar licitações de "colete balístico"**

**Passo 1** — Abra `benchmark_licitacoes.py` e adicione o pattern:
```python
PATTERNS: dict[str, str] = {
    "ambulancia":          r"ambul[aâ]ncia|ambulancia",
    "uti_movel":           r"uti m[oó]vel|uti movel",
    # ... patterns existentes ...

    # NOVO — colete balístico
    "colete_balistico":    r"colete bal[ií]stico|coletes bal[ií]sticos|proteção bal[ií]stica",
}
```

**Passo 2** — Adicione ao `config.yaml`:
```yaml
benchmark:
  patterns:
    - ambulancia
    - uti_movel
    - colete_balistico     # ← novo
```

**Passo 3** — Rode o benchmark:
```bash
python benchmark_licitacoes.py
python resumo_resultados.py benchmark
```

**Passo 4** — Leia os mismatches. Se o retrieval estiver indo para a categoria errada:
- Verifique se o portfólio tem entradas para "colete balístico"
- Se sim, ajuste os bônus de `intencao_veicular_forte` ou crie uma nova seção de intenção
- Se não, o portfólio precisa ser atualizado primeiro (conversar com Edward)

### Como adicionar um padrão com exclusões de contexto

Alguns produtos precisam excluir contextos específicos para evitar falsos positivos:
```python
# benchmark_licitacoes.py — CONTEXT_EXCLUDE_PATTERNS
CONTEXT_EXCLUDE_PATTERNS: dict[str, str] = {
    "ambulancia": r"decoração|ornamentação|evento|festa|camarim",
    "minivan":    r"hemodiálise|tratamento de hemodiálise",

    # NOVO — colete balístico: excluir artigos de fantasia/esporte
    "colete_balistico": r"fantasia|carnaval|airsoft|paintball",
}
```

---

## Executando via IDE (Spyder, Jupyter, VS Code)

Todos os scripts suportam configuração direta por variáveis no início do arquivo
**e** via `config.yaml`. Para usar em IDE sem passar argumentos CLI:

### Opção A — Editar `config.yaml` (recomendado)
Edite os valores desejados no `config.yaml` e execute o script normalmente.
O script carrega o arquivo automaticamente.

### Opção B — Passar argumentos simulados no script

No início de qualquer script, localize o bloco `main()` e substitua temporariamente:
```python
# MODO IDE — substitua a linha args = parser.parse_args() por:
args = parser.parse_args([
    "--backend", "semantic",
    "--threshold", "0.12",
    "--max", "50",
])
```

### Opção C — Usar config.yaml diferente por experimento

```bash
# CLI
python pncp_similaridade.py --config config_teste_coletes.yaml

# Ou em Python (Jupyter):
import subprocess
subprocess.run(["python", "pncp_similaridade.py", "--config", "config_teste.yaml"])
```

---

## Referência rápida de parâmetros

### `config.yaml` → `pipeline`

| Parâmetro | Default | Descrição |
|-----------|---------|-----------|
| `backend` | `semantic` | Motor de similaridade |
| `threshold` | `0.10` | Score mínimo para aprovar |
| `threshold_ambiguo` | `0.20` | Score mínimo para objetos ambíguos |
| `dias_minimos_preparo` | `7` | Dias mínimos até encerramento |
| `top_k_retrieval` | `10` | Candidatos recuperados por item |
| `top_k_debug_json` | `5` | Candidatos gravados no JSON |
| `retrieval_model` | `BAAI/bge-m3` | Modelo de embeddings |
| `reranker_model` | `BAAI/bge-reranker-v2-m3` | Modelo de reranking |

### `config.yaml` → `benchmark`

| Parâmetro | Default | Descrição |
|-----------|---------|-----------|
| `backend` | `semantic` | Motor de similaridade |
| `patterns` | lista de 6 | Produtos a testar |
| `max_cases` | `40` | Casos máximos por rodada |
| `top_k` | `5` | Candidatos por item |

### `config.yaml` → `calibracao`

| Parâmetro | Default | Descrição |
|-----------|---------|-----------|
| `bonus_servico_aph_categoria` | `0.20` | Bônus APH em serviços |
| `bonus_servico_aph_resgate_evacuacao` | `0.16` | Bônus resgate/evacuação em serviços |
| `bonus_servico_aph_bolsas` | `0.08` | Bônus bolsas em serviços APH |
| `bonus_servico_aph_oxigenio` | `0.06` | Bônus oxigênio/resgate em serviços |
| `penalidade_servico_aph_veicular` | `0.08` | Penalidade veicular em serviços APH |
| `bonus_veicular_forte_categoria` | `0.18` | Bônus veicular em aquisições diretas |
| `bonus_veicular_forte_resgate` | `0.07` | Bônus resgate veicular em aquisições |
| `penalidade_veicular_bolsas_coletes` | `0.12` | Penalidade bolsas/coletes em veicular |
| `bonus_viatura_rodoviaria_mobilidade` | `0.18` | Bônus mobilidade em viaturas rodoviárias |
| `penalidade_viatura_rodoviaria_moto` | `0.14` | Penalidade moto em viaturas rodoviárias |

---

## Fluxo sem consumo de créditos (Claude)

```bash
# 1. Você roda localmente (pode demorar)
python pncp_similaridade.py

# 2. Você gera o resumo (segundos)
python resumo_resultados.py

# 3. Cola o output no chat → Claude analisa e sugere ajuste no config.yaml

# 4. Você edita config.yaml, roda benchmark, cola resumo
python benchmark_licitacoes.py
python resumo_resultados.py benchmark
# → colar no chat se quiser segunda opinião
```

Créditos são usados apenas quando você cola o resultado para análise — não durante a execução dos scripts.
