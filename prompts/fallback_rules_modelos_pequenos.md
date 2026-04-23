# Regras de fallback para modelos pequenos

## Objetivo
Estas regras servem para execução em modelos locais menos potentes, com janela curta, menor capacidade de raciocínio ou menor estabilidade de saída JSON.

## 1. Modo econômico de contexto
Envie ao modelo, preferencialmente:

### Portfolio
- categoria
- subcategoria
- nome

### Portfolio (apoio, só se necessário)
- descricao_curta
- tags_semanticas
- aplicacoes resumidas

### Itens da licitação
- numeroItem
- descricao

### Itens (apoio, só se necessário)
- unidadeMedida
- informacaoComplementar
- ncmNbsCodigo
- materialOuServicoNome

## 2. Estratégia em duas etapas
### Etapa A — pré-filtro
Classifique rapidamente o contexto dominante da licitação:
- papelaria
- escolar
- odontologia
- gases medicinais
- esportivo
- administrativo
- EPI técnico
- APH
- incêndio
- resgate
- altura
- produtos perigosos

### Etapa B — detalhamento
Só detalhe profundamente itens com chance de:
- ADERENCIA_DIRETA
- ADERENCIA_PARCIAL_FORTE
- ADERENCIA_PARCIAL_FRACA

Itens claramente fora do escopo podem ter justificativa curta.

## 3. Regra de conservadorismo
Quando houver dúvida:
- prefira ADERENCIA_PARCIAL_FRACA a superestimar aderência
- prefira FALSO_POSITIVO_LEXICAL se a semelhança for só verbal
- prefira NAO_ADERENTE se o mercado for claramente outro

## 4. Lotes recomendados
### Modelos pequenos
- 25 a 50 itens por chamada

### Modelos médios
- 50 a 100 itens por chamada

### Modelos grandes
- 100+ itens, se a janela permitir

## 5. Saída enxuta
Para modelos pequenos, use o schema compacto.
Campos mínimos por item:
- numeroItem
- descricao
- classificacao
- grau_confianca
- produto_referencia
- justificativa

## 6. Heurística de tendência inicial
### Tendência forte para NAO_ADERENTE
- papelaria
- material escolar
- expediente
- odontologia
- gases medicinais
- esportivo
- escritório
- cozinha comum
- administrativo geral

### Tendência forte para ADERENTE / PARCIALMENTE_ADERENTE
- resgate
- incêndio
- altura
- espaço confinado
- APH
- proteção individual técnica
- descontaminação / produtos perigosos
- mergulho operacional
- segurança técnica crítica

## 7. Regra do campo enganoso
materialOuServicoNome pode estar errado. Se conflitar com a descrição, confie mais na descrição.

## 8. Regra do contexto dominante
Não deixe 1 item isolado alterar a leitura da licitação inteira.
O conjunto predominante deve pesar mais que exceções.
