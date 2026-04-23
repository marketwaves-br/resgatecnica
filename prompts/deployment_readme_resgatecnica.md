# Pacote de implantação — Triagem de aderência Resgatécnica

Este pacote reúne os arquivos necessários para implantar a análise de aderência de licitações ao portfólio da Resgatécnica em API, pipelines internos ou modelos locais via Ollama.

## Arquivos principais
- `system_prompt_resgatecnica_full.txt` — system prompt completo
- `system_prompt_resgatecnica_local_compacto.txt` — system prompt compacto para modelos locais
- `user_template_resgatecnica_full.txt` — template de entrada completo
- `user_template_resgatecnica_local_compacto.txt` — template de entrada compacto
- `output_schema_resgatecnica_full.json` — schema de saída completo
- `output_schema_resgatecnica_local_compacto.json` — schema de saída compacto
- `fallback_rules_modelos_pequenos.md` — regras para modelos locais fracos
- `fmt_estrutura_minima_implantacao.json` — estrutura mínima de input
- `example_payload_api_full.json` — exemplo de payload lógico para API/pipeline
- `example_openai_responses_request.json` — exemplo de request em API estilo Responses
- `example_ollama_chat_request.json` — exemplo de request para Ollama
- `example_ollama_curl.sh` — exemplo em curl para Ollama
- `portfolio_mestre_resgatecnica_lite_v2.json` — base recomendada para inferência
- `portfolio_mestre_resgatecnica_full_v2.json` — base completa para manutenção/auditoria
- `manifest_implantacao_resgatecnica.json` — inventário do pacote

## Recomendação de uso
### Produção / modelos mais capazes
- system prompt: `system_prompt_resgatecnica_full.txt`
- user template: `user_template_resgatecnica_full.txt`
- schema: `output_schema_resgatecnica_full.json`
- portfolio: `portfolio_mestre_resgatecnica_lite_v2.json`

### Modelos locais / Ollama / pouca janela
- system prompt: `system_prompt_resgatecnica_local_compacto.txt`
- user template: `user_template_resgatecnica_local_compacto.txt`
- schema: `output_schema_resgatecnica_local_compacto.json`
- portfolio: `portfolio_mestre_resgatecnica_lite_v2.json`
- regras adicionais: `fallback_rules_modelos_pequenos.md`

## Fluxo recomendado
1. Carregar o system prompt.
2. Injetar o portfolio mestre lite.
3. Injetar os itens da licitação em lotes.
4. Exigir saída em JSON validada por schema.
5. Reprocessar apenas itens com aderência potencial, se quiser economizar custo.

## Observações
- O catálogo oficial é a fonte semântica principal.
- O portfolio mestre lite é a melhor base de inferência para LLM.
- O portfolio full é a base para manutenção e auditoria.
- Em caso de conflito entre palavras isoladas e contexto operacional, o contexto deve prevalecer.
