---
description: Encerra a sessão de trabalho criando um checkpoint detalhado e atualizando toda a memória persistente do projeto Resgatécnica. Use sempre que o usuário disser "encerrar", "checkpoint", "salvar sessão", "até amanhã" ou similar.
---

Você está encerrando a sessão de trabalho. Execute o checkpoint completo seguindo estes passos em ordem:

## 1. Ler o estado atual dos arquivos

Leia os arquivos que foram modificados ou são relevantes para o projeto:
- `D:\Edward\Resgatecnica\pncp_scanner.py` — verificar configurações atuais (MODALIDADES, UFS, FILTROS, etc.)
- `D:\Edward\Resgatecnica\pncp_documentos.py` — verificar configurações atuais (TIPOS_DOCUMENTO, MANTER_COMPACTADO, INCREMENTAL, etc.)
- Qualquer outro script `.py` criado ou modificado nesta sessão
- `D:\Edward\Resgatecnica\exclusoes.yaml` — se foi modificado

## 2. Ler a memória atual

Leia os arquivos de memória existentes para não duplicar nem contradizer:
- `C:\Users\caldas\.claude\projects\D--Edward-Resgatecnica\memory\MEMORY.md`
- `C:\Users\caldas\.claude\projects\D--Edward-Resgatecnica\memory\project_context.md`
- `C:\Users\caldas\.claude\projects\D--Edward-Resgatecnica\memory\user_profile.md`

## 3. Atualizar project_context.md

Este é o arquivo mais importante. Reescreva-o completamente com um checkpoint minucioso incluindo **tudo** que foi feito nesta sessão. Seja exaustivo — na próxima sessão não haverá memória do que foi feito hoje além do que está neste arquivo.

O checkpoint deve conter obrigatoriamente:

**O que foi implementado/alterado nesta sessão**
- Liste cada mudança com detalhes técnicos suficientes para entender o "porquê"
- Inclua trechos de código ou nomes de funções quando relevante
- Documente decisões de design e a razão por trás delas

**Estado atual de cada script**
- Configurações-chave e seus valores exatos neste momento
- O que funciona, o que está em teste, o que ainda não foi implementado
- Dependências instaladas (pip install, etc.)

**Comportamentos a verificar / bugs conhecidos**
- Testes que o usuário precisa executar
- Comportamentos suspeitos observados
- Pontos de atenção para a próxima sessão

**Próximos passos**
- Ordenados por prioridade
- Com detalhes suficientes para retomar sem precisar reconstituir o contexto

**Padrões de código estabelecidos**
- Convenções que devem ser seguidas nos próximos scripts

## 4. Atualizar user_profile.md (se necessário)

Se nesta sessão você aprendeu algo novo sobre as preferências ou estilo de trabalho do usuário (Piolhitos), atualize o perfil. Exemplos do que vale registrar:
- Preferências de nomenclatura ou organização de código
- Como ele gosta de receber explicações
- Decisões de design que ele aprova ou rejeita consistentemente
- Domínios de conhecimento demonstrados

Não atualize se nada de novo foi aprendido.

## 5. Atualizar MEMORY.md (se necessário)

Se novos arquivos de memória foram criados nesta sessão, adicione-os ao índice em `MEMORY.md`. O índice deve ter uma linha por arquivo no formato:
`- [NomeArquivo.md](NomeArquivo.md) — descrição de uma linha`

## 6. Apresentar resumo ao usuário

Após atualizar tudo, apresente ao usuário:

```
## Checkpoint salvo

**O que foi documentado:**
[lista dos principais itens capturados — 5 a 10 pontos]

**Próximos passos registrados:**
[lista dos próximos passos em ordem de prioridade]

**Arquivos de memória atualizados:**
[lista dos arquivos que foram modificados]

Boa noite! Até a próxima sessão.
```

---

**Regras importantes:**
- Seja exaustivo no project_context.md. Um detalhe que parece óbvio hoje pode ser decisivo amanhã.
- Sempre inclua os valores atuais das variáveis de configuração, não apenas os nomes.
- Documente o PORQUÊ das decisões, não apenas O QUÊ foi feito.
- Se houver dúvida sobre incluir algo, inclua.
