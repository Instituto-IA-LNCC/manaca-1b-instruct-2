# AES-ENEM — feedback de redação grounded na nota real (Claude)

`kamel-usp/aes_enem_dataset` (MIT) tem redações reais de estudantes
brasileiros no formato ENEM com notas oficiais por competência (C1-C5,
0-200 cada), mas SEM feedback textual. O manacá-jaster já cobre ENEM só
como múltipla escolha (`conv_enem`); isso ensina uma capacidade diferente:
avaliar e dar feedback qualitativo sobre um texto dissertativo.

## Por que precisa do Claude

Uma nota sozinha não ensina o modelo a EXPLICAR o porquê. O Claude gera a
justificativa qualitativa, mas GROUNDED nas notas reais — nunca inventa
uma nota diferente da que o dataset já traz (mesmo princípio de "não
invente além do que foi dado" do `synth/generate_instructions.py`).

## Schema incerto — auditar antes de escala

O dataset tem múltiplos configs (JBCS2025, PROPOR2024,
sourceAWithGraders, sourceB, ...) possivelmente com nomes de coluna
diferentes. O script descobre configs dinamicamente e detecta colunas de
nota de forma tolerante (nota única 0-1000, `c1`..`c5`, `comp1`..`comp5`,
ou uma lista `grades`); reporta no manifest o que foi de fato encontrado.

## Uso

```bash
export ANTHROPIC_API_KEY=sk-ant-...
make aes-enem-dry-run                                          # sem custo, mostra config/colunas detectadas
make aes-enem-data ARGS="--config sourceAWithGraders --max_examples 500"
```

Saída: `data/sft/aes_enem/aes_enem_feedback.jsonl` (Alpaca) + manifest com
colunas detectadas e custo estimado.

Emoji é banido (tolerância zero) — o tokenizer do `manaca-1b-base` não tem
token dedicado pra emoji (ver `sft/personas_br/README.md` para o detalhe
técnico); descarte registrado no manifest como `skipped_emoji`.

## Custo real — `--effort low` por padrão

`claude-sonnet-5`/`claude-opus-5` rodam com *thinking* adaptativo LIGADO
por padrão quando `output_config.effort` não é especificado, gastando
boa parte do `max_tokens` em raciocínio invisível (confirmado ao vivo no
`personas_br`, mesma arquitetura de chamada — ver detalhe no README de
lá). Por isso o script usa `--effort low` por padrão (ajustável via
`--effort medium|high|xhigh|max` se a qualidade não for suficiente para
uma justificativa de nota mais elaborada).

## Retomada (créditos limitados)

Cada exemplo é gravado no disco (e o arquivo, `flush`ado) assim que sai
da API — nada fica só em memória. O manifest grava `processed_idx`
(posição no dataset) a cada 20 exemplos e ao final. Se a rodada parar no
meio (créditos esgotados, Ctrl-C, container derrubado) — inclusive se o
próprio script detectar um erro 400/403 de saldo/crédito e parar sozinho
(status `stopped_fatal_error`, saída com código 2) — **rode o mesmo
comando de novo** com o mesmo `--config`/`--model`: ele pula os exemplos
já tentados e usa modo *append* no arquivo de saída, sem duplicar nem
pagar de novo pelo que já foi processado.

**Rodar de novo com parâmetros DIFERENTES sobre o mesmo `--out`** (lote já
`completed`, ou `--config`/`--model` mudados) é bloqueado de propósito — o
comando para com erro em vez de sobrescrever silenciosamente exemplos já
pagos. Use `--out <outro-diretório>` para um lote novo separado, ou
`--overwrite` se a intenção é mesmo descartar o que está no diretório atual.

## Bug real corrigido: `grades` tem 6 elementos, não 5

A detecção de colunas de nota (`detect_grade_columns`) assumia que o
campo `grades` (quando presente como lista) tinha exatamente 5 valores
(C1-C5). Confirmado ao vivo (`kamel-usp/aes_enem_dataset`, config
`JBCS2025`): a lista tem **6** elementos — `[c1, c2, c3, c4, c5, total]`
(ex.: `[160, 120, 120, 120, 80, 600]`, onde `600` é a soma dos 5
primeiros). Corrigido para aceitar `len(grades) in (5, 6)`; quando o 6º
elemento existe, ele vira uma linha extra "Nota total: N (de 0 a 1000)"
no texto passado ao modelo.

## Variante via LLM local (sem custo de API)

`generate_aes_enem_feedback_local.py` — mesmo pipeline (redação real +
notas reais -> feedback grounded, mesmo `SYSTEM_PROMPT`/emoji-zero), mas
fala com endpoint(s) vLLM locais em vez da API da Anthropic. Mesma trava
de `--overwrite` e mesmo padrão de resume via `manifest.json` (com
`processed_ids` em vez de `processed_idx` único, já que as chamadas rodam
em paralelo via `ThreadPoolExecutor`).

**Diferença deliberada**: `--max_examples` aqui limita quantas LINHAS do
dataset são *tentadas* (por índice), não quantas são de fato *gravadas* —
a versão via Claude para assim que o total gravado atinge o teto, o que
não dá pra replicar de forma simples com processamento concorrente.

Auditoria (`--audit-sample N`, via Claude) usa critérios próprios
(`fiel_as_notas`/`fluente`/`especifico`) — aqui o "grounding" é literal
(as 5 notas fornecidas), então a auditoria checa se o feedback gerado é
consistente com ELAS especificamente, não com um texto-fonte genérico.

```bash
make llm-serve-0 && make llm-serve-1
make aes-enem-local-dry-run
make aes-enem-local-run ARGS="--out data/sft/aes_enem_local --max_examples 2000 --audit-sample 40"
```
