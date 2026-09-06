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
