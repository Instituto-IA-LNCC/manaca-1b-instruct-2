# GigaVerbo-v2 SFT (Polygl0t) — fonte pronta para o mix de instrução

`Polygl0t/gigaverbo-v2-sft` (Tucano 2, arXiv:2603.03543) é um dataset de
instrução em PT-BR **já pronto**: ~4,09M exemplos / ~2,1B tokens / 4,7 GB,
**Apache-2.0**, 100% gerado por LLM (não é tradução automática), filtrado por
juiz LLM (nota < 3 descartada) e decontaminado pelos autores contra os
benchmarks deles. Organizado em configs por categoria de tarefa
(`function_call`, `summarization`, `code`, `structured`, `translation`,
`math`, `math_cot`, `rewriting` e mais).

## Por que vale a pena

O corpus de SFT v1/v2 deste projeto é ~64% texto traduzido automaticamente
(Alpaca-PT + OPUS-100 — ver `docs/data/README.md`). O GigaVerbo-v2 SFT é
nativo e sintético numa escala que nenhum pipeline de geração própria deste
projeto alcançaria com orçamento de pesquisa, e cobre categorias
(code/math/function_call/structured) que o manacá-jaster mal toca.

## O que o conversor faz — e não faz

- **Descobre as configs dinamicamente** (`datasets.get_dataset_config_names`)
  — não fixamos a lista de categorias.
- **Dobra conversa multi-turno em Alpaca de turno único**: o `sft/train.py`
  deste repo não suporta multi-turno nativamente. Turnos anteriores ao
  último viram uma transcrição dentro de `input`; só o último par
  pergunta/resposta é usado como `instruction`/`output`. Perde-se o sinal
  de treino dos turnos intermediários — dar suporte a chat template
  multi-turno de verdade no `train.py` é o próximo passo natural (fora do
  escopo deste script).
- **Amostra por config** (`--max_per_task`, default 20.000): sem teto, esta
  única fonte (potencialmente centenas de milhares de linhas por config)
  dominaria o mix inteiro — o v2 completo hoje soma 141k exemplos.
- **Decontamina best-effort** contra os arquivos estáticos deste repo
  (`bench/`, `dpo/dpo_seeds/`, `sft/safety_seeds/`), normalizando acentos e
  caixa. **Não cobre** ENEM/BLUEX/OAB/ASSIN2/HateBR — esses são carregados
  dinamicamente do HF pelo `lm-eval`, não são arquivos deste repo. Rodar uma
  decontaminação contra eles é um passo futuro, não incluído aqui.

## Uso

```bash
make gigaverbo-v2-sft-data                    # default: todas as configs, 20k/config
make gigaverbo-v2-sft-data ARGS="--tasks code,math,math_cot --max_per_task 10000"
make gigaverbo-v2-sft-data ARGS="--max_per_task 0"   # sem teto (cuidado: pode ser ~4M exemplos)
```

Saída: `data/sft/gigaverbo_v2_sft/<config>.jsonl` + combinado
`gigaverbo_v2_sft.jsonl` + `manifest.json` (contagens, descartes por
contaminação/tamanho, `sha256` por config). Usar no mix via `--extra`, como
o manacá-jaster:

```bash
python sft/prepare_data.py --sources alpaca,aya,oasst --out data/sft_v3 \
    --shuffle --dedup \
    --extra data/sft/gigaverbo_v2_sft/gigaverbo_v2_sft.jsonl \
            data/sft_v2/jaster/manaca_jaster.jsonl
```

## Licença

Apache-2.0 — permissiva, compatível com uso comercial.
