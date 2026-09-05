# Quati — julgamento de relevância PT-BR nativo

`unicamp-dl/quati` (Nascimento et al., 2024, arXiv:2404.06976, CC BY 4.0) é
o primeiro dataset de Recuperação de Informação (IR) em PT-BR com
passagens E consultas 100% escritas por falantes nativos — a maioria dos
benchmarks de IR em PT é traduzida do inglês (MS MARCO).

## Por que é um script próprio (não uma tarefa do manacá-jaster)

O Quati não é um dataset único: são 3 configs relacionadas por ID
(`quati_10M_qrels`, `quati_all_topics`, `quati_10M_passages`) que precisam
ser juntadas. O registro `TASKS` do `build_jaster.py` assume um dataset
por tarefa, então esta junção fica num script próprio.

## Conversão

Julgamento binário de relevância (mesmo estilo dos labels de classificação
do jaster): dada a consulta e a passagem, "Sim" ou "Não" é relevante,
usando o `score` dos qrels contra `--relevance_threshold` (default 1.0 —
a escala exata do score não é documentada publicamente; o script reporta a
distribuição observada no log/manifest para calibrar).

NÃO geramos QA extrativo (responder a pergunta usando a passagem) — uma
consulta de busca não é necessariamente uma pergunta respondível pelo
texto; julgamento de relevância é o que os dados realmente suportam.

## Uso

```bash
make quati-data                                                    # default: 20k qrels
make quati-data ARGS="--max_examples 50000 --relevance_threshold 2"
```

Saída: `data/sft/quati/quati.jsonl` + `manifest.json` (inclui contagem
Sim/Não e passagens não localizadas). Usar via `--extra` no
`sft/prepare_data.py`.

## Custo de execução

A junção exige uma passada de streaming pelas 10M passagens (não há acesso
aleatório via streaming do HF) — custo de rede, não de RAM; os topics
cabem inteiros em memória.
