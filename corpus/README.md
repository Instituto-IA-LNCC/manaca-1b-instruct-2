# Corpus-fonte para instrução sintética — Manacá-1B-Instruct-2

Este `corpus/` **não é um pipeline de pré-treino** (isso vive em
[`manaca/corpus/`](https://github.com/brunoleomenezes/manaca/tree/main/corpus),
que constrói o Tier 1 do **base**). Aqui, os scripts baixam passagens de
**fontes nativas em PT-BR** que viram matéria-prima para um passo seguinte de
**geração de instrução sintética** (documento → `{instruction, input, output}`
via LLM professor), reduzindo a dependência do corpus de SFT v1/v2 em tradução
automática (Alpaca-PT/OPUS-100 ≈ 64% do total — ver `docs/data/README.md`).

Mesmo padrão de engenharia do `manaca/corpus/scripts/`: scripts numerados,
schema Parquet fixo, manifesto JSON atômico (resume idempotente por shard),
`passes_filter`/`quality_score`, `verify()` de integridade, logs via loguru.
As diferenças estão documentadas no docstring de cada script.

## Fontes

| # | Script | Fonte HF | Papel | Licença |
|---|--------|----------|-------|---------|
| 01 | `01_acquire_legalpt.py` | `eduagarcia/LegalPT_dedup` (menos `tesemo_v2`) | Domínio jurídico — ataca o OAB no acaso do instruct v1/v2 | mista, verificar por config (ver script) |
| 02 | `02_acquire_classiccpt.py` | `ClassiCC-Corpus/ClassiCC-PT` | Web nativo PT-BR filtrado por `edu_score`/`toxic_score` | aberta, herda termos do CC |
| 03 | `03_acquire_carolina.py` | `carolina-c4ai/corpus-carolina` (rev. `v1.2`) | Diversidade de registro/gênero (escrita, roleplay) | CC BY-NC-SA 4.0 — **não comercial** |

Critério de escolha (ver o levantamento completo na conversa que originou este
pipeline): excluídas todas as fontes já usadas no Tier 1 do **base**
(GigaVerbo, FineWeb-2, HPLT 2.0) ou com licença ainda não confirmada
(Aurora-PT, Aroeira, GigaVerbo-v2) ou comercial (Jabuticaba) ou cujo texto
bruto herda restrição de uma fonte não-aberta (CrawlPT/brWaC).

### Por que `tesemo_v2` é excluído do LegalPT (importante)

O LegalPT_dedup reempacota o Ulysses Tesemõ inteiro como um de seus 6
sub-corpora. O `manaca` (base) já extrai o Tesemõ por inteiro no seu Script
06. `01_acquire_legalpt.py` itera as configs do LegalPT_dedup individualmente
e **nunca solicita** a config `tesemo_v2` — não depende de inferir valores da
coluna `source`, é uma exclusão estrutural e auditável.

## Diferença de escopo vs. `manaca/corpus/`

Cada script tem um teto de documentos por padrão (`--max-docs`, tipicamente
150k-300k), porque a saída aqui alimenta um passo de geração sintética via
LLM (custo de API), não o pré-treino de um modelo do zero. `--max-docs 0`
remove o teto se um uso futuro precisar do corpus inteiro.

## Pré-requisitos

```bash
cp .env.example .env            # DATA_DIR, HF_TOKEN etc.
make build-posttrain            # imagem GPU (agora tambem instala requirements/corpus.txt)
```

**Nota de versão do `datasets`:** `requirements/corpus.txt` **não** fixa
`datasets` — reaproveita o `datasets==3.1.0` de `requirements/posttrain.txt`.
O Carolina Corpus ainda depende de um loading script custom
(`trust_remote_code=True`), suporte removido em `datasets>=4.0`. Instale
sempre as duas listas juntas (a imagem Docker já faz isso).

## Reproduzir

```bash
make corpus-legalpt      # -> $DATA_DIR/raw/legalpt/
make corpus-classiccpt   # -> $DATA_DIR/raw/classiccpt/
make corpus-carolina     # -> $DATA_DIR/raw/carolina/
make corpus-all          # os 3 em sequencia

# Argumentos extra (ex.: sem teto de documentos):
make corpus-legalpt ARGS="--max-docs 0"
make corpus-classiccpt ARGS="--min-edu 2.5 --max-toxic 0.3"
make corpus-carolina ARGS="--revision v2.0.1"
```

Cada script é idempotente (retoma do último shard/config em caso de
interrupção) e grava metadados em
`$DATA_DIR/checkpoints/acquisition_manifest.json` (uma entrada por fonte),
mais um relatório de distribuição específico por fonte
(`config_distribution.json`, `classiccpt_quality_stats.json`,
`taxonomy_distribution.json`).

## Próximo passo

Estes três diretórios de shards Parquet ainda **não são** dados de SFT — são
passagens brutas. O passo seguinte (fora do escopo destes scripts) é o
pipeline de geração de instrução sintética: amostrar passagens de qualidade,
gerar `{instruction, input, output}` grounded no texto via um LLM professor,
decontaminar contra ENEM/BLUEX/OAB e os benchmarks internos
(`bench/`), e registrar manifest com `sha256` por fonte antes de entrar em
`sft/prepare_data.py`.
