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

## Fontes brutas (precisam do passo de geração sintética depois)

| # | Script | Fonte HF | Papel | Licença |
|---|--------|----------|-------|---------|
| 01 | `01_acquire_legalpt.py` | `eduagarcia/LegalPT_dedup` (menos `tesemo_v2`) | Domínio jurídico — ataca o OAB no acaso do instruct v1/v2 | mista, verificar por config (ver script) |
| 02 | `02_acquire_classiccpt.py` | `ClassiCC-Corpus/ClassiCC-PT` | Web nativo PT-BR filtrado por `edu_score`/`toxic_score` | aberta, herda termos do CC |
| 03 | `03_acquire_carolina.py` | `carolina-c4ai/corpus-carolina` (rev. `v1.2`) | Diversidade de registro/gênero (escrita, roleplay) | CC BY-NC-SA 4.0 — **não comercial** |
| 04 | `04_acquire_madlad400.py` | `allenai/MADLAD-400` (config `pt`) | Web geral, domínios/períodos distintos do GigaVerbo | Apache 2.0 |
| 05 | `05_acquire_fineweb2.py` | `HuggingFaceFW/fineweb-2` (config `por_Latn`) | Web de alta qualidade (pipeline FineWeb) | ODC-By 1.0 |
| 06 | `06_acquire_hplt2.py` | `HPLT/HPLT2.0_cleaned` (config `por_Latn`) | Web geral, cobertura temporal distinta | CC0 (domínio público) |
| 07 | `07_acquire_portuguese_pd.py` | `PleIAs/Portuguese-PD` | Registro histórico/literário (pré-1884) | Domínio público |

### ClassiCC-PT — campo `id` não é confiável nos shards já adquiridos

Confirmado ao vivo (checagem em todo o corpus já baixado): o campo `id` do
`ClassiCC-Corpus/ClassiCC-PT` **não identifica um documento** — é um id de
segmento/WARC do Common Crawl compartilhado por centenas de documentos
diferentes (99,86% de "duplicação": 600.000 linhas, só 13.503 valores
distintos de `id`; um único valor cobria 609 documentos diferentes numa
amostra de 3.000). O campo que de fato varia por documento é `url`, que os
shards já gravados não preservam. `02_acquire_classiccpt.py` já foi
corrigido para preferir `url` em aquisições novas; `synth/generate_instructions.py:sample_passages()`
não depende mais de `id` para nada (usa a posição física da linha no shard
como chave), então os shards antigos continuam utilizáveis sem reaquisição.

### Portuguese-PD — ABANDONADO para geração sintética (qualidade insuficiente)

Testado ao vivo via `synth-local-run --source portuguese_pd`: primeiro
erro foi técnico (documentos são LIVROS/PERIÓDICOS INTEIROS por linha —
mediana de 287.780 caracteres, máximo de 29 milhões — estourava qualquer
`--max-model-len`; corrigido em `sample_passages()` com uma janela
aleatória de 12.000 caracteres, ver `_window_passage()`). Mas depois de
corrigido, uma amostra de 3 janelas revelou um problema de conteúdo mais
sério e não trivial de filtrar: **latim** (texto médico do séc. XVII),
**grego antigo** com OCR seriamente corrompido, e **o próprio aviso legal
do Google Books em inglês** ("Keep it legal... you are responsible for
ensuring...") colado como se fosse conteúdo do livro. O corpus mistura
bastante material não-português e ruído de OCR de scans antigos, além do
boilerplate do Google Books que vazou pro campo de texto.

**Decisão**: abandonado para geração sintética por agora (mesmo critério
do `aes_enem` — ver `sft/aes_enem/README.md`). Se retomar, precisa de
filtro de idioma (descartar janelas que não são majoritariamente PT-BR)
antes de mandar pro modelo — amostragem aleatória sozinha cai em lixo com
frequência alta demais pra valer a pena sem esse filtro.

### Fontes 04-06 — por que estavam descartadas e agora não estão

MADLAD-400, FineWeb-2 e HPLT 2.0 faziam parte do Tier 1 do **plano** de
pré-treino do `manaca`. Uma auditoria contra o repositório real do base
(`manaca-1b-base`) confirmou que o corpus final de pré-treino usou **só**
GigaVerbo (subamostra) + Ulysses Tesemõ + Wikipedia-pt (~20,1B tokens) — os
scripts dessas 3 fontes existem no `manaca-1b-base` mas nunca foram
executados/consumidos. Ou seja: "planejadas" ≠ "usadas" — estão livres de
sobreposição real com o base. Detalhe completo no docstring de cada script.

## Fontes de instrução já prontas (sem geração sintética)

| Diretório | Fonte HF | Tamanho | Licença |
|---|---|---|---|
| [`../sft/gigaverbo_v2_sft/`](../sft/gigaverbo_v2_sft/) | `Polygl0t/gigaverbo-v2-sft` | ~4M exemplos | Apache 2.0 |
| [`../sft/tucano_sft/`](../sft/tucano_sft/) | `TucanoBR/Tucano-SFT` | não confirmado (~600k?) | Apache 2.0 / MIT (subfontes) |
| [`../sft/quati/`](../sft/quati/) | `unicamp-dl/quati` (tarefa de relevância, 3 configs juntadas) | configurável | CC BY 4.0 |
| `../sft/jaster/build_jaster.py` (`extraglue_*`) | `PORTULAN/extraglue` | 4 tarefas NLI/STS/paráfrase | MIT (traduzido) |
| [`../sft/medpt/`](../sft/medpt/) | `AKCIT/MedPT` — QA médico REAL/nativo PT-BR | 384.095 pares | CC BY 4.0 |
| [`../sft/aya_pt/`](../sft/aya_pt/) | Aya Dataset (subset PT) — instrução humana nativa | ~1-10K | Apache 2.0 |
| `../sft/jaster/build_jaster.py` (`healthqa_br`) | `Larxel/healthqa-br` — múltipla escolha Revalida/Enare | 5.632 questões | CC BY 4.0 |
| [`../sft/juristcu/`](../sft/juristcu/) | `LeandroRibeiro/JurisTCU` (relevância jurídica, 3 CSVs juntados) | configurável | CC BY 4.0 |

## Fontes geradas via API (Claude) — o dado bruto não tem resposta pronta útil

| Diretório | Fonte HF | Por que precisa do Claude | Licença |
|---|---|---|---|
| [`../sft/personas_br/`](../sft/personas_br/) | `nvidia/Nemotron-Personas-Brazil` (6M personas) | Personas não são QA; Claude gera pedido+resposta condicionados à persona | CC BY 4.0 |
| [`../sft/aes_enem/`](../sft/aes_enem/) | `kamel-usp/aes_enem_dataset` (redações + notas reais) | Só há nota numérica, não feedback textual; Claude gera a justificativa GROUNDED na nota real | MIT |

## Fontes de preferência (2ª tentativa de DPO)

| Diretório | Fonte HF | Tamanho | Licença |
|---|---|---|---|
| [`../dpo/gigaverbo_v2_preferences/`](../dpo/gigaverbo_v2_preferences/) | `Polygl0t/gigaverbo-v2-preferences` | ~28,4K pares | Apache 2.0 |
| [`../dpo/reward_aira/`](../dpo/reward_aira/) | `nicholasKluge/reward-aira-dataset` | pequeno | Apache 2.0 |

O DPO on-policy do v1/v2 foi resultado negativo (dataset pequeno, gerado
pelo próprio modelo — ver `dpo/README-onpolicy.md`). Estas duas fontes são
maiores, geradas por modelos terceiros, e habilitam uma segunda tentativa —
**convertê-las não significa adotar o DPO**: a decisão é pela avaliação
(segurança no MT-Bench-PT + over-refusal estável), mesmo critério já usado.

## Fontes avaliadas e excluídas (com motivo)

| Fonte | Motivo |
|---|---|
| `UFRGS/brwac`, CrawlPT | Sem licença aberta declarada (nem no HF nem na fonte oficial UFRGS); acesso por formulário |
| `soberania/jabuticaba` | Comercial/gated |
| Aurora-PT, Aroeira, GigaVerbo-v2 (bruto) | Licença ainda não finalizada/declarada claramente |
| `thegoodfellas/blogset-br` | Sinal de licença conflitante (Apache-2.0 no HF vs. termo original PUC-RS de uso acadêmico) — precisa verificação manual antes de automatizar |
| `PleIAs/common_corpus` | Licença mista, não-uniforme por sub-fonte (fatia PT variável) — precisa auditoria por sub-fonte, não uma automação genérica |
| `Helsinki-NLP/OpenSubtitles2024` (pt) | Licença ambígua, herdada do OpenSubtitles.org |
| `factckbr/factckbr` (FACTCK.BR) | Licença não claramente declarada; corpus pequeno (1.309 claims) |
| SciELO parallel (`community-datasets/scielo`) | Licença não-uniforme (varia por artigo/periódico) |
| `dominguesm/Canarim-Instruct-PTBR` | CC BY-NC + redundante com Alpaca-PT já coberto (baixo valor marginal) |
| `hugoabonizio/clinical-protocols-br` | Sintético a partir de PCDTs do Ministério da Saúde, tipicamente CC BY-NC-SA — herda a restrição NC |
| SaudeBR-QA (artigo acadêmico) | Sem repositório de download público nem licença declarada encontrados |
| `Larxel/healthqa-br`, `AKCIT/MedPT` | Aceitos — CC BY 4.0 confirmado. Ver tabelas acima (`healthqa_br`, `medpt`) |
| `LeandroRibeiro/JurisTCU` | Aceito — CC-BY-4.0 confirmado na própria ficha (não apenas por busca). Ver `sft/juristcu/` |

## Diferença de escopo vs. `manaca/corpus/`

Cada script tem um teto de documentos/exemplos por padrão, porque a saída
aqui alimenta um passo de geração sintética via LLM (custo de API) ou já é
consumida direto no mix de SFT — não o pré-treino de um modelo do zero.
`--max-docs 0` / `--max_examples 0` remove o teto quando fizer sentido.

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
# Fontes brutas (corpus/scripts/, precisam do passo de geracao sintetica depois)
make corpus-all-raw     # 01-07 em sequencia
make corpus-legalpt ARGS="--max-docs 0"
make corpus-classiccpt ARGS="--min-edu 2.5 --max-toxic 0.3"
make corpus-carolina ARGS="--revision v2.0.1"
make corpus-madlad400 && make corpus-fineweb2 && make corpus-hplt2 && make corpus-portuguese-pd

# Fontes de instrucao ja prontas
make gigaverbo-v2-sft-data
make tucano-sft-data
make quati-data
./scripts/docker/run_posttrain.sh python sft/jaster/build_jaster.py --tasks extraglue_mnli,extraglue_qnli,extraglue_stsb,extraglue_mrpc --out data/sft/extraglue
make medpt-data
make aya-pt-data
make jaster-healthqa-br
make juristcu-data

# Fontes geradas via API (Claude) -- teste com --dry-run antes de gastar
make personas-br-dry-run && make personas-br-data ARGS="--num_personas 2000"
make aes-enem-dry-run && make aes-enem-data ARGS="--max_examples 500"

# Fontes de preferencia (2a tentativa de DPO)
make dpo-preferences-data
```

Cada script é idempotente (retoma do último shard/config em caso de
interrupção) e grava metadados em manifest JSON (por shard/fonte no caso
dos scripts de `corpus/`, ou `manifest.json` por diretório no caso dos
builders de `sft/`/`dpo/`).

## Próximo passo — geração de instrução sintética

Os diretórios de shards Parquet das fontes 01-07 ainda **não são** dados de
SFT — são passagens brutas. O pipeline que as transforma em
`{instruction, input, output}` (grounded no texto, via Claude/Anthropic
Batches API, com decontaminação e manifest de custo) já existe:
[`../synth/README.md`](../synth/README.md) ·
[`../synth/generate_instructions.py`](../synth/generate_instructions.py).

```bash
export ANTHROPIC_API_KEY=sk-ant-...
make synth-run ARGS="--source legalpt --max-examples 2000"
```

As fontes de instrução já prontas (`gigaverbo_v2_sft`, `tucano_sft`,
`quati`, `extraglue`, `medpt`, `aya_pt`, `healthqa_br`, `juristcu`) não
passam por este pipeline — já entram direto via `--extra` no
`sft/prepare_data.py`. As duas fontes geradas via Claude
(`personas_br`, `aes_enem`) também já saem prontas no schema Alpaca —
usam a API diretamente (chamadas síncronas, não Batches, pelo volume
menor), não o pipeline de passagens brutas.
