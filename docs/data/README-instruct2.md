# Corpus de SFT do Manacá-1B-Instruct-2 (em construção)<br>SFT corpus of Manacá-1B-Instruct-2 (in progress)

Ficha do dado gerado pra fase "instruct-2" — a mesma disciplina do
[`docs/data/README.md`](README.md) (v1/v2): só o que foi de fato gerado e
confirmado, com contagem real (`wc -l`/log de execução), não estimativa.
`sft/prepare_data.py` já rodou e combinou tudo num `manaca_sft.jsonl` de
instruct-2 único (ver "Mix final combinado" abaixo).

Duas frentes de geração, cada uma com seu próprio README de detalhe:
- **Via Claude** (API paga): `synth/README.md`, `sft/personas_br/README.md`,
  `sft/aes_enem/README.md`.
- **Via LLM local** (Qwen2.5-32B-Instruct-AWQ, vLLM, sem custo de API):
  `scripts/docker/README-vllm.md`, mesmos READMEs acima (seção "Variante
  via LLM local").

## Contagens reais (confirmadas via `wc -l`, 2026-09-07)

| Fonte | Método | Exemplos | Licença |
|---|---|--:|---|
| ExtraGLUE — mnli | conversão direta (`sft/jaster/build_jaster.py`) | 9.815 | PORTULAN/extraglue |
| ExtraGLUE — qnli | idem | 104.743 | PORTULAN/extraglue |
| ExtraGLUE — stsb | idem (split `train`, ver bug abaixo) | 5.749 | PORTULAN/extraglue |
| ExtraGLUE — mrpc | idem | 1.725 | PORTULAN/extraglue |
| Quati (relevância) | conversão direta | 4.889 | CC BY 4.0 |
| personas_br (Claude) | `generate_personas_sft.py` (2 lotes) | 430 | CC BY 4.0 (Nemotron-Personas-Brazil) |
| personas_br (LLM local) | `generate_personas_sft_local.py` | 4.783 | idem |
| gigaverbo_v2_sft | conversão direta (12 configs, 20k/config) | 240.000 | Apache-2.0 |
| LegalPT (manual) | colado no chat, sem custo | 100 | domínio público (TCU) |
| LegalPT (LLM local) | `synth/generate_instructions_local.py` | 4.943 | mista, ver config |
| ClassiCC-PT (LLM local) | idem | 4.658 | aberta (herda CC) |
| MADLAD-400 (LLM local) | idem | 4.503 | Apache-2.0 |
| FineWeb-2 (LLM local) | idem | 4.648 | ODC-By 1.0 |
| HPLT2 (LLM local) | idem | 4.704 | CC0 |
| **TOTAL** | | **395.690** | |

Para comparação: v1 = 85.892, v2 = 141.404 (`docs/data/README.md`). O
instruct-2 já soma **2,8x o v2**, majoritariamente via LLM local (sem
custo de API além de auditorias pontuais via Claude Haiku).

### Bug real corrigido: `extraglue_stsb` usava split `test` com gold label mascarado

Auditoria de amostra (`cat` + parse de uma linha aleatória por fonte,
antes do combine final) achou `output: "-1.0"` numa linha de
`extraglue_stsb`, fora da escala 1-5 pedida no prompt. Checagem da
distribuição completa confirmou **100% das 1.379 linhas eram `-1.0`**:
`sft/jaster/build_jaster.py` usava `split="test"` do PORTULAN/extraglue
(`stsb_pt-BR`), que segue a convenção GLUE de esconder o gold label do
conjunto de teste — treinar nisso ensinaria o modelo a sempre responder
"-1.0" independente da similaridade real. Trocado para `split="train"`
(5.749 linhas, rótulos 0.0-5.0 com distribuição normal, confirmado ao
vivo). `extraglue_mrpc` usa o mesmo `split="test"`, mas sua distribuição
(66.5% Sim / 33.5% Não) bate com o desbalanceamento conhecido do MRPC
original — não é o mesmo bug, mantido como está.

## Fontes avaliadas e ABANDONADAS (documentado para não redescobrir)

| Fonte | Motivo | Detalhe |
|---|---|---|
| Carolina Corpus | Licença CC BY-NC-SA 4.0 (não comercial) — incompatível com o restante do projeto | excluído antes de qualquer geração |
| Portuguese-PD | Conteúdo de baixa qualidade pra este uso: documentos são livros/periódicos inteiros (mediana 287k caracteres, máx. 29M), e amostras revelaram latim, grego antigo com OCR corrompido, e boilerplate do Google Books colado como texto | `corpus/README.md` |
| aes_enem (via LLM local) | LLM local (Qwen2.5-32B) inventa erros gramaticais específicos que não existem na redação real, ao justificar notas medianas — tarefa exige fidelidade palavra-por-palavra, mais difícil de manter que síntese/resumo geral | `sft/aes_enem/README.md` |
| Math-PT (`tiagoteixeira03/MATH-PT`) | É benchmark de avaliação publicado (PROPOR 2026), não corpus de treino — treinar nele contaminaria uma avaliação futura do próprio Manacá; também sem licença declarada no HF | nunca usado, só avaliado |
| MGSM / mgsm_mt | Mesma razão (é benchmark, "Multilingual **Grade School Math Benchmark**"), pequeno (250 problemas) e traduzido do inglês | nunca usado |

`aes_enem` (via Claude) nunca foi testado em dado real — é o caminho mais
promissor se essa fonte for retomada no futuro (ver `sft/aes_enem/README.md`).

## Infraestrutura nova desta sessão

- `scripts/docker/run_vllm_serve.sh` + `scripts/docker/README-vllm.md`: sobe
  um LLM local (Qwen2.5-32B-Instruct-AWQ) via vLLM, 1 instância por GPU
  (sem NVLink no Netuno — 2x RTX A5000).
- `synth/generate_instructions_local.py`, `sft/personas_br/generate_personas_sft_local.py`,
  `sft/aes_enem/generate_aes_enem_feedback_local.py`: variantes locais dos
  geradores via Claude, mesma trava de `--overwrite`, mesmo padrão de
  resume via `manifest.json`, auditoria opcional via Claude sobre amostra.
- `synth/generate_instructions.py:sample_passages()`: corrigido para não
  depender do campo `id` do corpus-fonte (confirmado não-confiável no
  ClassiCC-PT) e para janelar textos gigantes (`_window_passage()`,
  confirmado necessário no Portuguese-PD).

## Mix final combinado (`sft/prepare_data.py`, 2026-09-07)

```
python3 sft/prepare_data.py --sources alpaca,aya,oasst,translation,summarization \
  --out data/sft_instruct2 --shuffle --dedup --max_chars 5000 --extra <15 arquivos acima>
```

Mesma base v2 (Alpaca-PT/Aya/OASST/OPUS-100/XLSum, baixados ao vivo do HF)
mais as 15 fontes da tabela acima via `--extra`. `--dedup`/`--max_chars 5000`
descartam duplicatas e prompts longos demais — por isso alguns números
abaixo são levemente menores que a tabela de "contagens reais" (que é
pré-filtro).

| Fonte | Exemplos no mix |
|---|--:|
| alpaca (Alpaca-PT) | 51.747 |
| aya (Aya Dataset, pt) | 8.809 |
| oasst (OASST1, pt) | 315 |
| translation (OPUS-100 en-pt, 2 sentidos) | 38.754 |
| summarization (XLSum pt) | 4.218 |
| extraglue_mnli | 9.815 |
| extraglue_qnli | 104.678 |
| extraglue_stsb | 5.710 |
| extraglue_mrpc | 1.725 |
| quati | 4.856 |
| personas_br (Claude, lote 1) | 43 |
| personas_br (Claude, lote 2) | 387 |
| personas_br (LLM local) | 4.783 |
| gigaverbo_v2_sft | 239.624 |
| legalpt (LLM local) | 4.917 |
| classiccpt (LLM local) | 4.542 |
| madlad400 (LLM local) | 4.297 |
| fineweb2 (LLM local) | 4.641 |
| hplt2 (LLM local) | 4.669 |
| legalpt (manual, colado no chat) | 100 |
| **TOTAL** (`data/sft_instruct2/manaca_sft.jsonl`) | **498.630** |

Bug real corrigido antes desse combine (colisão de tag em `--extra`): o
`sft/prepare_data.py` tagueava o arquivo por-fonte do manifest só pelo
basename, então os 3 `personas_br.jsonl` (lotes diferentes) e os 2
`legalpt_synth.jsonl` (LLM local vs. manual) se sobrescreviam entre si no
breakdown — o `manaca_sft.jsonl` combinado sempre esteve correto, mas o
manifest mentia sobre a origem. Corrigido incluindo o diretório-pai na
tag (`personas_br_personas_br`, `personas_br_batch2_personas_br`,
`legalpt_legalpt_synth`, `manual_batches_legalpt_synth`, etc.) — validado
ao vivo no log acima: as três fontes de personas_br e as duas de legalpt
aparecem com contagens distintas e corretas.

## Próximo passo

`data/sft_instruct2/manaca_sft.jsonl` (498.630 exemplos) está pronto para
o `sft/train.py`. Falta rodar o treino de fato e avaliar contra o mix v2
(`bench/`) pra confirmar que o volume extra converte em ganho real de
qualidade, não só em tamanho de dataset.
