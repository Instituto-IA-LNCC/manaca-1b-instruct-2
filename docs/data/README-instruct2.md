# Corpus de SFT do Manacá-1B-Instruct-2 (em construção)<br>SFT corpus of Manacá-1B-Instruct-2 (in progress)

Ficha do dado gerado pra fase "instruct-2" — a mesma disciplina do
[`docs/data/README.md`](README.md) (v1/v2): só o que foi de fato gerado e
confirmado, com contagem real (`wc -l`), não estimativa. Complementa (não
substitui) o mix v1/v2 — ainda não há um `sft/prepare_data.py` rodado
combinando tudo isto num `manaca_sft.jsonl` de instruct-2 (próximo passo).

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
| ExtraGLUE — stsb | idem | 1.379 | PORTULAN/extraglue |
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
| **TOTAL** | | **391.320** | |

Para comparação: v1 = 85.892, v2 = 141.404 (`docs/data/README.md`). O
instruct-2 já soma **2,8x o v2**, majoritariamente via LLM local (sem
custo de API além de auditorias pontuais via Claude Haiku).

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

## Próximo passo

Montar o `sft/prepare_data.py` (ou um script equivalente) que combine todos
os arquivos da tabela acima num `manaca_sft_instruct2.jsonl` único, com
dedup/shuffle — ainda não feito.
