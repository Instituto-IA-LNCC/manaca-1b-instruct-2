# Treino do Manacá-1B-Instruct-2 — plano e lições do v1/v2<br>Training plan for Manacá-1B-Instruct-2 — lessons from v1/v2

Antes de rodar o SFT do instruct-2, extraímos o que dava pra aprender das
corridas reais do predecessor (`docs/training/README.md`, `docs/evaluation/
instruct-eval-pt.md`, `dpo/README-onpolicy.md` — todos no repo
`manaca-1b-instruct`, de onde este projeto foi bifurcado). Duas categorias de
lição: **quanto tempo custa de verdade** (empírico, não teórico) e **o que
"mais dado" de fato compra** (medido em avaliação real, não suposição).

## 1. Wall-clock real (não teórico) — a correção mais importante

Os logs de v1/v2 têm `train_runtime` real, em 2× RTX A5000 sem NVLink (mesmo
hardware do instruct-2):

| Corrida | Exemplos | Passos | Tempo real | Amostras/seg |
|---|--:|--:|--:|--:|
| SFT v1 | 85.892 | 670 | 17,1h | 2,789 |
| SFT v2 | 141.404 | 1.104 | 27,1h | 2,897 |

Throughput real e estável: **~2,85 exemplos/seg** total, quase independente
do tamanho médio do exemplo — o log mostra por quê: a cada passo aparece
`[WARNING] stage3.py: ... pytorch allocator cache flushes ... high memory
pressure`. O gargalo é comunicação/memória do ZeRO-3 sem NVLink, não FLOPS
puros — uma estimativa teórica por pico de FLOPS (que fizemos antes de achar
esses logs) errou por >10x. **Lição: para este hardware, planeje por
amostras/segundo medido, não por FLOPS teóricos.**

Com esse throughput, o instruct-2 (mix rebalanceado, ~294k exemplos —
ver `docs/data/README-instruct2.md`) para 2 épocas seria ~57h (~2,4 dias).
Para o plano de 3 épocas desta run (ver §3), ~85h (~3,6 dias).

## 2. O que "mais dado" comprou no v1→v2 (medido, `instruct-eval-pt.md`)

v2 (mix maior: +aya, +translation, +summarization, +jaster completo) vs. v1:

| Métrica | v1 | v2 | Leitura |
|---|--:|--:|---|
| MT-Bench-PT GERAL | 2,29 | **2,56** | melhorou |
| IFEval-PT instr-loose | 34,0 | **36,0** | melhorou |
| Bateria classificação (assin2_rte/faquad/hatebr/hatespeech, f1-macro) | ~mix | ~mix | empate dentro do ruído |
| ENEM/BLUEX/OAB (acurácia) | 20,82/23,55/23,57 | 20,55/22,16/22,71 | **piorou** (ambos na faixa do acaso) |

**Lição para o instruct-2**: mais dado/diversidade tende a ajudar geração e
seguir-instrução (o que os benchmarks MT-Bench/IFEval medem), não ajuda
(ou piora levemente) múltipla-escolha — calibre expectativa: não esperar que
o instruct-2 resolva ENEM/OAB só por ter mais dado.

Duas lições adicionais (não urgentes agora, guardadas para as fases
seguintes): o 1º DPO (pares off-policy do GigaVerbo) **não melhorou / sobre-
otimizou** — só funcionou on-policy depois; e o safety-SFT precisou de 4
tentativas até achar o equilíbrio recusa/prestatividade — mudar só o volume
de dado sem controlar a composição chegou a **piorar os dois lados** ao
mesmo tempo (ver `dpo/README-onpolicy.md` e `sft/README-safety.md` no
repo-mãe).

## 3. Receita para esta run: qualidade acima de tempo

Já que tempo não é o fator limitante desta run, duas mudanças em relação ao
mix "fiel ao LLM-jp" puro (2 épocas, sem validação):

### 3.1 Cortes de fonte — por qualidade, não por velocidade

`extraglue_qnli` (104.678→20.000) e `gigaverbo_v2_sft` (239.624→120.000)
continuam capados mesmo sem restrição de tempo — a razão real não é o
tempo de GPU que eles economizam, é que cada um sozinho dominava demais o
mix (qnli: 6x maior que os 3 irmãos do extraglue somados, mesmo formato de
tarefa repetido; gigaverbo: quase metade dos exemplos de uma única fonte/
estilo). Com `--per_device_train_batch_size 1`, cada exemplo pesa igual no
gradiente independente do tamanho — deixar um formato dominar 20%+ do mix é
um risco real de overfitting em formato, não uma questão de velocidade.
(Ver `docs/data/README-instruct2.md` para a tabela completa do mix
rebalanceado.)

### 3.2 Mais épocas + seleção do MELHOR checkpoint por eval_loss (não o último)

Em vez de travar em "2 épocas" (o mínimo fiel ao LLM-jp, calibrado pro
volume de dado *deles*, ~4,7x menor que o nosso), separamos um split de
validação e treinamos 3 épocas, deixando o `eval_loss` decidir qual
checkpoint é o melhor de verdade — em vez de assumir que o último é o
melhor. Implementado em `sft/train.py` (`--load_best_model_at_end`,
commit `d201d46`): ao final do treino, o Trainer recarrega o checkpoint de
MENOR `eval_loss` (não o último) antes de salvar. Isso dá o benefício de
"mais tempo de treino" sem o risco de sobre-ajuste que motivou o LLM-jp a
travar em 2 épocas pro full FT deles.

LR (1e-5), warmup (0,1, cosine) e effective batch size (256 = PDBS 1 × 2
GPUs × GA 128) continuam iguais — já validados nas 2 corridas reais do v1/v2
e batendo com a tabela canônica do LLM-jp (`references/phase3-sft/
llm-jp-sft-pipeline-analise.md`, seção 52).

## 4. Comandos (na ordem)

```bash
# 1. Cortar as duas fontes dominantes (script completo na conversa/PR)
#    -> data/sft/extraglue/extraglue_qnli_capped.jsonl
#    -> data/sft/gigaverbo_v2_sft/gigaverbo_v2_sft_capped.jsonl

# 2. Recombinar o mix (sft/prepare_data.py, --extra com os *_capped.jsonl)
#    -> data/sft_instruct2/manaca_sft.jsonl (~294k, já embaralhado)

# 3. Separar um split de validação (últimas 3.000 linhas do arquivo já
#    embaralhado -- estatisticamente representativo)
#    -> data/sft_instruct2/manaca_sft_train.jsonl (~291k)
#    -> data/sft_instruct2/manaca_sft_eval.jsonl (3k)

# 4. Treinar (full FT, 3 épocas, seleção por eval_loss)
git pull origin main   # traz o --load_best_model_at_end (commit d201d46)
DETACH=1 ./scripts/docker/run_sft.sh sft \
  --model_name_or_path menezesbruno/manaca-1b-base \
  --data_files data/sft_instruct2/manaca_sft_train.jsonl \
  --eval_data_files data/sft_instruct2/manaca_sft_eval.jsonl \
  --full_finetuning --gradient_accumulation_steps 128 \
  --num_train_epochs 3 --load_best_model_at_end \
  --output_dir /workspace/checkpoints/manaca-1b-instruct-2-full
```

Acompanhar: `tail -f '<log>' | grep -Ei 'loss|epoch'` — observe `eval_loss`
a cada ~500 passos; se ele já estiver subindo bem antes do fim das 3 épocas,
o `--load_best_model_at_end` cuida de não salvar o modelo sobre-ajustado.

## 5. Estimativa de tempo desta run

~291k exemplos de treino × 3 épocas / 2,85 amostras/seg ≈ **85h (~3,6 dias)**
de wall-clock (mais o overhead das avaliações a cada 500 passos, pequeno
frente ao treino). `DETACH=1` é obrigatório (container sobrevive ao logout
SSH); os checkpoints (`save_steps=500`, `save_total_limit=3`) são a rede de
segurança se a sessão cair no meio — confirme que `--resume_from_checkpoint
auto` reconecta antes de depender dele por dias.
