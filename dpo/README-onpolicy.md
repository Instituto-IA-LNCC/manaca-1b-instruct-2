# DPO on-policy — fiel ao LLM-jp, adaptado à nossa realidade

Corrige o 1º DPO (que usou pares off-policy do GigaVerbo e não melhorou / sobre-otimizou).
Aqui os pares de preferência são gerados **pelo próprio instruct** (on-policy) e
focam nos gaps que os benchmarks mediram: **segurança** (o modelo obedece a pedidos
nocivos) e **seguir instrução**. Método fiel ao LLM-jp (dados por juiz + linha de
segurança tipo AnswerCarefully); adaptações: on-policy, PT-BR, poucas épocas com
parada pelo sinal, e decisão pela avaliação (MT-Bench-PT / IFEval-PT).

## 1. Gerar os pares (na HPC, GPU)

```bash
cd $HOME/manaca-1b

# (a) SEGURANCA — regra, offline. Maior retorno. (rejected = obediencia do modelo)
DETACH=1 MODE=safety      ./scripts/run_gen_dpo_pairs.sh

# (b) INSTRUCAO — regra (checkers do IFEval), offline
DETACH=1 MODE=instruction ./scripts/run_gen_dpo_pairs.sh

# (c) QUALIDADE — juiz LLM (precisa de chave)
DETACH=1 MODE=quality ANTHROPIC_API_KEY=sk-ant-... ./scripts/run_gen_dpo_pairs.sh
```

Saída em `dpo_onpolicy/<mode>.jsonl` (esquema `{instruction, input, chosen, rejected}`).
**Não versionado** (o `rejected` de segurança é conteúdo nocivo; `.gitignore` cobre `dpo_onpolicy/`).

Junte os fluxos que quiser num arquivo só:
```bash
cat dpo_onpolicy/safety.jsonl dpo_onpolicy/instruction.jsonl > dpo_onpolicy/mix.jsonl
```

## 2. Treinar o DPO (poucas épocas + parada pelo sinal)

Fiel ao `llm-jp-dpo` (LoRA, LR 5e-7, ref = SFT v2), mas **1-2 épocas** e salvando por
época, para escolher o checkpoint antes da sobre-otimização (o erro do 1º DPO):

```bash
make dpo ARGS="--model_name_or_path /workspace/checkpoints/manaca-1b-instruct-v2-full \
  --data_files dpo_onpolicy/mix.jsonl \
  --output_dir /workspace/checkpoints/manaca-1b-instruct-v2-dpo2 \
  --beta 0.1 --learning_rate 5e-7 --num_train_epochs 2 \
  --save_strategy epoch --save_total_limit 5 --lora_r 64"
```

O alvo `make dpo` usa `configs/accelerate_config_ddp.yaml` por default (DDP, não
ZeRO-3): para LoRA de 1.7B o modelo cabe folgado em 1 GPU, e o ZeRO-3, ao
particionar os pesos, quebra o gradient checkpointing do DPO (o adapter liga/
desliga entre política e referência) com `CheckpointError` de metadados. Para
sobrepor: `ACCELERATE_CONFIG=configs/accelerate_config_zero3.yaml make dpo ...`
(necessário apenas para full FT).

No log, observe `rewards/chosen` e `rewards/margins`: **pare/escolha o checkpoint
antes de `rewards/chosen` virar negativo** (sinal de sobre-otimização). Se derivar
rápido, suba `--beta` para 0.2-0.3 (mantém o modelo mais perto do SFT).

## 3. Avaliar (só fica com o DPO se melhorar)

Comparar o DPO contra o **instruct-v2** com a nossa avaliação:
- **MT-Bench-PT, categoria `seguranca`** deve subir bastante (é o alvo).
- **Over-refusal** (`bench/overrefusal_pt`) não pode subir — o risco do mix ser
  96% "recuse" é o modelo virar recusador demais e negar pedidos legítimos.
- **MT-Bench-PT geral + IFEval-PT** não podem regredir.
- CALAME/LAMBADA não podem cair (base intacta).

```bash
# over-refusal: ANTES (SFT) e DEPOIS (DPO). 'sensivel_legitimo' nao pode subir.
LABEL=manaca-instruct-v2 ./scripts/eval/run_overrefusal_pt.sh
MODEL=/data/manaca/checkpoints/manaca-1b-instruct-v2-dpo2-merged \
  LABEL=manaca-instruct-v2-dpo2 ./scripts/eval/run_overrefusal_pt.sh
```

```bash
# gera respostas do DPO e julga (mesma infra do bench)
MODEL=/data/manaca/checkpoints/manaca-1b-instruct-v2-dpo2-merged \
  LABEL=manaca-instruct-v2-dpo2 ./scripts/eval/run_mtbench_pt.sh
python3 bench/mtbench_pt/judge.py --answers bench/mtbench_pt/answers/manaca-instruct-v2-dpo2.jsonl \
  --out bench/mtbench_pt/judged/manaca-instruct-v2-dpo2.jsonl
python3 bench/mtbench_pt/report.py \
  bench/mtbench_pt/judged/manaca-instruct-v2.jsonl \
  bench/mtbench_pt/judged/manaca-instruct-v2-dpo2.jsonl --out bench/mtbench_pt/mtbench-pt-dpo
```

Se `seguranca` sobe e o resto se mantém, o DPO valeu e vira o `instruct-v2-dpo` oficial.
Senão, o instruct-v2 (SFT) continua sendo o melhor, e a evidência fica registrada.

## 4. O resultado real do v2 (com evidência, não só prosa) — e a lição mais importante do projeto

Achado em [`docs/evaluation/safety-alignment-pt.md`](../docs/evaluation/safety-alignment-pt.md)
(já existe neste repo, herdado do fork — só não tinha sido lido até agora
nesta fase). Números reais, não estimativa:

| DPO | passos | rewards/chosen | margins | acurácia | segurança held-out |
|---|---|---|---|---|---|
| beta 0.1, LR 5e-7, 2 ép | 24 | ~0.00 | ~0.006 | 0.56 | — (sem efeito) |
| beta 0.1, LR 1e-6, 3 ép | 153 | **+0.85** | **1.03** | ~1.0 | **0/16** |

A 2ª rodada teve curva de recompensa **perfeita** (margem 1.0, acurácia ~1.0,
sem sobre-otimização) — e mesmo assim a geração livre **não mudou nada**: o
`dpo/diag_dpo.py` confirmou que o **adapter sem merge** (efeito real do DPO,
sem arredondamento) obedece **0/4 até nos próprios prompts de treino**. O bug
de merge em bf16 (o que eu tinha levantado como hipótese aqui antes de achar
esse documento) foi **formalmente descartado com essa evidência** — não é
merge, é método.

**Por que isso importa mais do que parece**: a métrica de treino do DPO
(`rewards/margin`, acurácia) mede a razão de probabilidade de duas sequências
específicas sob *teacher-forcing* — ela pode ficar perfeita sem nunca virar a
decisão do **primeiro token** na geração livre (greedy/amostragem), que é o
que realmente determina o comportamento. Quando o prior (aqui, "obedecer") é
forte e o dataset é pequeno, **DPO pode "treinar perfeito" e mudar zero
comportamento visível**. **Nunca confie em `rewards/margin`/acurácia como
sinal de sucesso — sempre valide com geração livre de verdade**
(`dpo/diag_dpo.py` ou amostras manuais), não só o log de treino.

**O que funcionou de verdade** (mesmo doc, §4): não foi um 3º ajuste de DPO —
foi trocar de método. **Safety-SFT** (cross-entropy direto no alvo de recusa,
estilo AnswerCarefully do LLM-jp) reverteu o comportamento onde o DPO não
conseguiu: base recusa 0/4, adapter+merged do safety-SFT recusam 4/4 (treino
e held-out), com o merge preservando o efeito dessa vez (prova incluída no
mesmo diagnóstico). Config vencedora (`safe4`, escolhida entre 4 pontos da
fronteira recusa:benigno): `K_REFUSE=4`, banco de contraste benigno
**disjunto** da avaliação (50 exemplos), 200 exemplos gerais, LoRA r=32, DDP,
3 épocas, LR 1e-4 — reproduzível via `sft/README-safety.md` no repo-mãe.

**Lição de higiene de dado, à parte**: as duas primeiras tentativas
(`safe`/`safe2`) usaram um conjunto de contraste benigno **contaminado** — os
MESMOS prompts que apareciam na avaliação de over-refusal — o que inflava a
métrica sem significar nada. Só ficou confiável a partir do `safe3`, com um
conjunto disjunto. **Isso já foi evitado por construção no split de
validação do SFT do instruct-2** (`manaca_sft_eval.jsonl` é excluído de
`manaca_sft_train.jsonl` antes do treino, nunca aparece nos dois), mas vale
manter essa disciplina explícita em qualquer dado de avaliação futuro
(inclusive quando gerarmos os pares on-policy `MODE=safety`/`MODE=instruction`
pro DPO/safety-SFT do instruct-2).

**Conclusão prática pro instruct-2**: se a intenção da 3ª tentativa de DPO
(`gigaverbo_v2_preferences` + `reward_aira`) for mudar comportamento
categórico forte (segurança, recusa) — o precedente diz que **SFT direcionado
tem mais chance de funcionar que DPO** para esse tipo de mudança neste
projeto. DPO pode ainda valer a pena pra ajustes mais sutis de estilo/
preferência onde não se está brigando contra um prior forte — mas pra
segurança especificamente, considerar repetir a receita do safety-SFT
(LoRA + cross-entropy no alvo) em vez de insistir em DPO puro.
