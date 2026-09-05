# Safety-SFT — segurança pelo método que funciona (estilo AnswerCarefully)

> **Resultado e modelo escolhido:** ver `docs/evaluation/safety-alignment-pt.md`.
> O ponto escolhido da fronteira foi o **safe4** (75% de segurança held-out,
> 8.8% de over-refusal), e a receita dele é o **default** deste fluxo
> (`K_REFUSE=4`, benign-help disjunto, `N_GENERAL=200`, 3 épocas).

## Por que SFT e não DPO

O DPO on-policy gentil (fiel ao LLM-jp: beta 0.1, LR baixo) **não reverteu** o
prior de obediência do Manacá. O diagnóstico (`dpo/diag_dpo.py`) mostrou que o
adapter do DPO — carregado direto, sem merge — **obedece até os próprios prompts
de treino** (0/4), apesar da margem de treino ter chegado a 1.0. A margem mede a
razão de probabilidade das sequências específicas sob teacher-forcing; ela não
vira o primeiro token da geração livre quando o prior de obediência é forte.

A segurança do LLM-jp **não veio de DPO** — veio do **AnswerCarefully**, um
dataset de **SFT de segurança**. SFT otimiza a recusa como ALVO de cross-entropy,
o que muda a geração de verdade. Este é o pivô: fiel ao que o LLM-jp de fato fez.

## 1. Montar o dataset (na HPC, GPU)

Balanceado para não virar recusador cego: **recusar o nocivo** + **ajudar o
benigno** (respostas úteis geradas pelo próprio SFT, on-policy) + **dados gerais**
(amostra do SFT v2, preserva a utilidade ampla).

```bash
cd $HOME/manaca-1b
DETACH_OK=; K_REFUSE=2 N_GENERAL=200 GPUS=device=0 ./scripts/run_gen_safety_sft.sh
wc -l sft_safety/safety_sft.jsonl
```

Saída `sft_safety/safety_sft.jsonl` (`{instruction, input, output}`). O `output` é
sempre recusa OU resposta útil (nunca conteúdo nocivo), mas o arquivo mistura
dados gerais reamostrados, então fica **fora do git** (`.gitignore`); a receita
(prompts em `dpo/dpo_seeds/safety_prompts.jsonl` + banco de recusas em
`sft/gen_safety_sft.py`) já está versionada.

## 2. Treinar (LoRA sobre o v2, DDP, poucas épocas)

```bash
DETACH=1 make sft-safety
#   -> adapter em manaca-1b-instruct-v2-safe, mesclado em manaca-1b-instruct-v2-safe-merged
#   acompanhar: tail -f /data/manaca/checkpoints/sft-logs/sft_*.log | grep -Ei 'loss|epoch'
```

Usa DDP (não ZeRO-3) e LoRA r=32, 3 épocas, LR 1e-4. Não sobrescreve o
`instruct-v2-full`. Se quiser mesclar em fp32 (evita arredondar deltas do LoRA):

```bash
make sft-safety ARGS="--no_merge"
./scripts/docker/run_posttrain.sh python sft/merge_lora.py \
  --base /workspace/checkpoints/manaca-1b-instruct-v2-full \
  --adapter /workspace/checkpoints/manaca-1b-instruct-v2-safe \
  --out /workspace/checkpoints/manaca-1b-instruct-v2-safe-merged
```

## 3. Avaliar (o mesmo tribunal do DPO)

```bash
M=/data/manaca/checkpoints/manaca-1b-instruct-v2-safe-merged

# (a) SEGURANÇA held-out — recusa deve SUBIR muito (era 0/16 no SFT)
MODEL=$M LABEL=safe ./scripts/eval/run_overrefusal_pt.sh \
  --prompts bench/safety_pt/heldout.jsonl --out_dir bench/safety_pt/answers

# (b) OVER-REFUSAL — recusa em benignos NÃO pode subir (era 0% no SFT)
MODEL=$M LABEL=safe ./scripts/eval/run_overrefusal_pt.sh

# (c) diagnóstico rápido lado a lado
BASE=manaca-1b-instruct-v2-full ADAPTER=manaca-1b-instruct-v2-safe \
  MERGED=manaca-1b-instruct-v2-safe-merged ./scripts/eval/run_diag_dpo.sh
```

Depois, se (a) subiu e (b) ficou estável, rode a utilidade (IFEval-PT offline +
MT-Bench-PT com juiz) para confirmar que não regrediu. Só então o
`instruct-v2-safe` vira o instruct oficial.

## 4. Escalando para o instruct-2 — o gap de recall e o caso pró-social

O `safe4` (acima) ficou em 75% de recall de segurança porque o banco de
recusas é só 8 frases fixas e os 50 seeds nocivos não cobrem fraseado
AMBÍGUO (bullying, piada humilhante disfarçada). Uma pesquisa (2026)
confirmou que não existe dataset pronto pra isso em PT-BR — três scripts
novos mineram o que dá pra minerar com responsabilidade:

- **`dpo/dpo_seeds/mine_harmful_prompts.py`** — extrai os **prompts**
  (não as respostas) do mesmo subset harmful do
  `Polygl0t/gigaverbo-v2-preferences`, deduplicado contra os seeds
  existentes. Escala o pool de 50 para potencialmente centenas. Seguro de
  reproduzir (texto sintético, não fala de pessoa real) — mas a
  `categoria` atribuída é heurística; rode com `--sample-audit N` antes.
- **`sft/safety_seeds/mine_refusal_bank.py`** — o par deste: extrai as
  respostas `chosen` (já seguras) do mesmo subset harmful, um banco de
  recusas muito maior e diverso que os 8 templates fixos. Produz um
  **candidato**, não um banco pronto: rode com `--sample-audit N` primeiro
  e confira manualmente antes de usar (as categorias de dano desse dataset
  não são documentadas publicamente).
- **`dpo/dpo_seeds/build_ambiguous_seeds.py`** — confirma estatisticamente
  (contagens agregadas, sem reproduzir texto real de ninguém) quais
  categorias de ódio/bullying de fato ocorrem em PT-BR (via ToLD-Br e
  HateBR), e gera novos seeds nocivos por TEMPLATE escrito à mão cobrindo
  fraseado indireto — o gap documentado no recall.

```bash
python dpo/dpo_seeds/mine_harmful_prompts.py --sample-audit 15    # audite antes
python dpo/dpo_seeds/mine_harmful_prompts.py                      # grava o candidato

python sft/safety_seeds/mine_refusal_bank.py --sample-audit 15    # audite antes
python sft/safety_seeds/mine_refusal_bank.py                      # grava o candidato

python dpo/dpo_seeds/build_ambiguous_seeds.py --stats-only        # so a evidencia
python dpo/dpo_seeds/build_ambiguous_seeds.py                     # gera os seeds

# depois de auditar, juntar tudo:
cat dpo/dpo_seeds/safety_prompts.jsonl \
    dpo/dpo_seeds/safety_prompts_ambiguous.jsonl \
    dpo/dpo_seeds/safety_prompts_mined.jsonl \
    > dpo/dpo_seeds/safety_prompts_v2.jsonl
```

O caso pró-social ("ajudar amigo em crise" sendo recusado por engano) é
diferente — fica do lado `benign_help.jsonl`, e nenhum dataset pronto foi
encontrado para ele; a resposta precisou ser escrita à mão com revisão,
não minerada.

## 5. Categoria "ajudar_em_crise" — resolvido (resposta pronta, revisada)

`sft/safety_seeds/benign_help.jsonl` ganhou 5 exemplos novos (`bh_51`-`bh_55`,
`tipo: "ajudar_em_crise"`) cobrindo exatamente o gap documentado (`sens_12`:
"amigo com pensamentos suicidas"), mais os casos adjacentes óbvios (risco
imediato, sinais de alerta, crise em primeira pessoa, automutilação) — todos
com resposta de ajuda + indicação do CVV (188, cvv.org.br) e, quando o risco
é imediato, SAMU (192). Texto revisado e aprovado antes de gravar (não
gerado automaticamente).

**Diferença técnica**: esses 5 exemplos já têm o campo `output` preenchido
no `benign_help.jsonl` (os outros 50 não têm). O `sft/gen_safety_sft.py`
foi ajustado para usar essa resposta pronta diretamente quando ela existe
— **não precisa de `--model`** para esses — em vez de gerar on-policy (que
é o que ainda acontece com o resto do arquivo, e é exatamente onde o
modelo sem alinhamento não sabe mencionar o CVV sozinho). Rodar
`python sft/gen_safety_sft.py --out ... --seeds ... --benign
sft/safety_seeds/benign_help.jsonl` sem `--model` já inclui as 5 respostas
de crise no dataset; o resto do contraste benigno continua exigindo o
checkpoint do SFT para a parte on-policy.
