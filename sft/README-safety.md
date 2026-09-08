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

## 6. Escalando o lado "ajuda" — `generate_benign_prompts.py`

Os itens 1-3 (§4-5) escalam o lado RECUSA e resolvem o caso "ajudar em
crise", mas o lado AJUDA continuava travado em 55 prompts, todos escritos
à mão. `sft/safety_seeds/generate_benign_prompts.py` escala esse lado:
gera **novos prompts** (não respostas) benignos-mas-de-fraseado-sensível
via Claude, em 10 categorias inspiradas no vão que separa "recusa" de
"ajuda" (linha do XSTest em inglês, sem versão pronta em PT-BR): segurança
digital, saúde não-emergencial, autodefesa pessoal, ficção com temas
pesados, direitos legais cotidianos, finanças e golpes, conflitos e
relacionamentos, prevenção de acidentes domésticos, denúncia de
abuso/assédio, segurança no trânsito.

**Por que gerar aqui é seguro sem auditoria obrigatória** (diferente dos
mineradores do lado nocivo, §4): o pior caso de um prompt malfeito é um
exemplo de baixa qualidade, não um exemplo perigoso — o pedido em si já é
benigno por construção do prompt de geração. O script gera só os
**prompts**; as respostas continuam on-policy via `gen_safety_sft.py`
(exceto se uma categoria nova também se revelar problemática on-policy,
caso em que o tratamento é o mesmo já usado para "ajudar_em_crise":
resposta pronta, escrita e revisada à mão).

Deduplica contra `benign_help.jsonl` E contra `bench/overrefusal_pt/prompts.jsonl`,
preservando o requisito existente de que o benign-help fique disjunto da
avaliação de over-refusal.

```bash
make safety-generate-benign-dry-run                 # mostra os prompts, sem custo
make safety-generate-benign                          # gera de verdade (Claude, chamadas sincronas)
make safety-generate-benign ARGS="--categories seguranca_digital,autodefesa_pessoal"

# depois de gerar, juntar com o original:
cat sft/safety_seeds/benign_help.jsonl \
    sft/safety_seeds/benign_help_generated.jsonl \
    > sft/safety_seeds/benign_help_v2.jsonl
```

Saída: `sft/safety_seeds/benign_help_generated.jsonl` (schema igual ao
original: `{id, tipo, prompt}`, IDs continuando de `bh_056`) +
`benign_help_generated_manifest.json` com contagem por categoria, tokens
gasto e custo estimado (chamadas síncronas, sem desconto de Batches —
volume pequeno o suficiente para não justificar a API de lotes).

## 7. Meta explícita: ~2.000 exemplos (acima do AnswerCarefully v2)

Decisão: em vez de "mais que 411", a meta agora é **~2.000** exemplos de
safety-SFT — acima até do AnswerCarefully v2 do LLM-jp (1.800). O lado
AJUDA já tinha `generate_benign_prompts.py` (§6); faltava o par do lado
RECUSA: **`dpo/dpo_seeds/generate_harmful_prompts.py`** gera novos
**pedidos** nocivos via API Claude (nunca as respostas — a resposta
continua sendo um dos templates fixos de `gen_safety_sft.py`), mesma
prática usada por datasets públicos de segurança (AnswerCarefully,
do-not-answer, HarmBench: um LLM gera as PERGUNTAS de red-team, não as
respostas nocivas). Reaproveita as 14 categorias já usadas em
`safety_prompts.jsonl`, **exceto** `autolesao` e `sexual` — essas duas
continuam exigindo tratamento manual revisado (mesmo cuidado já dado à
categoria "ajudar_em_crise" em §5), não geração em lote.

**A conta** (receita "safe4": `K_REFUSE=4`, `N_GENERAL=200`):

| Componente | Hoje (hand-written) | Alvo |
|---|---|---|
| seeds nocivos (recusa) | 50 → recusa=200 | ~400 → recusa≈1.600 |
| benign-help (ajuda) | 55 | ~255 |
| dados gerais | 200 | 200 |
| **TOTAL** | **455** (medido: 411) | **≈2.055** |

```bash
# audita antes de gravar (recomendado, mesmo nao sendo obrigatorio p/ este script)
make safety-generate-harmful-audit

# gera o candidato (default: 14 categorias x ate 30 pedidos/categoria)
make safety-generate-harmful

# confirma o progresso a qualquer momento, sem GPU e sem gastar API
# (so soma os arquivos de seeds/benign que ja existem no disco)
make safety-project-size
```

`safety-project-size` (`sft/safety_seeds/project_safety_sft_size.py`)
aplica a mesma fórmula do `gen_safety_sft.py` para projetar o total a
partir do que já está no disco — inclui os candidatos ainda não
auditados por padrão (avisa quais são) e aceita `--k_refuse`/`--n_general`
para simular a receita desejada.

Depois de auditar (`--sample-audit`) e aprovar os candidatos de
`generate_harmful_prompts.py` e `mine_harmful_prompts.py`, junte tudo:

```bash
cat dpo/dpo_seeds/safety_prompts.jsonl \
    dpo/dpo_seeds/safety_prompts_ambiguous.jsonl \
    dpo/dpo_seeds/safety_prompts_mined.jsonl \
    dpo/dpo_seeds/safety_prompts_generated.jsonl \
    > dpo/dpo_seeds/safety_prompts_v2.jsonl

cat sft/safety_seeds/benign_help.jsonl \
    sft/safety_seeds/benign_help_generated.jsonl \
    > sft/safety_seeds/benign_help_v2.jsonl

python sft/gen_safety_sft.py --out sft_safety/safety_sft_v2.jsonl \
    --seeds dpo/dpo_seeds/safety_prompts_v2.jsonl \
    --benign sft/safety_seeds/benign_help_v2.jsonl \
    --k_refuse 4 --general_data data/sft_v2/manaca_sft.jsonl --n_general 200 \
    --model <checkpoint-do-sft-v2>
```

## 8. Auditoria dos 4 scripts de escala (instruct-2, resultado real)

Rodei `--stats-only`/`--sample-audit`/`--dry-run` nos 4 scripts do §4 antes
de gravar qualquer coisa. Dois bugs reais corrigidos, uma fonte abandonada,
um problema de modelo (não de código) resolvido:

- **`build_ambiguous_seeds.py`**: o repo `franciellevargas/HateBR` não expõe
  mais as 9 colunas booleanas de categoria no HF (só anotadores brutos +
  `label_final`) — `count_categories()` retornava **zero em toda categoria,
  silenciosamente**. Trocado para `ruanchaves/hatebr` (já usado em
  `sft/jaster/build_jaster.py`), que tem as colunas certas — números reais
  confirmados (racismo 7, homofobia 13, sexismo 43, intolerância religiosa
  44, gordofobia 23, antissemitismo 1). A função agora falha alto se
  nenhuma categoria bater com o schema real, em vez de silenciar como zero.

- **`mine_harmful_prompts.py` — ABANDONADO.** A auditoria de amostra (30
  exemplos, 2 rodadas) achou conteúdo grave demais pra um filtro de
  palavra-chave resolver: plano de assassinato de uma pessoa **real e
  identificável** (o CEO da Apple), conteúdo antissemita repetido e
  extremo, terrorismo em massa em local real (aeroporto JFK), conteúdo
  envolvendo menores. Um filtro de slur (`contains_slur()`) foi adicionado
  e ajuda um pouco, mas o problema não é vocabulário, são cenários
  inteiros — mesmo padrão de decisão do aes_enem/Portuguese-PD: quando a
  auditoria revela que a fonte é inadequada, a resposta é abandonar, não
  filtrar mais. **Não usar esta fonte para os prompts do lado "recusa".**
  Todos os candidatos minerados/gerados (deste script e dos outros 3) agora
  ficam fora do git por padrão (`.gitignore`) — nunca commitar direto.

- **`mine_refusal_bank.py`**: mesmo bug do `<think>` cru já documentado em
  `dpo/gigaverbo_v2_preferences/README.md` (a fonte é a mesma). Corrigido
  com `strip_think()` — bônus: recuperou entradas que antes eram
  descartadas por estourar o limite de tamanho (o raciocínio cru inflava o
  texto além de `MAX_TEXT_LEN`; sem ele, a resposta final cabe). Subiu de
  3.191 para 3.996 recusas únicas só no subset `harmful-reasoning`. Amostra
  de 15 conferida: limpa, sem `<think>`, boa qualidade em PT-BR. **Esta
  fonte está OK pra usar** — extrai só as recusas (`chosen`), não os
  pedidos nocivos.

- **`generate_harmful_prompts.py`**: as 3 categorias mais "operacionalmente
  sensíveis" (`ciberataque`, `malware`, `privacidade`) retornavam resposta
  **vazia** com `claude-opus-5` (não é bug de parsing — não há texto
  nenhum pra extrair). Testado com `claude-sonnet-5`: gerou 30/30/30
  normalmente, conteúdo plausível e apropriado (spyware, phishing, quebra
  de senha wifi, ransomware — nada de nomes reais, nada extremo). Trocado
  o default do script pra `claude-sonnet-5` (também mais barato). As
  outras 11 categorias já funcionavam com qualquer um dos dois modelos.

**Conclusão prática**: a meta de ~2.055 exemplos (§7) continua de pé, mas
sem o `mine_harmful_prompts.py` — o lado "recusa" escala via
`generate_harmful_prompts.py` (agora com `claude-sonnet-5`, ~350 prompts
nas 14 categorias) + `build_ambiguous_seeds.py` (templates, evidência real)
+ os 50 originais, não via mineração do `gigaverbo-v2-preferences`.
