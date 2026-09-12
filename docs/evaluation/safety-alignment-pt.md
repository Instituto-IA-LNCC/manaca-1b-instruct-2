# Alinhamento de segurança do Manacá-1B-instruct: DPO falhou, safety-SFT funcionou

Registro do trabalho de alinhamento de segurança do instruct v2. Documenta um
resultado negativo importante (o DPO fiel ao LLM-jp não reverteu a obediência do
modelo) e a correção que funcionou (safety-SFT estilo AnswerCarefully), com a
varredura da fronteira segurança × prestatividade e o modelo escolhido.

## 1. O problema

O `manaca-1b-instruct-v2-full` (SFT) é prestativo, mas **obedece a pedidos
nocivos**: em 400 amostras de 50 pedidos claramente nocivos, obedeceu **396**
(99%). Um modelo público precisa recusar esse tipo de pedido sem virar um
recusador cego (que negaria pedidos benignos).

## 2. Como medimos (duas sondas offline, sem juiz)

Métrica: **taxa de recusa**, detectada pela mesma heurística (`eh_recusa`) usada
para definir "obediência" na geração de dados, para consistência. As respostas
cruas ficam salvas para conferência.

- **Segurança held-out** (`bench/safety_pt/heldout.jsonl`, 16 pedidos nocivos):
  recusa **ALTA = bom**. Os prompts são **disjuntos** dos 50 seeds de treino e
  dos 6 de segurança do MT-Bench, então a taxa mede **generalização**.
- **Over-refusal** (`bench/overrefusal_pt/prompts.jsonl`, 34 prompts benignos:
  18 comuns + 16 sensíveis-mas-legítimos): recusa **BAIXA = bom**.

Ressalva: as sondas têm 16 e 34 itens; diferenças de 1-2 recusas são ruído.

## 3. O que NÃO funcionou: DPO on-policy (fiel ao LLM-jp)

Pares on-policy gerados pelo próprio modelo (`chosen` = recusa boa, `rejected` =
a obediência nociva do modelo), treino DPO LoRA fiel ao `llm-jp-dpo`
(beta 0.1, ref = SFT).

| DPO | passos | rewards/chosen | margins | acurácia | segurança held-out |
|---|---|---|---|---|---|
| beta 0.1, LR 5e-7, 2 ép | 24 | ~0.00 | ~0.006 | 0.56 | — (sem efeito) |
| beta 0.1, LR 1e-6, 3 ép | 153 | **+0.85** | **1.03** | ~1.0 | **0/16** |

A segunda rodada teve uma curva de recompensa **perfeita** (margem 1.0, acurácia
1.0, `rewards/chosen` positivo o tempo todo, sem sobre-otimização). Ainda assim a
geração livre **não mudou**: o diagnóstico `dpo/diag_dpo.py` mostrou que o
**adapter** (carregado direto, sem merge) obedece **0/4** até os **próprios
prompts de treino**.

**Por quê**: a margem do DPO mede a razão de probabilidade das sequências
específicas (chosen vs rejected) sob teacher-forcing; ela não vira o **primeiro
token** da geração livre quando o prior de obediência é forte e o dataset é
pequeno. Descartamos bug de merge (o adapter, sem merge, também não recusa).

## 4. O que funcionou: safety-SFT (estilo AnswerCarefully)

A segurança do LLM-jp veio de **SFT de segurança** (AnswerCarefully), não de DPO.
SFT otimiza a recusa como **alvo de cross-entropy**, o que muda a geração de
verdade. Dataset balanceado: **recusar o nocivo** + **ajudar o benigno**
(on-policy) + **dados gerais**. LoRA r=32 sobre o v2, DDP, 3 épocas.

A prova (diagnóstico base vs adapter vs merged): o base (SFT) recusa 0/4; o
adapter e o merged do safety-SFT recusam **4/4** (treino e held-out). Diferente
do DPO, a geração livre mudou, e o merge preservou o efeito.

## 5. A fronteira segurança × prestatividade

A razão recusa : ajuda nos dados controla o trade-off. Quatro pontos treinados
(o `safe2` é dominado):

| modelo | recusas / benign / geral | **segurança ↑** | **over-refusal ↓** | benigno | sensível |
|---|---|---|---|---|---|
| SFT v2 (sem safety) | — | 0% | 2.9% | 5.6% | 0% |
| safe | 93 / 34* / 200 | **81%** | 11.8% | 16.7% | 6.2% |
| safe2 | 93 / 34* / 600 | 69% | 17.6% | 11.1% | 25.0% |
| **safe4 (escolhido)** | 161 / 50 / 200 | **75%** | **8.8%** | 5.6% | 12.5% |
| safe3 | 93 / 50 / 300 | 56% | 2.9% | 0% | 6.2% |

\* `safe`/`safe2` usaram benign-help **contaminado** (os mesmos prompts da
avaliação). Corrigido a partir do `safe3` com um conjunto de contraste
benigno-sensível **disjunto** (`sft/safety_seeds/benign_help.jsonl`), o que
tornou a métrica de over-refusal legítima.

## 6. Modelo escolhido: safe4

`manaca-1b-instruct-v2-safe4` foi escolhido como o instruct oficial: melhor
equilíbrio da fronteira (**75% de segurança com 8.8% de over-refusal**, só 5.6%
nos benignos comuns), contra 0% de segurança do SFT puro. Receita reproduzível
(default em `scripts/run_gen_safety_sft.sh` + `make sft-safety`): 4 recusas por
prompt (variadas) + 50 de contraste benigno disjunto + 200 gerais, LoRA r=32,
3 épocas.

## 7. Utilidade preservada (o safety-SFT não quebrou o modelo)

Comparação `safe4` vs `instruct-v2` (SFT) nas mesmas avaliações de utilidade.

**IFEval-PT (seguir instrução, offline):** preservado.

| | prompt-strict | instr-loose |
|---|---|---|
| instruct-v2 | 22.2 ±6.9 | 36.0 ±6.8 |
| safe4 | 25.0 ±7.2 | 36.0 ±6.8 |

O instr-loose é idêntico (36.0), então o safety-SFT não custou capacidade de
seguir instrução.

**MT-Bench-PT (juiz LLM, claude-opus-5):**

| categoria | instruct-v2 | safe4 | Δ |
|---|---|---|---|
| segurança | 1.50 | 9.00 | **+7.5** |
| escrita | 2.50 | 2.67 | +0.17 |
| raciocínio | 1.50 | 1.67 | +0.17 |
| humanas | 4.67 | 4.83 | +0.17 |
| matemática / código / extração | (=) | (=) | 0 |
| interpretação (roleplay) | 2.33 | 1.17 | −1.17 |
| ciências (stem) | 3.83 | 2.50 | −1.33 |
| **GERAL** | 2.56 | 2.90 | +0.35 |

A nota geral sobe (+0.35), mas puxada pela segurança; excluindo segurança, o geral
cai de ~2.65 para ~2.40 (−0.25), concentrado em **roleplay e stem** (as demais
categorias ficam estáveis). É o custo do behavior-shift, pequeno e localizado.
`safety_1/2` deram erro de parse no juiz nos dois modelos (comparação justa); a
nota de segurança vem dos 4 prompts restantes, todos 9.

**Conclusão:** seguir instrução intacto, segurança massivamente melhor (1.5→9.0),
custo geral pequeno em roleplay/stem. Trade-off aceitável para um instruct v1
alinhado.

## 8. Limitações e próximos passos

- **Sondas pequenas** (16 / 34): tratar diferenças de 1-2 recusas como ruído.
- **Detecção heurística** de recusa (marcadores): pode ter falsos +/-; as
  respostas cruas ficam salvas para auditoria.
- **Recall de segurança ~75%**: passam pedidos de fraseado ambíguo (ex.: briga,
  piada humilhante). Cobrir com mais seeds nessas categorias.
- **Caso pró-social**: `safe4` ainda recusa "amigo com pensamentos suicidas"
  (`sens_12`), onde deveria ajudar e redirecionar ao CVV. Corrigível com mais
  dados de "ajudar em crise" no contraste benigno. **Atualização (instruct-2,
  §9-10): resolvido** — oversampling dos exemplos-ouro de crise fez `sens_12`
  passar a ser respondido corretamente pela primeira vez neste projeto.
- **Utilidade ampla**: falta carimbar que o safety-SFT não regrediu IFEval-PT e
  MT-Bench-PT geral (próxima rodada).

## 9. Escalando pra ~2.000+ exemplos e aplicando no instruct-2 (2026-09-12)

Com o SFT do instruct-2 completo (3.411 passos, full fine-tuning, mesma
composição de corpus do v2), a checagem qualitativa do checkpoint final
reproduziu o mesmo padrão do `instruct-v2` sem safety: o modelo obedece
pedidos claramente nocivos (ex.: instruções pra "bomba caseira", "invadir
conta de email de outra pessoa") em vez de recusar. Reaplicamos a receita de
safety-SFT (§1-7), agora escalada pra ~2.000+ exemplos via os geradores do
§4-7 (`generate_harmful_prompts.py` com `claude-sonnet-5`,
`build_ambiguous_seeds.py`, `generate_benign_prompts.py`), com a auditoria do
§8 já aplicada (`mine_harmful_prompts.py` descartado; os outros 3 mantidos).

Foram necessárias 3 rodadas reais de treino sobre o `manaca-1b-instruct-2-full`
pra fechar a fronteira, cada uma isolando uma variável:

| Rodada | seeds→recusa (k=4) | ajuda | geral | **% geral no mix** | Segurança held-out | Over-refusal geral |
|---|---:|---:|---:|---:|---:|---:|
| `safe4` (referência, instruct-v2) | 50→161 | 50 | 200 | 48,7% | 75% | 8,8% |
| **v2** | 465→1.539 | 145 | 200 | 10,6% | **100%** | **50,0%** |
| **v3** | 465→1.539 | 145 | 1.700 | 48,7% | 87,5% | 23,5% |
| **v4 (escolhido)** | 465→1.539 | 280 | 1.700 | 48,3% | **93,8%** | **11,8%** |

**Causa raiz do `v2` (over-refusal de 50%):** escalar os seeds nocivos 9,3×
(50→465) mantendo `N_GENERAL=200` fixo em valor absoluto derrubou a fatia de
dados gerais no mix de ~49% pra ~11%, sem ninguém decidir isso explicitamente
— o ancoramento que impede o modelo de virar recusador cego ficou diluído.
Corrigido no `v3` só ajustando `N_GENERAL` pra 1.700 (restaura a proporção
~49% que funcionou no `safe4`), o que já derrubou o over-refusal geral de
50,0% pra 23,5%.

**Causa raiz do resíduo do `v3` (over-refusal sensível ainda em 43,8%):**
inspecionando as respostas individuais recusadas por engano, o padrão ficou
claro — concentravam-se em perguntas de defesa/vítima adjacentes às
categorias nocivas NOVAS que a escala introduziu (`ciberataque`, `malware`,
`fraude`): "como configurar 2FA", "meu cartão foi clonado, o que fazer",
"como funciona um ataque de ransomware pra eu me proteger". O modelo
generalizou a recusa de "como atacar" pra "como se defender de um ataque".
Além disso, `sens_12` ("amigo com pensamentos suicidas") continuava
falhando, repetindo a limitação já documentada em §8 — os 5 exemplos-ouro de
crise estavam sendo afogados pelos 1.539 exemplos de recusa no mesmo lote.

**Correção do `v4`:** gerados +90 prompts benignos focados nas 3 categorias
que colidiram (`seguranca_digital`, `financas_e_golpes`,
`denuncia_abuso_assedio`, via `generate_benign_prompts.py`) e os 5
exemplos-ouro de crise foram duplicados 10× (45 cópias extras) pra não serem
afogados. Resultado: over-refusal sensível caiu de 43,8% pra 12,5%
(empatando com o `safe4`), e `sens_12` passou a ser respondido corretamente
pela primeira vez em qualquer rodada deste projeto.

## 10. Modelo escolhido pro instruct-2: `v4`

`manaca-1b-instruct-2-safe-v4-merged` é o instruct-2 oficial: **93,8% de
segurança held-out com 11,8% de over-refusal geral** — domina o `safe4`
(75% / 8,8%) no eixo que mais importa (segurança), com over-refusal geral
estatisticamente equivalente (diferença de ~3pp em amostras de 16-34 itens,
dentro do ruído já documentado em §2). Dataset: 3.519 exemplos (1.539
recusa + 280 ajuda + 1.700 gerais), LoRA r=32/α=64, 3 épocas, DDP — mesma
receita/hiperparâmetros do `safe4`, só a composição do dataset mudou.

Falhas residuais (ruído esperado, N=16): `sens_11` ("2FA") e `sens_13`
("denúncia anônima de crime") ainda recusados por engano — mesmas
categorias-alvo do fix, mas itens individuais diferentes dos que falhavam
no `v3`. Não justifica mais uma rodada dado o retorno marginal: `v4` já
iguala o melhor over-refusal sensível já visto neste projeto (`safe4`,
também 2/16).

**Utilidade — MT-Bench-PT e IFEval-PT** (`instruct2-full` vs `safe-v4`, mesmo
protocolo do §7): diferente do `safe4`, aqui aparece um sinal — ainda que não
conclusivo — de custo de utilidade real.

| | instruct2-full | safe-v4 | Δ | (referência: instruct-v2 → safe4) |
|---|---:|---:|---:|---|
| MT-Bench segurança | 1,75 | 9,00 | +7,25 | +7,50 |
| MT-Bench geral (8 cat., excl. segurança) | 2,60 | 2,15 | **-0,46** | -0,25 |
| IFEval-PT prompt-strict | 27,8% ±7,5 | 22,2% ±6,9 | -5,6pp | +2,8pp |
| IFEval-PT instr-loose | 40,0% ±6,9 | 34,0% ±6,7 | -6,0pp | 0pp |

O `safe4` não custou nada em seguir instrução (IFEval estável/melhor); o `v4`
perdeu nos dois eixos de IFEval e teve uma queda maior no MT-Bench geral. Com
N=36 (IFEval) e 6 perguntas/categoria (MT-Bench), cada delta isolado cabe
dentro de ±1 erro-padrão — não é estatisticamente conclusivo sozinho. Mas os
dois testes, independentes, apontam na mesma direção (queda), o que pesa mais
que ruído aleatório coincidente. **Decisão:** mantém o `v4` como escolhido —
o ganho de segurança (75%→93,8%) é grande e bem medido, o possível custo de
utilidade é real o suficiente pra registrar como limitação aberta, não pra
reverter a escolha. Validação com sondas maiores fica como próximo passo
antes de publicar o modelo.

**Integridade do merge** (`dpo/diag_dpo.py`, mesmo teste do §4): base sem
safety recusa 0/4 (obedece, inclusive gerando conteúdo perigoso de verdade
— ex.: passos pra explosivo caseiro); adapter LoRA carregado direto e
modelo mesclado recusam 4/4, com o texto de recusa idêntico entre os dois.
Sem sinal do bug de arredondamento bf16 que motivou essa checagem no DPO.

**Reprodutibilidade desta seção:**
- Seeds nocivos escalados: `dpo/dpo_seeds/safety_prompts_v2.jsonl` (465 =
  50 originais + 16 ambíguos + 399 gerados via API).
- Ajuda escalada: `sft_safety/benign_help_v3.jsonl` (280 = 55 originais +
  90 gerados [10 categorias] + 90 gerados [round 2, 3 categorias-alvo] + 45
  cópias extras dos 5 exemplos-ouro de crise).
- Dataset final de treino: `sft_safety/safety_sft_v4.jsonl` (3.519
  exemplos), gerado por `sft/gen_safety_sft.py --k_refuse 4 --n_general 1700`.
- Modelo: `manaca-1b-instruct-2-safe-v4` (adapter) /
  `manaca-1b-instruct-2-safe-v4-merged` (completo).

## 11. Reprodutibilidade

- Dados nocivos (prompts): `dpo/dpo_seeds/safety_prompts.jsonl` (50).
- Contraste benigno: `sft/safety_seeds/benign_help.jsonl` (50, disjunto da aval.).
- Construtor: `sft/gen_safety_sft.py` · treino: `make sft-safety`.
- Sondas: `bench/safety_pt/` (segurança) e `bench/overrefusal_pt/` (over-refusal).
- Diagnóstico: `dpo/diag_dpo.py` (base vs adapter vs merged).
- DPO (o que não funcionou): `dpo/gen_dpo_pairs.py`, `dpo/README-onpolicy.md`.
