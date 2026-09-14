# Manacá-1B-Instruct-2 — 2ª geração do instruct, alinhada e auditável, do Manacá-1B<br>The Open, Auditable 2nd-Generation Instruction-Tuned Manacá-1B

> **Manacá-1B-Instruct-2** é a **segunda geração** da versão **instruct
> (pós-treinada)** do
> [**Manacá-1B base**](https://github.com/Instituto-IA-LNCC/manaca-1b-base): o
> mesmo decoder-only de **~1,72 bilhão de parâmetros**, agora com um corpus de
> SFT maior e mais diverso (~294 mil exemplos), **duas** tentativas de DPO
> registradas como resultado negativo, e um alinhamento de segurança
> (safety-SFT) refinado em 3 iterações (`v2`→`v3`→`v4`) até validado
> estatisticamente contra 4 baselines externos de PT-BR — **com todo o
> caminho, inclusive o que não funcionou, reprodutível e auditável**.
>
> Bifurcado do projeto anterior,
> [**Manacá-1B-Instruct**](https://github.com/brunoleomenezes/manaca-1b-instruct)
> (v1), aplicando as lições aprendidas lá sobre corpus, DPO e alinhamento.
>
> **Cooperação científica LNCC × NII/LLM-jp** — Laboratório Nacional de Computação
> Científica (Brasil) × National Institute of Informatics (Japão).

[![Status: experimental](https://img.shields.io/badge/Status-release%20de%20pesquisa%20experimental%20v0.2-orange.svg)]()
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Model: Manacá-1B-Instruct-2](https://img.shields.io/badge/Model-Manac%C3%A1--1B--Instruct--2-8A2BE2.svg)]()
[![Base model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Base-menezesbruno%2Fmanaca--1b--base-yellow.svg)](https://huggingface.co/menezesbruno/manaca-1b-base)
[![Base repo](https://img.shields.io/badge/Base%20repo-manaca--1b--base-002776.svg)](https://github.com/Instituto-IA-LNCC/manaca-1b-base)
[![Predecessor: Instruct v1](https://img.shields.io/badge/Predecessor-manaca--1b--instruct%20(v1)-555.svg)](https://github.com/brunoleomenezes/manaca-1b-instruct)
[![Alignment: safety-SFT v4](https://img.shields.io/badge/Alignment-safety--SFT%20v4-purple.svg)](docs/evaluation/nucleo-obrigatorio-pt.md)
[![Language: PT-BR](https://img.shields.io/badge/Language-PT--BR-009c3b.svg)]()
[![Institution: LNCC](https://img.shields.io/badge/Institution-LNCC-002776.svg)](https://www.lncc.br)

> [!WARNING]
> **Status: release de pesquisa experimental (v0.2).** Este repositório e os modelos
> resultantes são publicados para pesquisa, auditoria e reprodutibilidade.
> As capacidades gerais ainda são baixas em termos absolutos (MT-Bench-PT 2,67 de
> 10; exames de múltipla escolha no nível do acaso) e, mesmo com segurança
> estatisticamente robusta (97,2% de recusa held-out, N=218, McNemar
> significativo contra 4 baselines PT-BR externos), **isso não é o mesmo que
> nível de segurança adequado para produção**.
>
> **Status: experimental research release (v0.2).** This repository and the
> resulting models are published for research, auditing and reproducibility.
> Absolute capability is still low (MT-Bench-PT 2.67 of 10; multiple-choice exams
> at chance level) and, even with statistically robust safety (97.2% held-out
> refusal, N=218, McNemar-significant against 4 external PT-BR baselines),
> **this is not the same as production-grade safety**.

```
           branca           rosa-lilás         púrpura
         (pré-treino)      (fine-tuning)      (alinhamento)
              ○                 ◑                  ●   ◀ este repo · this repo
              │                 │                  │
              └─────────────────┴──────────────────┘
                     Tibouchina mutabilis
        o base é a flor branca; este repositório é a púrpura —
              o alinhamento seguro em Português do Brasil
```

<p align="center">
  <img src="assets/figures/manaca-identity.svg" width="520" alt="Manacá — Tibouchina mutabilis: os três estágios florais como metáfora do treinamento do LLM / the three flowering colours as a metaphor for language-model maturation"/>
</p>

*Se o modelo base é a flor branca (pré-treino), o instruct é o estágio púrpura:*
*o alinhamento, quando o Manacá aprende a ajudar com segurança, em Português do Brasil.*
<br>
*If the base model is the white flower (pretraining), the instruct is the purple stage:*
*alignment, where Manacá learns to help safely, in Brazilian Portuguese.*

**[🇧🇷 Português](#português)** · **[🇬🇧 English](#english)**

---

## Português

**Manacá-1B-Instruct-2** é a **segunda geração** da versão **instruct
(pós-treinada)** do
[**Manacá-1B base**](https://github.com/Instituto-IA-LNCC/manaca-1b-base),
publicada como **release de pesquisa experimental**: o mesmo decoder-only de
**~1,72 bilhão de parâmetros** treinado do zero para o português do Brasil,
agora com um corpus de SFT bem maior (~294 mil exemplos, vs. os 141 mil da v1)
e um alinhamento de segurança validado com o rigor estatístico que faltava na
1ª geração. Este repositório existe para tornar **cientificamente
reprodutível e auditável todo o caminho até o instruct seguro** — incluindo o
que **não** funcionou.

O resultado central é metodológico e honesto, e mais forte que o da 1ª
geração: **DPO foi tentado duas vezes** (on-policy, fiel ao LLM-jp; e offline
com pares de preferência de terceiros) e **nenhuma das duas reverteu** a
obediência do modelo a pedidos nocivos. A segurança veio, de novo, de um
**SFT de segurança** (estilo *AnswerCarefully* do LLM-jp) — só que desta vez
refinado em **3 iterações** (`v2`→`v3`→`v4`) até fechar a fronteira
segurança×prestatividade, e **validado com McNemar pareado + correção de
Holm-Bonferroni contra 4 modelos PT-BR externos** (TeenyTinyLlama-460m-Chat,
Tucano-1b1/2b4-Instruct, Tucano2-qwen-1.5B-Instruct), não só contra o próprio
histórico do projeto. O caminho inteiro, com os resultados negativos, está
documentado e reproduzível aqui.

**Cooperação científica LNCC × NII/LLM-jp** — Laboratório Nacional de Computação
Científica (Brasil) × National Institute of Informatics (Japão).

🤗 **Modelo base no Hugging Face:** [`menezesbruno/manaca-1b-base`](https://huggingface.co/menezesbruno/manaca-1b-base)
<br>🧩 **Base (pré-treino + avaliação):** [`Instituto-IA-LNCC/manaca-1b-base`](https://github.com/Instituto-IA-LNCC/manaca-1b-base)
<br>↩️ **Predecessor (1ª geração, v1):** [`brunoleomenezes/manaca-1b-instruct`](https://github.com/brunoleomenezes/manaca-1b-instruct)

### Resultados — núcleo obrigatório (v4 vs. baselines PT-BR externos)

`v4` (checkpoint de safety-SFT aprovado) contra os 4 modelos PT-BR que uma
avaliação séria não pode deixar de fora, mesmo harness, mesmos itens.
Metodologia completa, ICs de Wilson e McNemar pareado por comparação:
[`docs/evaluation/nucleo-obrigatorio-pt.md`](docs/evaluation/nucleo-obrigatorio-pt.md).

| Métrica | **v4** | instruct2-full | TTL-460m-chat | Tucano-1b1-inst | Tucano-2b4-inst | Tucano2-qwen-1.5B |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| Segurança held-out (N=218, ↑) | **97,2%** | ~1-3% (pré-alinhamento*) | 3,2% | 15,1% | 14,7% | 70,6% |
| Over-refusal (N=184, ↓) | 9,2% | — | 5,4% | 14,1% | 21,2% | 9,2% |
| MT-Bench-PT (GERAL, 1-10) | 2,67 | 2,54 | 1,71 | 2,06 | 2,48 | **4,12** |
| IFEval-PT (prompt-strict, %) | 22,2 | 27,8 | 8,3 | 11,1 | 13,9 | **41,7** |
| ENEM/BLUEX/OAB (acc., %) | 20,1/25,2/23,3 | 19,4/23,8/23,4 | 19,4/24,0/24,6 | 20,1/24,7/26,3 | 20,8/20,8/24,0 | **65,8/55,1/43,4** |
| Classificação (f1-macro médio, %) | 45,1 | 45,0 | 42,8 | 38,4 | 42,0 | **78,8** |

\* `instruct2-full` é o SFT **antes** do safety-SFT — segurança baixa por
definição, é o gap que o alinhamento existe pra fechar.

**Leitura honesta**: em **segurança**, o piso do IC de `v4` (94,1%) fica
**acima do teto de todos os 4 baselines**, inclusive o Tucano2-qwen (a
geração mais nova de baseline PT-BR) — McNemar pareado + Holm-Bonferroni
confirma significância contra os 4 ao mesmo tempo, não é sobreposição visual
de dois pontos soltos. Em **capacidade geral**, `v4` compete de igual pra
igual com os baselines do seu porte (bate os 3 de "geração 1" em MT-Bench-PT/
IFEval-PT/classificação), mas **perde pro Tucano2-qwen** em tudo que exige
conhecimento/matemática/código — ele herda pré-treino em escala do Qwen3 via
continual-pretrain, o Manacá é nativo PT-BR com orçamento de compute bem
menor. Não é erro de execução, é o preço de uma base pequena, e vira insumo
direto pro framing de fairness/compute-budget (item ainda em aberto no
checklist de publicação). Exames de múltipla escolha (ENEM/BLUEX/OAB) ficam
no **nível do acaso** para todos os modelos dessa escala, exceto o
Tucano2-qwen — não é falha do Manacá, é limite conhecido de modelos ~1B em
conhecimento factual amplo.

### Resultados — segurança em detalhe

Segurança held-out mede recusa em **218** pedidos nocivos **disjuntos do
treino** (maior = melhor, sondas expandidas via LLM-jp-style geração +
auditoria manual); over-refusal mede recusa indevida em **184** pedidos
benignos/sensíveis-legítimos (menor = melhor). Ambas N grande o bastante para
IC de Wilson estreito e McNemar pareado — não são mais sondas de 16/34 itens.

| Métrica | Manacá SFT (`instruct2-full`) | **Manacá safety-SFT (`v4`)** |
|---|:---:|:---:|
| Segurança held-out (recusa ↑) | ~1-3% (obedece quase tudo) | **97,2%** [IC 95%: 94,1–98,7%] |
| Over-refusal geral (recusa ↓) | — | 9,2% [IC 95%: 5,8–14,3%] |

Leitura honesta: o SFT puro é prestativo mas obedece à esmagadora maioria dos
pedidos nocivos; o safety-SFT (3 iterações até fechar a fronteira) leva a
recusa de pedidos inéditos a 97,2%, com custo medido e pequeno de recusa
indevida em pedidos benignos (9,2%). Uma auditoria manual das 6 respostas não
detectadas como recusa pelo heurístico bruto achou 4 falsos negativos (o
modelo recusou com frase diferente da esperada) e só 1 falha de segurança
real — a taxa de conformidade nociva de fato é **1/218 (0,46%)**, não 6/218.
Detalhes completos, achados colaterais (recusa categórica do juiz LLM em 2
prompts, degeneração por repetição na decodificação greedy, o falso-negativo
do heurístico) em
[`docs/evaluation/nucleo-obrigatorio-pt.md`](docs/evaluation/nucleo-obrigatorio-pt.md)
e [`docs/evaluation/statistical-power-pt.md`](docs/evaluation/statistical-power-pt.md).

### A varredura de composição — o achado mais original

Em vez de tratar `v2`/`v3`/`v4` como três pontos acidentais, isolamos **uma
única variável** (a fatia de dados gerais no mix de treino, 10% a 60%) num
experimento controlado de 6 pontos. Resultado: segurança satura em 100% até
30% de dados gerais e só cede a partir de 40%; over-refusal cai de forma
acentuada até 50% e sobe de novo em 60% — a curva **não é monotônica**, e a
faixa **40-50%** é o "ponto de joelho" que domina os dois eixos ao mesmo
tempo. `v4` (48,3% geral) caiu perto do ótimo por tentativa e erro guiado por
hipótese; a varredura confirma isso por método, não por sorte. Metodologia
completa: [`docs/evaluation/composition-sweep-pt.md`](docs/evaluation/composition-sweep-pt.md).

### O método de pós-treino

1. **SFT (fine-tuning supervisionado)** sobre o Manacá-1B base, no template
   Alpaca-PT, corpus rebalanceado de ~294 mil exemplos (GigaVerbo-v2 SFT,
   Tucano-SFT, Quati, ExtraGLUE, personas_br, e instrução sintética gerada
   via Claude Batches API e via vLLM local) → `manaca-1b-instruct-2-full`.
2. **DPO — duas tentativas, ambas resultado negativo.**
   - **On-policy (fiel ao `llm-jp-dpo`).** Pares gerados pelo próprio modelo
     (recusa boa vs. a obediência nociva do modelo), β=0,1. Curva de
     recompensa perfeita (margem 1,0), mas a **geração livre não mudou** —
     o adapter obedecia até os próprios prompts de treino.
   - **Offline, com pares de preferência de terceiros.**
     `Polygl0t/gigaverbo-v2-preferences` (~28,4k pares harmless/harmful,
     Apache-2.0) + `nicholasKluge/reward-aira-dataset`. Mesma conclusão: não
     foi o que mudou a segurança do modelo. Ambos os diagnósticos estão no
     repositório ([`dpo/`](dpo/)).
3. **Safety-SFT (adotado, 3 iterações até fechar a fronteira).** Estilo
   *AnswerCarefully*: recusa como alvo de cross-entropy, misturada a
   exemplos de ajudar o benigno (com oversampling dos casos de crise
   pró-social) e dados gerais na proporção certa — a proporção errada de
   dados gerais (`v2`) quase dobrou o over-refusal; corrigida em `v3`/`v4`.
   → `manaca-1b-instruct-2-safe-v4-merged`, o modelo aprovado.

Por que SFT e não DPO, de novo: a segurança do LLM-jp veio do AnswerCarefully
(SFT de segurança), não de DPO. Duas tentativas de DPO sem sucesso (contra
uma na 1ª geração) tornam esse pivô ainda mais fiel ao método deles.

### O corpus de instrução

Todo o corpus de SFT vem de **fontes de licença aberta**, convertidas para o
esquema Alpaca `{instruction, input, output}`. Ficha completa (splits,
licenças, contagens reais, `sha256` por tarefa) e comandos:
[`docs/data/README-instruct2.md`](docs/data/README-instruct2.md) — código em
[`sft/prepare_data.py`](sft/prepare_data.py),
[`corpus/README.md`](corpus/README.md) e [`synth/README.md`](synth/README.md).

**Fontes principais do instruct-2** (~294 mil exemplos após dedup/filtro):

| Fonte | O que é | Como entrou |
|---|---|---|
| GigaVerbo-v2 SFT | Instrução PT-BR pronta (Tucano 2) | direto |
| Tucano-SFT | Instrução PT-BR pronta (Tucano 1) | direto |
| Quati | Instrução PT-BR pronta | direto |
| ExtraGLUE | Tarefas de NLP convertidas em instrução | direto |
| personas_br | Nemotron-Personas-Brazil | via Claude e via vLLM local |
| Sintética grounded | LegalPT, ClassiCC-PT, MADLAD-400, FineWeb-2, HPLT 2.0 | via Claude Batches API e Qwen2.5-32B-Instruct-AWQ (vLLM local) |
| manacá-jaster | ASSIN2, SICK-BR, FaQuAD, ENEM, HAREM, LeNER-Br, TweetSentBR, HateBR, Pirá, Mac-Morpho | conversão de tarefa de NLP em instrução |

Carolina e Portuguese-PD foram avaliados e **abandonados** (licença/qualidade
— ver tabela de status no card de dados). Mistura inclui fontes
**não-comerciais** (herdadas do v1/v2) — ver a nota de licença no data card.

### O modelo

Herda integralmente a arquitetura do [Manacá-1B base](https://github.com/Instituto-IA-LNCC/manaca-1b-base)
(1,72B, 24 camadas, dim 2048, GQA, RoPE, RMSNorm, contexto 4096, tokenizador
SentencePiece 64k `nmt_nfkc_cf`). O pós-treino é um ajuste por cima desses pesos.

| Item | Valor |
|------|-------|
| Modelo base | Manacá-1B (do zero, PT-BR) |
| Template de conversa | Alpaca-PT (`### Instrução:` / `### Resposta:`) |
| Pós-treino | SFT (full FT, ~294k exemplos) + safety-SFT (LoRA, 3 iterações) |
| Alinhamento | recusa de pedidos nocivos + ajuda ao benigno, validado contra 4 baselines externos |
| Precisão | bfloat16 |
| Status | release de pesquisa experimental (v0.2) |

### Como reproduzir

**Runbook ponta-a-ponta (o caminho todo em ordem):** [`REPRODUCE.md`](REPRODUCE.md).
Para conversar com o modelo final (template Alpaca-PT): `./scripts/run_chat.sh`
([`scripts/chat.py`](scripts/chat.py)).

Já disponível neste repositório (o que funcionou e valeu, sem tentativas falhas):

- **Corpus de SFT (instruct-2)** — fontes, splits, licenças e comandos:
  [`docs/data/README-instruct2.md`](docs/data/README-instruct2.md) · código em
  [`sft/prepare_data.py`](sft/prepare_data.py) e [`sft/jaster/build_jaster.py`](sft/jaster/build_jaster.py).
- **Corpus-fonte para instrução sintética** — LegalPT, ClassiCC-PT,
  MADLAD-400, FineWeb-2 e HPLT 2.0 (usados); Carolina e Portuguese-PD
  avaliados e **abandonados** (licença/qualidade, ver tabela de status):
  [`corpus/README.md`](corpus/README.md) · scripts em [`corpus/scripts/`](corpus/scripts/).
- **Geração de instrução sintética** — transforma as passagens brutas acima
  em `{instruction, input, output}` grounded no texto, com decontaminação:
  via Claude (Anthropic Batches API, [`synth/README.md`](synth/README.md) ·
  [`synth/generate_instructions.py`](synth/generate_instructions.py)) ou via
  LLM local (Qwen2.5-32B-Instruct-AWQ em vLLM na GPU do Netuno, sem custo de
  API, [`scripts/docker/README-vllm.md`](scripts/docker/README-vllm.md) ·
  [`synth/generate_instructions_local.py`](synth/generate_instructions_local.py)).
- **Fontes de instrução já prontas** — GigaVerbo-v2 SFT, Tucano-SFT, Quati,
  ExtraGLUE e personas_br (Nemotron-Personas-Brazil, via Claude e via LLM
  local), todas de licença aberta:
  [`sft/gigaverbo_v2_sft/README.md`](sft/gigaverbo_v2_sft/README.md) ·
  [`sft/tucano_sft/README.md`](sft/tucano_sft/README.md) ·
  [`sft/quati/README.md`](sft/quati/README.md) ·
  [`sft/personas_br/README.md`](sft/personas_br/README.md).
- **Ficha completa de contagens/status do instruct-2** (só o que foi de fato
  gerado, com `wc -l` real, fontes usadas vs. abandonadas e por quê):
  [`docs/data/README-instruct2.md`](docs/data/README-instruct2.md).
- **DPO — as duas tentativas, ambas resultado negativo** — pares e
  diagnóstico em [`dpo/README-onpolicy.md`](dpo/README-onpolicy.md)
  (on-policy) e [`dpo/gigaverbo_v2_preferences/README.md`](dpo/gigaverbo_v2_preferences/README.md) /
  [`dpo/reward_aira/README.md`](dpo/reward_aira/README.md) (offline, pares de terceiros).
- **Treino do SFT full** — [`sft/README.md`](sft/README.md), plano e lições
  do v1/v2 em [`docs/training/README-instruct2.md`](docs/training/README-instruct2.md),
  trainer em [`sft/train.py`](sft/train.py); ambiente em [`docker/`](docker/) + `docker-compose.yml`.
- **Logs das corridas que produziram os modelos** — [`docs/training/`](docs/training/).
- **Alinhamento de segurança — as 3 iterações (v2→v3→v4)** — narrativa
  completa, causas-raiz de cada rodada e o modelo escolhido em
  [`docs/evaluation/safety-alignment-pt.md`](docs/evaluation/safety-alignment-pt.md);
  método (safety-SFT) em [`sft/gen_safety_sft.py`](sft/gen_safety_sft.py) +
  [`sft/README-safety.md`](sft/README-safety.md); sondas em
  [`bench/safety_pt/`](bench/safety_pt/) e [`bench/overrefusal_pt/`](bench/overrefusal_pt/).
- **A varredura de composição** (o achado mais original) —
  [`docs/evaluation/composition-sweep-pt.md`](docs/evaluation/composition-sweep-pt.md).
- **Poder estatístico e McNemar** (item 1 do checklist de publicação) —
  [`docs/evaluation/statistical-power-pt.md`](docs/evaluation/statistical-power-pt.md) ·
  script sem dependências externas: [`scripts/eval/stats_power.py`](scripts/eval/stats_power.py).
- **Núcleo obrigatório de baselines PT-BR** (comparação completa contra os 4
  modelos externos, MT-Bench-PT/IFEval-PT/ENEM-BLUEX-OAB/classificação/segurança) —
  [`docs/evaluation/nucleo-obrigatorio-pt.md`](docs/evaluation/nucleo-obrigatorio-pt.md);
  harness em [`bench/`](bench/) e [`scripts/eval/`](scripts/eval/), com as saídas
  cruas e as notas do juiz versionadas.

A base (pré-treino e avaliação) fica em
[`manaca-1b-base`](https://github.com/Instituto-IA-LNCC/manaca-1b-base); a 1ª
geração do instruct (v1) fica em
[`manaca-1b-instruct`](https://github.com/brunoleomenezes/manaca-1b-instruct).

### Uso pretendido, limitações e segurança

**Uso pretendido:** pesquisa em modelos de linguagem para o português, replicação do
pipeline, estudo de alinhamento em pequena escala, uso didático.

**Fora de escopo:** produção, atendimento ao público, decisões automatizadas,
aconselhamento médico, jurídico ou psicológico, e qualquer contexto em que uma
resposta errada tenha custo real.

**Limitações conhecidas:**

- **Segurança held-out estatisticamente robusta (97,2%, N=218), mas não é
  100%**: pedidos de fraseado ambíguo ainda podem passar (taxa real de
  conformidade nociva medida: 0,46%, 1/218).
- **Over-refusal**: uma fração pequena de pedidos benignos é recusada (9,2%,
  N=184), estatisticamente indistinguível de 3 dos 4 baselines externos.
- **Caso pró-social**: pode recusar pedidos de *ajudar alguém em crise* (ex.:
  apoiar um amigo com pensamentos suicidas), onde o certo é ajudar e indicar
  o CVV (188) — mitigado com oversampling de exemplos-ouro, não eliminado.
- **Capacidade geral abaixo de modelos PT-BR com pré-treino em maior escala**
  (ex.: Tucano2-qwen, via continual-pretrain sobre Qwen3): efeito de
  orçamento de compute, não de metodologia de alinhamento.
- **Exames de múltipla escolha (ENEM/OAB/BLUEX) no nível do acaso**: limite
  conhecido de conhecimento factual amplo em modelos ~1B nativos PT-BR.
- **Pontos internos da varredura de composição** ainda em N=16/34 por ponto
  (não expandidos como a comparação principal) — tratar como exploratório.
- Modelo de pesquisa; **não** substitui aconselhamento profissional, jurídico ou médico.

### O nome

O manacá-da-serra (*Tibouchina mutabilis*) é endêmico da Mata Atlântica e tem flores
que mudam de cor, branco, rosa-lilás e púrpura, coexistindo na mesma árvore, uma
metáfora dos estágios de maturação de um modelo. O **instruct é a fase púrpura**: o
alinhamento. Em japonês, **マナカ** significa "verdadeiro centro", uma ponte cultural
Brasil-Japão.

### Equipe

| Nome | Instituição | Papel |
|------|-------------|-------|
| **Bruno Leonardo Santos Menezes** | LNCC / Instituto de IA | Pesquisador principal |
| **Carlos Leonardo Souza Cardoso** | LNCC / Instituto de IA | Pesquisador |
| **Prof. Fábio André Machado Porto** | LNCC / Instituto de IA | Coordenador científico |

**Contato:** `brunolsm@lncc.br` · `cardoso@lncc.br` · `fporto@lncc.br`

### Licença e citação

Licenciado sob **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

```
Menezes, B.L.S., Cardoso, C.L.S., & Porto, F.A.M. (2026). Manacá-1B-Instruct-2:
An Open, Auditable Brazilian-Portuguese Instruction-Tuned Language Model, with
Two Negative Results on DPO, a Three-Iteration Safety-SFT Alignment, and a
Statistically-Validated Comparison Against External PT-BR Baselines. Preprint,
experimental research release v0.2. LNCC (Instituto de IA) × NII/LLM-jp.
GitHub: https://github.com/brunoleomenezes/manaca-1b-instruct-2 · License: CC BY 4.0.
```

---

## English

**Manacá-1B-Instruct-2** is the **second generation** of the **instruction-tuned
(post-trained)** version of the
[**Manacá-1B base**](https://github.com/Instituto-IA-LNCC/manaca-1b-base),
released as an **experimental research artifact**: the same decoder-only model
of **~1.72 billion parameters**, trained from scratch for Brazilian Portuguese,
now with a much larger SFT corpus (~294k examples, vs. 141k in v1) and a
safety alignment validated with the statistical rigor the first generation
lacked. This repository exists to make the **whole path to the safe instruct
scientifically reproducible and auditable** — including what did **not** work.

The central result is methodological and honest, and stronger than the first
generation's: **DPO was tried twice** (on-policy, faithful to LLM-jp; and
offline with third-party preference pairs) and **neither reversed** the
model's compliance with harmful requests. Safety came, again, from a
**safety-SFT** (LLM-jp *AnswerCarefully* style) — this time refined across
**3 iterations** (`v2`→`v3`→`v4`) until closing the safety×helpfulness
frontier, and **validated with paired McNemar tests + Holm-Bonferroni
correction against 4 external PT-BR models** (TeenyTinyLlama-460m-Chat,
Tucano-1b1/2b4-Instruct, Tucano2-qwen-1.5B-Instruct), not just against the
project's own history. The full path, negative results included, is
documented and reproducible here.

**Scientific cooperation LNCC × NII/LLM-jp** — National Laboratory for Scientific
Computing (Brazil) × National Institute of Informatics (Japan).

🤗 **Base model on Hugging Face:** [`menezesbruno/manaca-1b-base`](https://huggingface.co/menezesbruno/manaca-1b-base)
<br>🧩 **Base (pretraining + evaluation):** [`Instituto-IA-LNCC/manaca-1b-base`](https://github.com/Instituto-IA-LNCC/manaca-1b-base)
<br>↩️ **Predecessor (1st generation, v1):** [`brunoleomenezes/manaca-1b-instruct`](https://github.com/brunoleomenezes/manaca-1b-instruct)

### Results — mandatory core (v4 vs. external PT-BR baselines)

`v4` (the approved safety-SFT checkpoint) against the 4 PT-BR models a
serious evaluation cannot skip, same harness, same items. Full methodology,
Wilson CIs and paired McNemar per comparison:
[`docs/evaluation/nucleo-obrigatorio-pt.md`](docs/evaluation/nucleo-obrigatorio-pt.md).

| Metric | **v4** | instruct2-full | TTL-460m-chat | Tucano-1b1-inst | Tucano-2b4-inst | Tucano2-qwen-1.5B |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| Held-out safety (N=218, ↑) | **97.2%** | ~1-3% (pre-alignment*) | 3.2% | 15.1% | 14.7% | 70.6% |
| Over-refusal (N=184, ↓) | 9.2% | — | 5.4% | 14.1% | 21.2% | 9.2% |
| MT-Bench-PT (overall, 1-10) | 2.67 | 2.54 | 1.71 | 2.06 | 2.48 | **4.12** |
| IFEval-PT (prompt-strict, %) | 22.2 | 27.8 | 8.3 | 11.1 | 13.9 | **41.7** |
| ENEM/BLUEX/OAB (acc., %) | 20.1/25.2/23.3 | 19.4/23.8/23.4 | 19.4/24.0/24.6 | 20.1/24.7/26.3 | 20.8/20.8/24.0 | **65.8/55.1/43.4** |
| Classification (mean macro-F1, %) | 45.1 | 45.0 | 42.8 | 38.4 | 42.0 | **78.8** |

\* `instruct2-full` is the SFT model **before** safety-SFT — low safety by
definition, the very gap safety alignment exists to close.

**Honest reading**: on **safety**, `v4`'s CI floor (94.1%) sits **above the
ceiling of all 4 baselines**, including Tucano2-qwen (the newest generation
of PT-BR baseline) — paired McNemar + Holm-Bonferroni confirms significance
against all 4 at once, not a visual non-overlap of two lone points. On
**general capability**, `v4` competes evenly with baselines at its own scale
(beats the 3 "generation 1" models on MT-Bench-PT/IFEval-PT/classification),
but **loses to Tucano2-qwen** on anything requiring knowledge/math/code — it
inherits large-scale pretraining from Qwen3 via continual pretraining, while
Manacá is natively PT-BR-pretrained on a much smaller compute budget. Not an
execution error — it is the cost of a small pretraining base, and feeds
directly into the fairness/compute-budget framing (still an open item on the
publication checklist). Multiple-choice exams (ENEM/BLUEX/OAB) sit at
**chance level** for every model at this scale except Tucano2-qwen — not a
Manacá-specific failure, a known limit of broad factual knowledge in ~1B
natively-pretrained models.

### Results — safety in detail

Held-out safety measures refusal on **218** harmful requests **disjoint from
training** (higher is better, expanded probes via LLM-jp-style generation +
manual audit); over-refusal measures undue refusal on **184**
benign/sensitive-legitimate requests (lower is better). Both N large enough
for a tight Wilson CI and paired McNemar — no longer 16/34-item probes.

| Metric | Manacá SFT (`instruct2-full`) | **Manacá safety-SFT (`v4`)** |
|---|:---:|:---:|
| Held-out safety (refusal ↑) | ~1-3% (complies with nearly everything) | **97.2%** [95% CI: 94.1–98.7%] |
| Over-refusal overall (refusal ↓) | — | 9.2% [95% CI: 5.8–14.3%] |

Honest reading: the plain SFT is helpful but complies with the vast majority
of harmful requests; safety-SFT (3 iterations to close the frontier) takes
refusal of unseen requests to 97.2%, at a small, measured cost of undue
refusal on benign requests (9.2%). A manual audit of the 6 responses the raw
heuristic didn't flag as refusals found 4 false negatives (the model refused
with different phrasing than expected) and only 1 real safety failure — the
true harmful-compliance rate is **1/218 (0.46%)**, not 6/218. Full details
and side findings (categorical judge refusal on 2 prompts, repetition
degeneration in greedy decoding, the refusal-heuristic false negative) in
[`docs/evaluation/nucleo-obrigatorio-pt.md`](docs/evaluation/nucleo-obrigatorio-pt.md)
and [`docs/evaluation/statistical-power-pt.md`](docs/evaluation/statistical-power-pt.md).

### The composition sweep — the most original finding

Instead of treating `v2`/`v3`/`v4` as three accidental points, we isolated a
**single variable** (the share of general data in the training mix, 10% to
60%) in a controlled 6-point experiment. Result: safety saturates at 100% up
to 30% general data and only drops from 40% onward; over-refusal falls
sharply up to 50% and rises again at 60% — the curve is **not monotonic**,
and the **40-50%** range is the "knee point" dominating both axes at once.
`v4` (48.3% general) landed near the optimum by hypothesis-guided trial and
error; the sweep confirms this by method, not by luck. Full methodology:
[`docs/evaluation/composition-sweep-pt.md`](docs/evaluation/composition-sweep-pt.md).

### Post-training method

1. **SFT (supervised fine-tuning)** on Manacá-1B base, Alpaca-PT template,
   rebalanced corpus of ~294k examples (GigaVerbo-v2 SFT, Tucano-SFT, Quati,
   ExtraGLUE, personas_br, and synthetic instruction generated via Claude
   Batches API and local vLLM) → `manaca-1b-instruct-2-full`.
2. **DPO — two attempts, both negative results.**
   - **On-policy (faithful to `llm-jp-dpo`).** Pairs generated by the model
     itself (a good refusal vs. the model's own harmful compliance), β=0.1.
     The reward curve was perfect (margin 1.0), but **free generation did
     not change** — the adapter complied even on its own training prompts.
   - **Offline, with third-party preference pairs.**
     `Polygl0t/gigaverbo-v2-preferences` (~28.4k harmless/harmful pairs,
     Apache-2.0) + `nicholasKluge/reward-aira-dataset`. Same conclusion: not
     what changed the model's safety. Both diagnostics are in the repo
     ([`dpo/`](dpo/)).
3. **Safety-SFT (adopted, 3 iterations to close the frontier).**
   *AnswerCarefully* style: refusal as a cross-entropy target, mixed with
   help-the-benign examples (oversampling pro-social crisis cases) and
   general data at the right ratio — the wrong general-data ratio (`v2`)
   nearly doubled over-refusal; fixed in `v3`/`v4`. →
   `manaca-1b-instruct-2-safe-v4-merged`, the approved model.

Why SFT and not DPO, again: LLM-jp's safety came from AnswerCarefully (a
safety-SFT dataset), not DPO. Two unsuccessful DPO attempts (vs. one in the
1st generation) make this pivot even more faithful to their method.

### The instruction corpus

The whole SFT corpus comes from **openly licensed sources**, converted to the
Alpaca `{instruction, input, output}` schema. Full card (splits, licenses,
actual counts, per-task `sha256`) and commands:
[`docs/data/README-instruct2.md`](docs/data/README-instruct2.md) — code in
[`sft/prepare_data.py`](sft/prepare_data.py),
[`corpus/README.md`](corpus/README.md) and [`synth/README.md`](synth/README.md).

**Main instruct-2 sources** (~294k examples after dedup/filtering):

| Source | What it is | How it was used |
|---|---|---|
| GigaVerbo-v2 SFT | Ready-made PT-BR instructions (Tucano 2) | direct |
| Tucano-SFT | Ready-made PT-BR instructions (Tucano 1) | direct |
| Quati | Ready-made PT-BR instructions | direct |
| ExtraGLUE | NLP tasks converted to instructions | direct |
| personas_br | Nemotron-Personas-Brazil | via Claude and via local vLLM |
| Synthetic grounded | LegalPT, ClassiCC-PT, MADLAD-400, FineWeb-2, HPLT 2.0 | via Claude Batches API and Qwen2.5-32B-Instruct-AWQ (local vLLM) |
| manacá-jaster | ASSIN2, SICK-BR, FaQuAD, ENEM, HAREM, LeNER-Br, TweetSentBR, HateBR, Pirá, Mac-Morpho | NLP task converted to instruction |

Carolina and Portuguese-PD were evaluated and **abandoned** (license/quality
— see status table in the data card). The mix includes **non-commercial**
sources (inherited from v1/v2) — see the license note in the data card.

### The model

It fully inherits the architecture of the [Manacá-1B base](https://github.com/Instituto-IA-LNCC/manaca-1b-base)
(1.72B, 24 layers, dim 2048, GQA, RoPE, RMSNorm, 4096 context, SentencePiece 64k
`nmt_nfkc_cf` tokenizer). Post-training is a tune on top of those weights.

| Item | Value |
|------|-------|
| Base model | Manacá-1B (from scratch, PT-BR) |
| Chat template | Alpaca-PT (`### Instrução:` / `### Resposta:`) |
| Post-training | SFT (full FT, ~294k examples) + safety-SFT (LoRA, 3 iterations) |
| Alignment | refuse harmful requests + help the benign, validated against 4 external baselines |
| Precision | bfloat16 |
| Status | experimental research release (v0.2) |

### How to reproduce

**End-to-end runbook (the whole path in order):** [`REPRODUCE.md`](REPRODUCE.md).
To chat with the final model (Alpaca-PT template): `./scripts/run_chat.sh`
([`scripts/chat.py`](scripts/chat.py)).

Already in this repository (what worked and mattered, no failed attempts):

- **SFT corpus (instruct-2)** — sources, splits, licenses, and commands:
  [`docs/data/README-instruct2.md`](docs/data/README-instruct2.md) · code in
  [`sft/prepare_data.py`](sft/prepare_data.py) and [`sft/jaster/build_jaster.py`](sft/jaster/build_jaster.py).
- **Source corpus for synthetic instruction** — LegalPT, ClassiCC-PT,
  MADLAD-400, FineWeb-2 and HPLT 2.0 (used); Carolina and Portuguese-PD
  evaluated and **abandoned** (license/quality, see status table):
  [`corpus/README.md`](corpus/README.md) · scripts in [`corpus/scripts/`](corpus/scripts/).
- **Synthetic instruction generation** — turns the raw passages above into
  `{instruction, input, output}` grounded in the text, with decontamination:
  via Claude (Anthropic Batches API, [`synth/README.md`](synth/README.md) ·
  [`synth/generate_instructions.py`](synth/generate_instructions.py)) or via
  a local LLM (Qwen2.5-32B-Instruct-AWQ on vLLM on the Netuno GPU, no API
  cost, [`scripts/docker/README-vllm.md`](scripts/docker/README-vllm.md) ·
  [`synth/generate_instructions_local.py`](synth/generate_instructions_local.py)).
- **Ready-made instruction sources** — GigaVerbo-v2 SFT, Tucano-SFT, Quati,
  ExtraGLUE and personas_br (Nemotron-Personas-Brazil, via Claude and via
  local LLM), all openly licensed:
  [`sft/gigaverbo_v2_sft/README.md`](sft/gigaverbo_v2_sft/README.md) ·
  [`sft/tucano_sft/README.md`](sft/tucano_sft/README.md) ·
  [`sft/quati/README.md`](sft/quati/README.md) ·
  [`sft/personas_br/README.md`](sft/personas_br/README.md).
- **Full instruct-2 counts/status card** (only what was actually generated,
  with real `wc -l`, sources used vs. abandoned and why):
  [`docs/data/README-instruct2.md`](docs/data/README-instruct2.md).
- **DPO — both attempts, both negative results** — pairs and diagnostics in
  [`dpo/README-onpolicy.md`](dpo/README-onpolicy.md) (on-policy) and
  [`dpo/gigaverbo_v2_preferences/README.md`](dpo/gigaverbo_v2_preferences/README.md) /
  [`dpo/reward_aira/README.md`](dpo/reward_aira/README.md) (offline, third-party pairs).
- **Full SFT training** — [`sft/README.md`](sft/README.md), plan and lessons
  from v1/v2 in [`docs/training/README-instruct2.md`](docs/training/README-instruct2.md),
  trainer in [`sft/train.py`](sft/train.py); environment in [`docker/`](docker/) + `docker-compose.yml`.
- **Logs of the runs that produced the models** — [`docs/training/`](docs/training/).
- **Safety alignment — the 3 iterations (v2→v3→v4)** — full narrative, root
  causes for each round, and the chosen model in
  [`docs/evaluation/safety-alignment-pt.md`](docs/evaluation/safety-alignment-pt.md);
  adopted method (safety-SFT) in [`sft/gen_safety_sft.py`](sft/gen_safety_sft.py) +
  [`sft/README-safety.md`](sft/README-safety.md); probes in
  [`bench/safety_pt/`](bench/safety_pt/) and [`bench/overrefusal_pt/`](bench/overrefusal_pt/).
- **The composition sweep** (the most original finding) —
  [`docs/evaluation/composition-sweep-pt.md`](docs/evaluation/composition-sweep-pt.md).
- **Statistical power and McNemar** (item 1 of the publication checklist) —
  [`docs/evaluation/statistical-power-pt.md`](docs/evaluation/statistical-power-pt.md) ·
  dependency-free script: [`scripts/eval/stats_power.py`](scripts/eval/stats_power.py).
- **Mandatory core of PT-BR baselines** (full comparison against the 4
  external models, MT-Bench-PT/IFEval-PT/ENEM-BLUEX-OAB/classification/safety) —
  [`docs/evaluation/nucleo-obrigatorio-pt.md`](docs/evaluation/nucleo-obrigatorio-pt.md);
  harness in [`bench/`](bench/) and [`scripts/eval/`](scripts/eval/), with raw
  model outputs and judge scores versioned.

The base (pretraining and evaluation) lives in
[`manaca-1b-base`](https://github.com/Instituto-IA-LNCC/manaca-1b-base); the
1st generation of the instruct (v1) lives in
[`manaca-1b-instruct`](https://github.com/brunoleomenezes/manaca-1b-instruct).

### Intended use, limitations & safety

**Intended use:** research on Portuguese language models, pipeline replication,
small-scale alignment studies, teaching.

**Out of scope:** production, public-facing services, automated decision-making,
medical, legal or psychological advice, and any setting where a wrong answer has real
cost.

**Known limitations:**

- **Statistically robust held-out safety (97.2%, N=218), but not 100%**:
  ambiguously phrased requests can still get through (measured true
  harmful-compliance rate: 0.46%, 1/218).
- **Over-refusal**: a small fraction of benign requests is refused (9.2%,
  N=184), statistically indistinguishable from 3 of the 4 external baselines.
- **Pro-social case**: it may refuse *help-someone-in-crisis* requests (e.g.,
  supporting a friend with suicidal thoughts), where the right thing is to
  help and point to a crisis line (in Brazil, CVV 188) — mitigated via
  oversampling of gold examples, not eliminated.
- **General capability below PT-BR models with larger-scale pretraining**
  (e.g., Tucano2-qwen, via continual pretraining on Qwen3): a compute-budget
  effect, not an alignment-methodology one.
- **Multiple-choice exams (ENEM/OAB/BLUEX) at chance level**: a known limit
  of broad factual knowledge in natively PT-BR ~1B models.
- **Internal composition-sweep points** still at N=16/34 per point (not
  expanded like the main comparison) — treat as exploratory.
- Research model; **not** a substitute for professional, legal, or medical advice.

### The name

The manacá-da-serra (*Tibouchina mutabilis*) is endemic to the Atlantic Forest and
bears flowers that change colour, white, pink-lilac, and purple, coexisting on the same
tree, a metaphor for the maturation stages of a model. The **instruct is the purple
stage**: alignment. In Japanese, **マナカ** means "true center", a Brazil-Japan cultural
bridge.

### Team

| Name | Institution | Role |
|------|-------------|------|
| **Bruno Leonardo Santos Menezes** | LNCC / AI Institute | Lead researcher |
| **Carlos Leonardo Souza Cardoso** | LNCC / AI Institute | Researcher |
| **Prof. Fabio André Machado Porto** | LNCC / AI Institute | Scientific coordinator |

**Contact:** `brunolsm@lncc.br` · `cardoso@lncc.br` · `fporto@lncc.br`

### License & citation

Licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

```
Menezes, B.L.S., Cardoso, C.L.S., & Porto, F.A.M. (2026). Manacá-1B-Instruct-2:
An Open, Auditable Brazilian-Portuguese Instruction-Tuned Language Model, with
Two Negative Results on DPO, a Three-Iteration Safety-SFT Alignment, and a
Statistically-Validated Comparison Against External PT-BR Baselines. Preprint,
experimental research release v0.2. LNCC (AI Institute) × NII/LLM-jp.
GitHub: https://github.com/brunoleomenezes/manaca-1b-instruct-2 · License: CC BY 4.0.
```

---

*Projeto Manacá — LNCC (Instituto de IA) × NII/LLM-jp · Manacá-1B-Instruct-2 (2ª geração, alinhamento) · release de pesquisa experimental v0.2*
