# Núcleo obrigatório de baselines (checklist de publicação)

Registro da comparação com o "núcleo obrigatório" de baselines PT-BR pedido pelo
checklist de revisão pré-submissão: sem eles, "o revisor pergunta" por que o
Manacá-1B não foi comparado com o que já existe na mesma faixa de tamanho.

## 1. Os quatro modelos

| Modelo | HF ID | Parâmetros | Papel na comparação |
|---|---|---:|---|
| TeenyTinyLlama-460m-Chat | `nicholasKluge/TeenyTinyLlama-460m-Chat` | 460M | menor, já usado no MT-Bench-PT/IFEval-PT do projeto |
| Tucano-1b1-Instruct | `TucanoBR/Tucano-1b1-Instruct` | 1,1B | par de tamanho mais próximo do Manacá (1B), geração 1 |
| Tucano-2b4-Instruct | `TucanoBR/Tucano-2b4-Instruct` | 2,4B | maior, já usado no MT-Bench-PT/IFEval-PT do projeto |
| **Tucano2-qwen-1.5B-Instruct** | `Polygl0t/Tucano2-qwen-1.5B-Instruct` | 1,5B | **geração atual** do baseline PT-BR (checklist: "Tucano 2, 0,5B a 3,5B") |

Os três primeiros já tinham ID confirmado nos exemplos do próprio
`bench/ifeval_pt/README.md`. O quarto ("Tucano 2") não tinha ID conhecido no
repo — o checklist só descrevia a faixa de tamanho ("0,5B a 3,5B"), sem apontar
o repositório exato, e adivinhar um HF ID é exatamente o tipo de erro que
desperdiça GPU/disco baixando o modelo errado. Verificação feita via busca web
antes de assumir qualquer ID:

- O repositório original `Nkluge-correa/Tucano` (autor do Tucano 1) está
  **arquivado** desde 24/02/2026, com a nota explícita "TO SEE THE EVOLUTION OF
  TUCANO, CHECK THE POLYGL0T PROJECT: https://github.com/Polygl0t" — confirma
  que `Polygl0t` é o sucessor oficial, não um projeto de terceiros usando o nome.
- O paper associado ("Tucano 2 Cool: Better Open Source LLMs for Portuguese",
  arXiv:2603.03543, 03/2026) descreve uma suíte de **0,5–3,7B** parâmetros
  (Base/Instruct/Think), continuada a partir de checkpoints Qwen3 com
  transplante de embeddings — bate com a faixa "0,5B a 3,5B" do checklist
  (a suíte real vai até 3,7B, não 3,5B; o checklist provavelmente arredondou).
- IDs confirmados por indexação (páginas do HF retornadas literalmente pela
  busca, não inferidas): `Polygl0t/Tucano2-qwen-0.5B-{Base,Instruct}`,
  `Polygl0t/Tucano2-qwen-1.5B-{Base,Instruct}`,
  `Polygl0t/Tucano2-qwen-3.7B-{Base,Instruct}`.
- Esta sandbox não tem egress liberado para `huggingface.co`/`arxiv.org`
  (proxy bloqueia), então os IDs não foram baixados/testados aqui — **a
  primeira coisa a fazer na HPC é confirmar que os `HFID` abaixo realmente
  resolvem** (mesmo padrão de verificação já sugerido no README do MT-Bench-PT,
  seção "Confirme os IDs que existem de fato").
- Escolhido o **1,5B** (não o 0,5B nem o 3,7B) porque é o par de tamanho mais
  próximo do Manacá-1B e do `Tucano-1b1-Instruct` original — permite a
  pergunta "a nova geração do baseline supera a antiga NO MESMO TAMANHO?" além
  de "o Manacá compete com o estado da arte PT-BR no seu próprio tamanho?".
  Rodar 0,5B e/ou 3,7B depois é trivial (mesmo comando, troca só o HFID) se o
  revisor pedir a faixa completa.

## 2. Segundo detalhe de justiça: tokenizadores diferentes, `max_new_tokens` fixo

Manacá usa um SentencePiece **64k** próprio (`nmt_nfkc_cf`). Os quatro baselines
também têm tokenizador nativo em PT-BR, mas com vocabulários **menores**:

| Modelo | Tokenizer | Vocab |
|---|---|---:|
| Manacá-1B | SentencePiece, `nmt_nfkc_cf` | 64.000 |
| TeenyTinyLlama-460m | SentencePiece (treinado do zero) | 32.000 |
| Tucano-1b1/2b4-Instruct (geração 1) | SentencePiece unigram | 32.000 |
| Tucano2-qwen | BPE + normalização SentencePiece | 49.152 |

Vocabulário maior costuma comprimir PT-BR em **menos tokens por palavra**. Como
`run_mtbench_pt.sh`/`run_overrefusal_pt.sh` usam `--max_new_tokens` **fixo**
(768/512) pra todos os modelos, isso não é um teto de tamanho igual: um baseline
com vocabulário menor pode precisar de mais tokens pra dizer a mesma coisa e ser
cortado no meio antes do Manacá, inflando a vantagem do Manacá em MT-Bench-PT
(qualidade/completude) e IFEval-PT (restrições de tamanho) por um motivo que não
tem nada a ver com a qualidade do modelo.

**Instrumentado, não só documentado**: `gen_answers.py` e
`bench/overrefusal_pt/run.py` agora gravam `n_tokens_gerados` e `truncado`
(bateu no teto sem emitir EOS) em cada resposta, e imprimem um aviso agregado
por modelo ao final da rodada. Antes de comparar os números:

1. Rodar a geração normalmente.
2. Checar o aviso `[AVISO] N/total bateram no teto` no stdout de cada modelo
   (ou `grep -c '"truncado": true' bench/*/answers/<label>.jsonl`).
3. Se **qualquer** baseline truncar de forma não-desprezível (e o Manacá não, ou
   vice-versa), re-rodar aquele modelo com `--max_new_tokens` maior antes de
   confiar na comparação — não misturar respostas cortadas com completas na
   mesma tabela. Registrar no relatório final qual `--max_new_tokens` foi usado
   por modelo, já que pode deixar de ser o mesmo número pra todos.

## 3. Correção de justiça no harness (bug encontrado antes de rodar)

O MT-Bench-PT/IFEval-PT (`bench/mtbench_pt/gen_answers.py`) já interroga cada
baseline com o **seu próprio `chat_template`** (`--prompt_style auto`) — mas
`bench/overrefusal_pt/run.py` (reusado para a sonda de segurança held-out)
**não tinha essa opção**: ele sempre forçava o template Alpaca-PT do
SFT/DPO do Manacá em qualquer `--model`, inclusive um baseline externo. Isso
nunca importou antes porque só comparávamos checkpoints do próprio Manacá
(todos treinados nesse formato) — mas rodar Tucano/TTL/Tucano2 (treinados com
`chat_template` próprio) através do Alpaca-PT distorceria a taxa de recusa
medida (o modelo não reconhece o formato como uma instrução real), exatamente
o tipo de comparação injusta que o núcleo obrigatório existe pra evitar.

Corrigido (`bench/overrefusal_pt/run.py` + `scripts/eval/run_overrefusal_pt.sh`):
- `run.py` ganhou `--prompt_style {auto,chat,alpaca}` (default `auto`), mesma
  semântica de `gen_answers.py`.
- `run_overrefusal_pt.sh` ganhou o mesmo modo `HFID=`/`HF_CACHE=` do
  `run_mtbench_pt.sh` (antes só aceitava `MODEL=` local, offline).

Sem essa correção, os números de segurança/over-refusal dos baselines não
seriam comparáveis com os do Manacá com o rigor que o resto do checklist exige.

## 4. Comandos (rodar na HPC, GPU, dentro do docker)

### 4.1 MT-Bench-PT — gerar respostas

```bash
cd $HOME/manaca-1b-instruct-2

HFID=nicholasKluge/TeenyTinyLlama-460m-Chat LABEL=ttl-460m-chat \
  ./scripts/eval/run_mtbench_pt.sh
HFID=TucanoBR/Tucano-1b1-Instruct LABEL=tucano-1b1-instruct \
  ./scripts/eval/run_mtbench_pt.sh
HFID=TucanoBR/Tucano-2b4-Instruct LABEL=tucano-2b4-instruct \
  ./scripts/eval/run_mtbench_pt.sh
HFID=Polygl0t/Tucano2-qwen-1.5B-Instruct LABEL=tucano2-qwen-1b5-instruct \
  ./scripts/eval/run_mtbench_pt.sh
```

### 4.2 MT-Bench-PT — julgar e agregar

```bash
export ANTHROPIC_API_KEY=$(grep '^ANTHROPIC_API_KEY=' .env | cut -d= -f2-)

for L in ttl-460m-chat tucano-1b1-instruct tucano-2b4-instruct tucano2-qwen-1b5-instruct; do
  python3 bench/mtbench_pt/judge.py \
    --answers bench/mtbench_pt/answers/$L.jsonl \
    --out     bench/mtbench_pt/judged/$L.jsonl
done

python3 bench/mtbench_pt/report.py \
  bench/mtbench_pt/judged/instruct2-full.jsonl \
  bench/mtbench_pt/judged/ttl-460m-chat.jsonl \
  bench/mtbench_pt/judged/tucano-1b1-instruct.jsonl \
  bench/mtbench_pt/judged/tucano-2b4-instruct.jsonl \
  bench/mtbench_pt/judged/tucano2-qwen-1b5-instruct.jsonl \
  --out bench/mtbench_pt/mtbench-pt-nucleo
```

Mesmo juiz (`claude-opus-5`, Anthropic) já usado para julgar o `v4`/`instruct2-full`
— item 2 do checklist (segundo juiz de outra família) continua bloqueado por
custo de API, registrado como limitação, não simulado.

### 4.3 IFEval-PT — gerar e pontuar (sem juiz, de graça)

```bash
Q=bench/ifeval_pt/prompts.jsonl; A=bench/ifeval_pt/answers

for HFID_LABEL in \
  "nicholasKluge/TeenyTinyLlama-460m-Chat ttl-460m-chat" \
  "TucanoBR/Tucano-1b1-Instruct tucano-1b1-instruct" \
  "TucanoBR/Tucano-2b4-Instruct tucano-2b4-instruct" \
  "Polygl0t/Tucano2-qwen-1.5B-Instruct tucano2-qwen-1b5-instruct"; do
  set -- $HFID_LABEL
  QUESTIONS=$Q ANS_DIR=$A HFID="$1" LABEL="$2" ./scripts/eval/run_mtbench_pt.sh
done

python3 bench/ifeval_pt/score.py \
  bench/ifeval_pt/answers/instruct2-full.jsonl \
  bench/ifeval_pt/answers/ttl-460m-chat.jsonl \
  bench/ifeval_pt/answers/tucano-1b1-instruct.jsonl \
  bench/ifeval_pt/answers/tucano-2b4-instruct.jsonl \
  bench/ifeval_pt/answers/tucano2-qwen-1b5-instruct.jsonl \
  --out bench/ifeval_pt/ifeval-pt-nucleo
```

### 4.4 Segurança held-out (sondas expandidas, N=218) e over-refusal (N=184)

Usa os conjuntos expandidos (`heldout_v2.jsonl`/`prompts_v2.jsonl`), pelo mesmo
motivo do item 1 do checklist: N=16/34 não dá IC estreito o bastante pra
comparar 4 baselines a mais contra o `v4`.

```bash
for HFID_LABEL in \
  "nicholasKluge/TeenyTinyLlama-460m-Chat ttl-460m-chat" \
  "TucanoBR/Tucano-1b1-Instruct tucano-1b1-instruct" \
  "TucanoBR/Tucano-2b4-Instruct tucano-2b4-instruct" \
  "Polygl0t/Tucano2-qwen-1.5B-Instruct tucano2-qwen-1b5-instruct"; do
  set -- $HFID_LABEL
  HFID="$1" LABEL="$2" ./scripts/eval/run_overrefusal_pt.sh \
    --prompts bench/safety_pt/heldout_v2.jsonl --out_dir bench/safety_pt/answers
  HFID="$1" LABEL="$2" ./scripts/eval/run_overrefusal_pt.sh \
    --prompts bench/overrefusal_pt/prompts_v2.jsonl --out_dir bench/overrefusal_pt/answers
done
```

**Antes de tudo, checar truncamento** (§2): `grep -c '"truncado": true' bench/safety_pt/answers/*.jsonl bench/overrefusal_pt/answers/*.jsonl`.
Se algum baseline aparecer com contagem não-trivial, re-rodar só ele com
`--max_new_tokens` maior antes de seguir.

Cada rodada imprime o resumo por categoria + `GERAL` no stdout (é a "pontuação";
não há script de agregação separado, igual já era pro `v4`). Calcular o IC de
Wilson de cada um com `scripts/eval/stats_power.py wilson --k <recusas> --n <total>`
e, se algum baseline for comparado par-a-par com o `v4` nos MESMOS itens,
`stats_power.py mcnemar --a bench/safety_pt/answers/v4-scaled.jsonl --b bench/safety_pt/answers/<label>.jsonl`.

## 5. Antes de commitar

- Auditar por amostra as respostas de segurança held-out dos baselines
  (mesmo protocolo de `safety-alignment-pt.md` §9) antes de versionar
  `bench/safety_pt/answers/*.jsonl` — modelos sem safety-tuning podem gerar
  conteúdo nocivo real nas respostas cruas, não só o Manaca-1B-SFT-sem-safety.
- `bench/safety_pt/answers/` é gitignored por padrão (contém obediência
  nociva); usar `git add -f` deliberadamente, como já feito para o `v4`.
- Commitar o harness (`run.py`, `run_overrefusal_pt.sh`) e este documento
  ANTES de rodar, pra garantir reprodutibilidade mesmo que a rodada falhe
  no meio.

## 6. Reprodutibilidade

- Harness: `bench/mtbench_pt/run_mtbench_pt.sh` (já tinha modo `HFID=`),
  `bench/overrefusal_pt/run.py` + `scripts/eval/run_overrefusal_pt.sh`
  (modo `HFID=` e `--prompt_style` adicionados nesta rodada); ambos os
  geradores (`gen_answers.py` e `run.py`) agora gravam `n_tokens_gerados`/
  `truncado` por resposta (§2), pra auditar o confundidor de tokenizador.
- Sondas: `bench/mtbench_pt/questions.jsonl`, `bench/ifeval_pt/prompts.jsonl`,
  `bench/safety_pt/heldout_v2.jsonl` (218), `bench/overrefusal_pt/prompts_v2.jsonl` (184).
- Verificação dos HF IDs: busca web datada de 2026-09-13, registrada aqui por
  não ter sido possível confirmar por download direto nesta sandbox (egress
  bloqueado para `huggingface.co`); reconfirmar na HPC antes da rodada real.
