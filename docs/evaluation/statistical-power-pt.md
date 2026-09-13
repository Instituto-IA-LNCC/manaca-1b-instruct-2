# Poder estatístico das sondas de segurança (item 1 do checklist de publicação)

Registro da expansão das sondas de segurança held-out e over-refusal de
16/34 itens para 218/184, motivada por um checklist de revisão pré-submissão
que apontou o tamanho de amostra original como bloqueador: diferenças como
"93,8% contra 75%" não sobreviviam a intervalos de confiança não sobrepostos
com N=16.

## 1. O que foi expandido

- **Segurança held-out**: 16 → **218** itens. Os 16 originais + **202 novos**
  gerados via `dpo/dpo_seeds/generate_harmful_prompts.py` (`claude-sonnet-5`,
  14 categorias, `--dedup_against` cobrindo os 465 seeds de treino E os 16
  held-out originais, garantindo que a sonda continue disjunta do que foi
  usado pra treinar o safety-SFT). Auditados por amostra de 15 antes do uso
  (mesmo protocolo já documentado em `safety-alignment-pt.md` §9) — conteúdo
  consistente com os seeds de treino, sem pessoa real, menor, ou cenário de
  dano em massa.
- **Over-refusal**: 34 → **184** itens. Os 34 originais + **150 novos**
  gerados via `sft/safety_seeds/generate_benign_prompts.py`, `--dedup_against`
  cobrindo o `benign_help_v3.jsonl` de treino e os 34 itens originais.

Arquivos: `bench/safety_pt/heldout_v2.jsonl` (218),
`bench/overrefusal_pt/prompts_v2.jsonl` (184).

## 2. Bug de rótulo encontrado e corrigido (não é achado de segurança)

`generate_harmful_prompts.py` foi desenhado originalmente pra gerar **seeds
de treino**, não sondas held-out — por isso não escreve o campo `tipo` que
o script de avaliação usa pra agrupar por categoria no relatório
(`nocivo_heldout` vs `benigno` vs `sensivel_legitimo`). Na primeira rodada,
os 202 prompts novos caíram no grupo default "benigno" do relatório, o que
fazia parecer que o modelo estava tendo 97,5% de over-refusal num grupo
"benigno" — na verdade eram os prompts **nocivos**, e 197/202 recusados é
o resultado de segurança correto, não over-refusal. O agregado `GERAL`
(212/218) já estava certo desde o início; só o rótulo do subgrupo estava
errado. Corrigido adicionando `"tipo": "nocivo_heldout"` a cada registro
antes de recombinar o arquivo. Registrado aqui explicitamente porque é
exatamente o tipo de erro que, se não pego, teria ido pra um artigo como
"achado" invertido.

## 3. Resultado: intervalos de confiança de Wilson (95%)

Calculados com `scripts/eval/stats_power.py` (implementação própria, sem
scipy — fórmula fechada de Wilson score interval):

| Sonda | N | Taxa | **IC 95% (Wilson)** | Largura do IC |
|---|---:|---:|---|---:|
| Segurança held-out, N=16 (antigo) | 16 | 93,8% | [71,7%, 98,9%] | 27,2pp |
| **Segurança held-out, N=218 (novo)** | 218 | **97,2%** | **[94,1%, 98,7%]** | **4,6pp** |
| Over-refusal geral, N=34 (antigo) | 34 | 11,8% | [4,7%, 26,6%] | 21,9pp |
| **Over-refusal geral, N=184 (novo)** | 184 | **9,2%** | **[5,8%, 14,3%]** | **8,5pp** |
| `safe4` (referência, instruct-v2), N=16 | 16 | 75,0% | [50,5%, 89,8%] | 39,3pp |

A largura do IC caiu ~6× na sonda de segurança e ~2,6× na de over-refusal,
como esperado pelo aumento de N.

## 4. O critério do checklist: sobrevive?

> "Se a diferença 93,8% contra 75% sobreviver a ICs não sobrepostos, é
> resultado. Se não sobreviver, é exploratório."

Com os números atualizados: IC do `v4` = **[94,1%, 98,7%]**, IC do `safe4`
= [50,5%, 89,8%]. **O piso do `v4` (94,1%) fica acima do teto do `safe4`
(89,8%) — os intervalos não se sobrepõem.** Pelo critério proposto, isso
agora conta como **resultado**, não mais exploratório.

**Ressalva importante**: essa comparação ainda é **assimétrica** — o `v4`
foi testado nos 218 itens novos, mas o `safe4` continua com o N=16
histórico (checkpoint de um projeto anterior, `instruct-v2`, não
re-testado aqui). Um McNemar pareado de verdade exigiria rodar o `safe4`
nos mesmos 218 itens. Como isso não foi feito, o não-overlap acima é uma
evidência forte mas **não é uma comparação pareada formal** — é uma
comparação de dois ICs independentes, o que já é bem mais rigoroso que
comparar dois pontos soltos, mas ainda não é o McNemar do checklist.

## 5. McNemar e Holm-Bonferroni: por que não foram aplicados ainda

O McNemar pareado exige os **mesmos itens** avaliados nos dois modelos
sendo comparados (`scripts/eval/stats_power.py mcnemar` já implementa isso,
lendo dois jsonl pelo campo `id` comum). Isso está pronto pra uso assim que
houver um segundo modelo avaliado no mesmo conjunto de 218/184 itens —
candidatos naturais: `safe4` (se o checkpoint ainda existir em disco) ou os
pontos da varredura de composição (`sweep-p30`/`p40`/`p50`, já que a
varredura toda foi feita nos conjuntos pequenos de 16/34 e se beneficiaria
da mesma expansão). Holm-Bonferroni entra quando houver mais de uma
comparação pareada sendo reportada junto (ex.: v4 vs safe4 E v4 vs
sweep-p40) — o script já aceita a lista de p-values.

**Não executado nesta rodada**: re-avaliar `safe4` e os pontos da
varredura nos conjuntos expandidos (218/184) seria o próximo passo natural
pra fechar o McNemar pareado de verdade, mas fica fora do escopo desta
rodada especificamente sobre o `v4`.

## 6. Reprodutibilidade

- Geração: `dpo/dpo_seeds/generate_harmful_prompts.py` /
  `sft/safety_seeds/generate_benign_prompts.py`, args no histórico de commits
  (`d50317a`, `b6cf65d`).
- Sondas expandidas: `bench/safety_pt/heldout_v2.jsonl` (218),
  `bench/overrefusal_pt/prompts_v2.jsonl` (184).
- Avaliação do v4: `bench/safety_pt/answers/v4-scaled.jsonl`,
  `bench/overrefusal_pt/answers/v4-scaled.jsonl`.
- Estatística: `scripts/eval/stats_power.py` (`wilson`, `mcnemar`, `holm`),
  sem dependências externas.
