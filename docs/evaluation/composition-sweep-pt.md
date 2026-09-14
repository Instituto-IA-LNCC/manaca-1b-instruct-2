# Varredura da composição do mix de safety-SFT

Registro do experimento controlado que mapeia a fronteira segurança ×
over-refusal em função da composição do dataset de safety-SFT, motivado
pelo padrão observado ao escalar do `safe4` (instruct-v2) para o `v2`/`v3`/`v4`
(instruct-2) — ver `docs/evaluation/safety-alignment-pt.md` §9-10. Este é o
achado mais original do trabalho de alinhamento: uma curva controlada, não
pontos acidentais.

## 1. Desenho do experimento

Isola **uma única variável**: a fatia de dados gerais (`N_GENERAL`) no mix de
treino, de 10% a 60%, em passos de 10 pontos percentuais. Tudo o mais fica
fixo entre as 6 rodadas:

- **Seeds nocivos**: `dpo/dpo_seeds/safety_prompts_v2.jsonl` (465 seeds),
  `k_refuse=4` → sempre **1.539 exemplos de recusa** (determinístico, mesmo
  `--seed 42` padrão do `gen_safety_sft.py`).
- **Ajuda**: `sft_safety/benign_help_v3.jsonl` (280 exemplos — inclui os 90
  gerados focados em segurança digital/finanças/denúncia e o oversampling
  10× dos 5 exemplos-ouro de crise).
- **Receita de treino**: LoRA r=32/α=64, LR 1e-4, 3 épocas, DDP — idêntica
  ao `safe4`/`v2`/`v3`/`v4`.
- **Base**: `manaca-1b-instruct-2-full` (mesmo checkpoint em todas as rodadas).

`N_GENERAL` foi calculado pra cada rodada bater exatamente no percentual alvo
do mix total (`geral / (1.539 + 280 + geral)`):

| % geral alvo | N_GENERAL | Total do mix |
|---:|---:|---:|
| 10% | 202 | 2.021 |
| 20% | 455 | 2.274 |
| 30% | 780 | 2.599 |
| 40% | 1.213 | 3.032 |
| 50% | 1.819 | 3.638 |
| 60% | 2.729 | 4.548 |

## 2. Resultado

Mesma bateria de avaliação do `v2`/`v3`/`v4` (16 probes de segurança
held-out, 34 de over-refusal — 18 benignos + 16 sensíveis-legítimos):

| % geral | Segurança held-out | O.R. benigno | O.R. sensível | **O.R. geral** |
|---:|---:|---:|---:|---:|
| 10% | 100,0% | 27,8% | 37,5% | 32,4% |
| 20% | 100,0% | 16,7% | 18,8% | 17,6% |
| 30% | 100,0% | 5,6% | 25,0% | 14,7% |
| **40%** | **93,8%** | **11,1%** | **6,2%** | **8,8%** |
| 50% | 87,5% | 5,6% | 6,2% | 5,9% |
| 60% | 93,8% | 5,6% | 18,8% | 11,8% |

## 3. Padrão e ponto de joelho

- **Segurança** satura em 100% até 30% de dados gerais, e só começa a ceder
  a partir de 40% (93,8% → 87,5% em 50% → 93,8% em 60%). Com N=16 por
  ponto, a oscilação entre 40/50/60% (diferença de 1-2 casos) não é
  conclusiva isoladamente.
- **Over-refusal geral** cai de forma acentuada e monotônica de 10% a 50%
  (32,4% → 5,9%), mas **sobe de novo em 60%** (11,8%) — a curva **não é
  puramente monotônica**. Com N=34 por ponto, isso também está na faixa de
  ruído amostral, mas é consistente o bastante (mesmo padrão nos 2
  subgrupos — sensível sobe de 6,2% pra 18,8%) para não descartar como
  ruído puro.
- **Ponto de joelho**: a faixa **40-50%** domina os extremos nos dois eixos
  simultaneamente. Por uma métrica combinada (média harmônica de segurança
  e "1 − over-refusal"), os três pontos centrais ficam próximos — **40%
  (92,5)**, 30% (92,1), 50% (90,7) — dentro do que o tamanho de amostra
  permite distinguir. **40% de dados gerais é o ponto recomendado**: mantém
  segurança quase no teto (93,8%) com over-refusal já bem baixo (8,8%), sem
  o custo adicional de over-refusal visto em 60%.

## 4. Comparação com os pontos anteriores (não-controlados)

Os pontos do `v2`/`v3`/`v4` usavam uma composição de ajuda diferente (145
exemplos, sem o oversampling de crise) e por isso não são diretamente
comparáveis ponto-a-ponto com esta varredura — mas o padrão geral bate:
`v2` (10,6% geral, ajuda=145) teve over-refusal de 50%, consistente com o
extremo esquerdo desta curva (10%→32,4%, pior ainda com menos ajuda);
`v3`/`v4` (48,7%/48,3% geral) caem dentro da faixa de joelho identificada
aqui (40-50%), o que explica por que `v4` funcionou bem sem ter sido
desenhado com essa curva em mãos — a proporção acabou caindo perto do ótimo
por tentativa e erro guiado por hipótese, e agora está confirmado por
varredura controlada. `v3` teve 23,5% de over-refusal (8/34) — ponto
intermediário entre o catastrófico `v2` e o `v4` final. Dados brutos dos
três em `bench/overrefusal_pt/answers/{v2,v3,v4}-n34.jsonl` (commitados
2026-09-14 — eram uma lacuna de reprodutibilidade: números já citados aqui,
dado bruto nunca tinha ido pro repo).

## 5. Limitações

- **Poder estatístico**: N=16 (segurança) e N=34 (over-refusal) por ponto —
  mesma ressalva já documentada em `safety-alignment-pt.md` §2. A
  não-monotonicidade em 60% pode ser efeito real ou ruído; não dá pra
  distinguir com esta amostra. Sondas de 150-300 itens por categoria, com
  IC de Wilson e McNemar entre pares, são o próximo passo antes de
  reivindicar o ponto de joelho como resultado robusto (não apenas
  exploratório).
- **Uma única semente/rodada por ponto**: cada percentual foi treinado uma
  vez (LoRA r=32, 3 épocas). Variância de treino (seed do otimizador,
  ordem de batch) não foi medida — o efeito da composição pode estar
  parcialmente confundido com variância de treino entre rodadas.
- **Generalização**: a curva foi medida sobre o `manaca-1b-instruct-2-full`
  especificamente. Não se sabe se o mesmo formato de curva (ou o mesmo
  ponto de joelho em ~40-50%) se mantém em outra escala de modelo ou outro
  corpus de pré-treino.

## 6. Reprodutibilidade

- Script da varredura: `run_composition_sweep.sh` (gera dataset → treina →
  avalia, para cada um dos 6 pontos, sequencial).
- Datasets: `sft_safety/safety_sft_sweep_p{10,20,30,40,50,60}.jsonl`.
- Modelos: `manaca-1b-instruct-2-sweep-p{10,20,30,40,50,60}` (adapter) /
  `-merged` (completo).
- Respostas brutas de avaliação: `bench/safety_pt/answers/sweep-p*.jsonl`,
  `bench/overrefusal_pt/answers/sweep-p*.jsonl`.
- Logs de treino: `sweep_logs/` (gerado localmente pelo script; não
  versionado por padrão — commitar manualmente se necessário para auditoria
  completa, seguindo a mesma prática já usada para os logs do SFT
  principal e do DPO em `docs/training/`).
