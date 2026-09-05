# Tucano-SFT — instrução pronta (TucanoBR)

`TucanoBR/Tucano-SFT` é o dataset de instrução usado para treinar os
próprios Tucano-Instruct (1b1/2b4): concatenação de
`cnmoro/GPT4-500k-Augmented-PTBR-Clean` (MIT), `rhaymison/orca-math-portuguese-64k`
(Apache-2.0) e `nicholasKluge/instruct-aira-dataset-v3` (Apache-2.0).

## Por que vale a pena

Mais instrução PT-BR (nativa + traduzida) pronta, sem custo de geração —
mesma categoria do `sft/gigaverbo_v2_sft/`. Traz math (orca-math) e
conversação (Aira), categorias que o manacá-jaster mal cobre.

## Aviso de proveniência (transparência)

Uma das 3 subfontes foi gerada consultando GPT-4 da OpenAI, cujos termos
de uso restringem usar saídas para treinar modelos concorrentes — tensão
conhecida e tolerada na comunidade (a linhagem Alpaca, já no v1/v2 deste
projeto, tem a mesma característica). Registrado por auditoria, não é um
bloqueio.

## Uso

```bash
make tucano-sft-data                              # default: 50k exemplos
make tucano-sft-data ARGS="--max_examples 0"      # sem teto
```

Saída: `data/sft/tucano_sft/tucano_sft.jsonl` + `manifest.json`. Usar via
`--extra` no `sft/prepare_data.py`, como o manacá-jaster/gigaverbo_v2_sft.

## Limitações

Mesma dobra de conversa multi-turno em Alpaca de turno único do
`gigaverbo_v2_sft` (train.py não suporta multi-turno nativo), e mesma
decontaminação best-effort contra os arquivos estáticos do repo (não
cobre ENEM/BLUEX/OAB/ASSIN2/HateBR, carregados via HF pelo lm-eval).
