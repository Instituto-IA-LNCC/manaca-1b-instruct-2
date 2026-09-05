# Reward-Aira — fonte complementar de preferência

`nicholasKluge/reward-aira-dataset` (projeto Aira, Apache-2.0): dataset de
preferência PT-BR menor, complementar ao
[`gigaverbo_v2_preferences`](../gigaverbo_v2_preferences/) para a mesma
segunda tentativa de DPO deste projeto.

Mesmo esquema de saída (`{instruction, input, chosen, rejected}`) e mesma
política de adoção: converter não decide usar — a avaliação decide (ver
`dpo/README-onpolicy.md`).

## Uso

```bash
make dpo-reward-aira
```

Saída: `dpo_onpolicy/reward_aira/reward_aira.jsonl` (não versionado).
