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

## Auditoria real (34.992 pares, medido — não amostra)

- **Sem `<think>`** (0% dos pares) — diferente do `gigaverbo_v2_preferences`,
  sem risco de formato de raciocínio não ensinado no SFT.
- **Viés de verbosidade real e sistemático**: `chosen` é 37% mais longo em
  média que `rejected` (1.703 vs 1.244 chars), e `chosen > rejected` em
  67,7% dos pares — não é ruído de amostra, é o dataset inteiro. Risco
  real do DPO aprender "responda mais longo" como atalho pra "responda
  melhor" em vez de aprender qualidade de verdade. **Antes de usar**:
  considerar balancear/filtrar pares com diferença de tamanho extrema, ou
  pelo menos monitorar verbosidade do modelo pós-DPO na avaliação (não só
  segurança/over-refusal).
