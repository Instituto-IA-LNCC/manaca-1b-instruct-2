# GigaVerbo-v2 Preferences — 2ª tentativa de DPO

`Polygl0t/gigaverbo-v2-preferences` (Tucano 2, arXiv:2603.03543, Apache
2.0): ~28,4K pares chosen/rejected em 4 subconjuntos (harmless/harmful ×
com/sem raciocínio), `chosen` gerado por Qwen2.5-32B-Instruct e `rejected`
por Qwen2.5-7B-Instruct.

## Por que importa

O DPO on-policy do v1/v2 foi **resultado negativo documentado**
(`dpo/README-onpolicy.md`, `docs/evaluation/safety-alignment-pt.md`): o
dataset era pequeno e gerado pelo próprio Manacá, e o prior de obediência
era forte demais para a margem de treino mudar a geração livre. Este
dataset é maior, gerado por dois modelos terceiros (não o modelo em
treino) e focado em harmlessness — dá uma segunda chance ao método, com
uma característica que faltava (volume + separação clara chosen/rejected).

**Não é garantia de sucesso.** Converter aqui não decide adotar o DPO —
isso é decidido pela mesma avaliação já usada no projeto (segurança no
MT-Bench-PT sobe, over-refusal não sobe, IFEval-PT/geral não regride).

## Uso

```bash
make dpo-gigaverbo-v2-preferences
```

Saída: `dpo_onpolicy/gigaverbo_v2_preferences/gigaverbo_v2_preferences.jsonl`
no esquema `{instruction, input, chosen, rejected}` que `dpo/train.py`
consome (não versionado — `dpo_onpolicy/` está no `.gitignore`, como as
demais gerações de pares de segurança deste repo).

## Auditoria real (28.421 pares, medido — não amostra)

- **`<think>...</think>` exposto em 48,0% dos pares** (chosen e/ou
  rejected): `chosen` vem de Qwen2.5-32B-Instruct com o bloco de
  raciocínio literal na string, não só a resposta final. O SFT do Manacá
  (template Alpaca, `### Resposta:`) nunca ensinou esse formato — treinar
  DPO direto nesse dado arrisca o modelo aprender a emitir `<think>` tags
  soltas fora de contexto (o DPO reforça preferência relativa, não ensina
  sintaxe nova de forma confiável). **Antes de usar**: extrair só o texto
  após `</think>` (ou descartar os pares com `<think>`) em ambos
  chosen/rejected.
- **Sem viés de tamanho** (checado, não só amostra): `chosen` é em média
  *mais curto* que `rejected` (1.812 vs 2.120 chars), e `chosen > rejected`
  em só 51,6% dos pares — praticamente 50/50. Diferente do
  `reward_aira` (ver README lá), aqui não há risco de o DPO aprender
  "responda mais longo = melhor".
