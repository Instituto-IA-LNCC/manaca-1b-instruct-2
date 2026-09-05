# Personas-BR — SFT diverso via personas brasileiras (NVIDIA Nemotron, Claude)

`nvidia/Nemotron-Personas-Brazil` (CC BY 4.0) é 6 milhões de personas
sintéticas em português, ancoradas em distribuições demográficas reais do
IBGE. Diferente de tudo já usado no projeto, não é QA nem texto cru — é
matéria-prima para gerar instrução diversa e culturalmente brasileira via
Claude, evitando o problema comum de SFT sintético ficar
genérico/"americanizado".

## Como funciona

Para cada persona sorteada (reservoir sampling, 1 passada de streaming —
6M linhas não cabe na RAM), pede ao Claude um pedido plausível que essa
pessoa faria a um assistente de IA + uma resposta útil, em uma única
chamada JSON. Chamadas síncronas (volume da ordem de milhares por rodada,
não Batches). Decontamina contra os arquivos estáticos deste repo (mesmo
mecanismo do `synth/generate_instructions.py`).

## Uso

```bash
export ANTHROPIC_API_KEY=sk-ant-...
make personas-br-dry-run                                   # sem custo
make personas-br-data ARGS="--num_personas 2000"
```

Saída: `data/sft/personas_br/personas_br.jsonl` (Alpaca) + manifest com
custo estimado (tokens reais, sem desconto de Batches).
