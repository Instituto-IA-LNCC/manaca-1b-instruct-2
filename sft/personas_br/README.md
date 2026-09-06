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

**Emoji é banido (tolerância zero), por causa do tokenizer, não de estilo.**
O tokenizer do `manaca-1b-base` é um SentencePiece Unigram treinado do
zero só no corpus PT-BR do projeto (`character_coverage=0.9995`,
`byte_fallback=True`) — emoji não tem token dedicado, cada um vira ~4
tokens de fallback por byte UTF-8, sem nenhuma exposição no pré-treino.
Não é só "caro" (4x mais tokens que o normal): o modelo pode aprender a
gerar só parte da sequência de bytes e produzir UTF-8 inválido. O
`SYSTEM_PROMPT` já pede para não usar; `count_emojis(output) > 0` é uma
rede de segurança contra o modelo ignorar a instrução (registrado no
manifest como `skipped_emoji_pesado`).

## Uso

```bash
export ANTHROPIC_API_KEY=sk-ant-...
make personas-br-dry-run                                   # sem custo
make personas-br-data ARGS="--num_personas 2000"
```

Saída: `data/sft/personas_br/personas_br.jsonl` (Alpaca) + manifest com
custo estimado (tokens reais, sem desconto de Batches).

## Custo real — `--effort low` por padrão

`claude-sonnet-5`/`claude-opus-5` rodam com *thinking* adaptativo LIGADO
por padrão quando `output_config.effort` não é especificado. Confirmado
ao vivo: uma rodada de 500 personas gastou **US$7,15** e manteve só 48
exemplos (90% descartados pelo próprio modelo) — a média de tokens de
saída por chamada (~1.191) ficou quase no teto do `max_tokens`, mesmo
contando as centenas de chamadas que só deveriam responder
`{"skip": true}` (quase de graça). Ou seja, a maior parte do orçamento
estava indo pra raciocínio invisível, não pra resposta em si — o que
também explica a taxa alta de skip (o modelo ficava sem espaço de token
pra escrever uma resposta completa). Por isso o script usa
`--effort low` por padrão (ajustável via `--effort medium|high|xhigh|max`
se a qualidade não for suficiente).

## Retomada (créditos limitados)

Cada exemplo é gravado no disco (e o arquivo, `flush`ado) assim que sai
da API — nada fica só em memória. O manifest grava `processed_idx`
(posição na amostra) a cada 20 personas e ao final. Se a rodada parar no
meio (créditos esgotados, Ctrl-C, container derrubado) — inclusive se o
próprio script detectar um erro 400/403 de saldo/crédito e parar sozinho
(status `stopped_fatal_error`, saída com código 2) — **rode o mesmo
comando de novo** com os mesmos `--num_personas`/`--seed`/`--model`: ele
pula as personas já tentadas e usa modo *append* no arquivo de saída, sem
duplicar nem pagar de novo pelo que já foi processado.
