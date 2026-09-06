# Lotes manuais — geração sintética sem custo de API

Mesma tarefa de `synth/generate_instructions.py` (transformar uma passagem
de `corpus/scripts/` em `{instruction, input, output, task}` grounded, sem
inventar fatos além do que está no texto), mas gerada MANUALMENTE por
Claude nesta sessão de código, a partir de passagens coladas no chat pelo
usuário — sem gastar créditos da API, para complementar as fontes que
usam a Batches API quando o orçamento está apertado.

Cada `<fonte>_synth.jsonl` segue o mesmo schema Alpaca e a mesma
convenção de `task` (`synth_<fonte>_<task_type>`) do pipeline via API,
para que `sft/prepare_data.py` trate os dois de forma idêntica. Cada um
tem um `<fonte>_manifest.json` companheiro documentando a proveniência
(passagens de origem, método manual, verificação de decontaminação).

**Por que não fica em `data/`**: `data/` está no `.gitignore` (dados
gerados via script não são versionados, qualquer um reproduz rodando o
script). Este lote não tem script de regeneração determinístico — foi
composto uma vez, manualmente — por isso o arquivo final é versionado
diretamente aqui.

**Escopo**: pequeno por natureza (gerar manualmente não escala como a
API). Serve para cobrir uma fatia de uma fonte sem custo enquanto o
orçamento de API é usado nas fontes que precisam de volume maior.
