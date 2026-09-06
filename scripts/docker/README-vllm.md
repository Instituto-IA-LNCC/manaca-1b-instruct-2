# LLM local (vLLM) para escalar geração sintética sem custo de API

Alternativa a `synth/generate_instructions.py` / `sft/personas_br/generate_personas_sft.py`
/ `sft/aes_enem/generate_aes_enem_feedback.py` (que usam a API da Anthropic):
roda um modelo aberto na GPU do Netuno e serve via vLLM num endpoint
OpenAI-compatible, pra gerar em volume sem gastar créditos.

## Por que Qwen2.5-32B-Instruct-AWQ

- Cobertura de PT-BR competitiva com Llama 3.x nas gerações recentes (2.5/3).
- Faixa de tamanhos mais granular que a Llama (7B/14B/32B/72B) — dá pra
  ajustar à VRAM disponível sem pular de 8B pra 70B.
- Já tem precedente no domínio: o Tucano 2/GigaVerbo-v2 usou Qwen2.5-32B como
  "professor" pra anotar qualidade em dados PT-BR.
- Licença Apache-2.0 (mais permissiva que a "community license" da Llama).
- Em AWQ (4-bit) cabe **sozinho numa única RTX A5000 de 24 GB** — não precisa
  de tensor-parallel entre as duas GPUs do Netuno.

## Por que 2 instâncias (1 GPU cada), não `--tensor-parallel-size 2`

As 2 RTX A5000 do Netuno **não têm NVLink** (ver `.env.example`, comentário
de `GPUS_PER_NODE`). Tensor parallelism sem NVLink faz all-reduce a cada
camada via PCIe — gargalo real pra throughput de geração em lote. Rodar duas
instâncias independentes (uma por GPU) e balancear as chamadas entre as duas
no client escala 2x sem esse overhead.

## Uso

```bash
make llm-serve-0   # GPU 0, porta 8000
make llm-serve-1   # GPU 1, porta 8001 (rodar junto pra ter as duas)

# acompanhar o download do modelo (so na 1a vez, pode levar minutos)
docker logs -f manaca-vllm-0

# smoke test depois de "Uvicorn running on ..." aparecer no log
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen2.5-32B-Instruct-AWQ",
  "messages": [{"role": "user", "content": "Responda em pt-BR: qual a capital do Brasil?"}]
}'
```

Fallback se a VRAM/o download de 32B AWQ não couber ou for problemático:
`MODEL=Qwen/Qwen2.5-14B-Instruct-AWQ make llm-serve-0`.

Parar: `docker stop manaca-vllm-0 manaca-vllm-1 && docker rm manaca-vllm-0 manaca-vllm-1`.

## Status

**Validado ao vivo no Netuno** (06/09/2026): as duas instâncias sobem OK
com `ENFORCE_EAGER=1` (ver nota abaixo). Teste de geração em PT-BR
respondeu correto, fluente e sem alucinação visível pra uma pergunta
factual simples.

Nota sobre `ENFORCE_EAGER=1`: sem essa flag, a primeira subida (cold
start) inclui uma fase de compilação/captura de CUDA graph que demorou
vários minutos sem emitir log nenhum nesse hardware (RTX A5000, sem as
otimizações de datacenter) — fácil de confundir com um travamento. Rodar
com `ENFORCE_EAGER=1` pula essa fase (custa algum throughput de
inferência) e é o que foi usado na validação. Revisitar se vale tirar
depois de confirmar throughput suficiente em uso real.

## Geração via LLM local

`synth/generate_instructions_local.py` — mesmo pipeline de
`synth/generate_instructions.py` (amostragem de passagens, SYSTEM_PROMPT,
decontaminação, esquema de saída `{instruction, input, output, task}`),
mas falando com os endpoints OpenAI-compatible do vLLM em vez da API da
Anthropic. Mesma política de emoji zero de `sft/personas_br/` e
`sft/aes_enem/`, resume via manifest.json incremental, circuit-breaker que
tira um endpoint da rotação se ele cair, e uma auditoria opcional barata
via Claude (`--audit-sample N`) sobre uma amostra do lote gerado —
**recomendado** rodar pelo menos uma vez por fonte antes de confiar no
lote inteiro, já que um modelo aberto menor tende a alucinar mais e ser
menos disciplinado em "não inventar fatos" do que o Claude.

```bash
make synth-local-dry-run ARGS="--source classiccpt --max-examples 5"
make synth-local-run ARGS="--source classiccpt --max-examples 2000 --audit-sample 30"
```

Saída: `data/synth_local/<source>/<source>_synth.jsonl` + `manifest.json`
(+ `audit_report.json` se `--audit-sample > 0`) — mesmo uso via `--extra`
de `sft/prepare_data.py` que os demais builders deste repo.
