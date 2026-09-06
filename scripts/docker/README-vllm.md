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

**Ainda não testado neste cluster.** Primeira vez rodando vLLM no Netuno —
espere possíveis problemas de compatibilidade driver/CUDA da imagem vLLM com
o driver do host (mesma classe de problema já visto com a imagem NGC PyTorch
neste projeto). Rode e cole o log de erro se travar na largada.

## Próximo passo (depois que o servidor subir OK)

Adaptar um script de geração (provavelmente uma cópia de
`sft/personas_br/generate_personas_sft.py` ou um script novo em cima do
mesmo padrão de decontaminação/manifesto/emoji-zero deste projeto) pra
apontar pro client OpenAI-compatible do vLLM em vez do client Anthropic,
com round-robin entre as portas 8000/8001. **Importante**: manter uma
camada barata de auditoria via Claude sobre uma AMOSTRA do que o modelo
local gerar antes de confiar no lote inteiro — modelos abertos menores
tendem a alucinar mais e ser menos disciplinados em "não inventar fatos
além do texto" do que o Claude, que é a mesma preocupação de qualidade
que motivou o cuidado com grounding no resto do pipeline sintético deste
projeto.
