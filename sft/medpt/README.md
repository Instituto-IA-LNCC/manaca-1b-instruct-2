# MedPT — QA médico real e nativo em PT-BR

`AKCIT/MedPT` (D'Addario et al., 2025, arXiv:2511.11878, aceito na LREC
2026, CC BY 4.0) é o maior corpus público de QA médico REAL e NATIVO em
português do Brasil: 384.095 pares de interações reais paciente-médico,
3.200+ condições, 104 especialidades.

## Por que preenche uma lacuna real

O projeto tem corpus jurídico, enciclopédico e crawl genérico, mas nenhum
QA nativo de domínio — muito menos saúde, onde tradução do inglês perde
nuance clínica e cultural (doenças endêmicas, gírias regionais).

## Conversão

Direta, sem API: `instruction` = pergunta do paciente, `output` = resposta
real do profissional. Filtra respostas muito curtas (< 20 caracteres) ou
muito longas (> 4000, provável erro de parsing).

## Ressalva de licença

CC-BY-4.0 confirmado via busca (arXiv + paper-page do HF) — o acesso
direto à ficha do HF estava bloqueado no ambiente em que este script foi
escrito. O script tenta confirmar via `huggingface_hub.dataset_info` e
imprime o que encontrar; **confira a ficha manualmente** antes de usar em
escala grande.

## Uso

```bash
make medpt-data                              # tudo, default
make medpt-data ARGS="--max_examples 100000"
```

Saída: `data/sft/medpt/medpt.jsonl` (Alpaca) + `manifest.json` (distribuição
por especialidade, contagem de descartes).
