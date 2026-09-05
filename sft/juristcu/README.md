# JurisTCU — julgamento de relevância jurídica PT-BR

`LeandroRibeiro/JurisTCU` (Fernandes et al., 2026, Language Resources and
Evaluation, CC-BY-4.0) é um dataset de IR jurídica: 16.045 documentos de
jurisprudência do TCU, 150 consultas (reais de log de busca + sintéticas)
e 2.250 julgamentos de relevância (score 0-3, revisão manual).

## Por que é um script próprio (mesmo padrão do Quati)

Assim como o Quati, o JurisTCU não é um dataset único — são 3 arquivos CSV
relacionados por ID (`doc.csv`, `query.csv`, `qrel.csv`) baixados via
`huggingface_hub.hf_hub_download` (não `datasets.load_dataset`, que espera
configs padronizadas).

## Por que preenche uma lacuna real

Conteúdo jurídico/institucional brasileiro com CONSULTAS REAIS de
usuários — complementa o LegalPT (só corpus, sem consulta) e o Quati
(domínio geral, não jurídico).

## Conversão

Julgamento binário de relevância, mesmo estilo do Quati: dada a consulta e
o trecho (ENUNCIADO+EXCERTO), "Sim" ou "Não" é relevante, usando o `SCORE`
do qrel (escala 0-3 **documentada** na própria ficha, ao contrário do
Quati) contra `--relevance_threshold` (default 2.0).

## Uso

```bash
make juristcu-data
make juristcu-data ARGS="--relevance_threshold 1"
```

Saída: `data/sft/juristcu/juristcu.jsonl` (Alpaca) + `manifest.json`.
