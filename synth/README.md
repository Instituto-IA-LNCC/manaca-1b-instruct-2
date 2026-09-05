# Geração de instrução sintética — o passo que faltava

Transforma as passagens brutas de `corpus/scripts/` (LegalPT, ClassiCC-PT,
Carolina, MADLAD-400, FineWeb-2, HPLT 2.0, Portuguese-PD — todas em
`$WORK_DIR/raw/<fonte>/shard_*.parquet`) em exemplos de SFT no esquema
Alpaca, via a Anthropic Batches API (Claude).

## Por que Batches (não chamadas síncronas)

50% mais barato, até 100.000 requisições por lote, resultados disponíveis
por 29 dias. O fluxo é desacoplado: `submit` manda o lote e sai na hora
(não precisa manter o processo rodando até 24h); `collect` busca os
resultados quando prontos.

## Como funciona

1. **Amostragem em 2 passadas**: lê só `id`/`score` de todos os shards
   (não carrega texto de possivelmente centenas de milhares de docs na
   RAM), monta um pool ranqueado por qualidade, sorteia `--max-examples`
   com viés para score alto, e só então relê o texto dos IDs sorteados.
2. **Um prompt por passagem**: pede para o Claude escolher o tipo de tarefa
   mais adequado (resumo, pergunta_resposta, explicacao_simples,
   extração_pontos_chave, reescrita_tom, questão_estilo_prova), gerar uma
   instrução realista e responder **usando só o que está na passagem**
   (sem inventar fatos). O modelo pode devolver `{"skip": true}` se a
   passagem não render um bom exemplo.
3. **A passagem inteira vira o `input`** do exemplo final (`Contexto: ...`,
   mesmo padrão do manacá-jaster) — o modelo aprende a responder com base
   no contexto dado.
4. **Decontaminação best-effort** contra os arquivos estáticos deste repo
   (mesma limitação documentada em `sft/gigaverbo_v2_sft/` e
   `sft/tucano_sft/`: não cobre ENEM/BLUEX/OAB/ASSIN2/HateBR).

## Uso

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# Testar sem gastar (so mostra os prompts que seriam enviados)
python synth/generate_instructions.py run --source classiccpt --max-examples 5 --dry-run

# Fluxo completo (submete, espera, coleta)
python synth/generate_instructions.py run --source legalpt --max-examples 2000

# Fluxo desacoplado (nao precisa manter o processo rodando ate 24h)
python synth/generate_instructions.py submit --source classiccpt --max-examples 5000
python synth/generate_instructions.py collect --state data/synth/classiccpt/batch_state.json
```

Ou via `make`:

```bash
make synth-run ARGS="--source legalpt --max-examples 2000"
make synth-dry-run ARGS="--source carolina --max-examples 5"
```

Saída: `data/synth/<fonte>/<fonte>_synth.jsonl` + `manifest.json` (custo
estimado, distribuição de task_type, descartes por contaminação/JSON
inválido/decisão do próprio modelo). Usar via `--extra` no
`sft/prepare_data.py`, como os demais builders deste repo.

## Modelo e custo

Default `claude-opus-5` (não baixamos para um modelo mais barato por
padrão — é decisão do usuário via `--model`). O manifest reporta tokens de
entrada/saída reais e uma **estimativa** de custo (tabela de preços
aproximada embutida no script, já com o desconto de 50% do Batches
aplicado — confira o preço vigente antes de rodar em escala grande).

## Limitações conhecidas

- Grounding é por instrução ("não invente fatos além da passagem"), não
  verificado automaticamente — auditoria de amostra é recomendada antes de
  confiar em um lote grande.
- Decontaminação não cobre benchmarks carregados dinamicamente via HF
  (ENEM/BLUEX/OAB/ASSIN2/HateBR).
- `--max-tokens` default (1500) é pensado para um par instrução+resposta;
  passagens que peçam respostas muito longas podem ser truncadas — ajuste
  se necessário.
