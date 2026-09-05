# Aya Dataset (subset PT) — instrução humana nativa

O Aya Dataset (Singh et al., 2024, CohereForAI/CohereLabs, Apache-2.0) é
204.114 pares prompt-completion anotados por FALANTES HUMANOS (não
sintético, não tradução automática) em 65 idiomas, incluindo português.

## Por que preenche uma lacuna real

O projeto tem SFT sintético (gigaverbo-v2-sft) e traduzido automaticamente
(ExtraGLUE) em abundância, mas pouca instrução humana nativa e livre em
tipo de tarefa. O manacá-jaster é humano mas sempre no molde de benchmark
acadêmico. É pequeno, mas cobre estilo/variedade que sintéticos tendem a
homogeneizar.

## Fonte

Tenta `botbot-ai/aya_dataset_pt` (subset PT já filtrado) primeiro; se
falhar, cai para `CohereLabs/aya_dataset` (completo, multilíngue) e filtra
pela coluna de idioma (`language`/`language_code`/`lang`, detecção
tolerante de código `pt`/`por`/`portuguese`). Override manual com
`--source <hf_id>`.

## Uso

```bash
make aya-pt-data
make aya-pt-data ARGS="--source CohereLabs/aya_dataset"
```

Saída: `data/sft/aya_pt/aya_pt.jsonl` (Alpaca) + `manifest.json` (reporta
qual fonte/coluna de idioma foi de fato usada).
