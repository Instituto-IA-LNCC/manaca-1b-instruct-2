Acuracia IFEval-PT (%) ± erro padrao. strict=resposta crua; loose=normalizada.

| Modelo | Prompt strict | Prompt loose | Instr strict | Instr loose |
|---|---|---|---|---|
| instruct2-full | 27.8 ±7.5 | 27.8 ±7.5 | 40.0 ±6.9 | 40.0 ±6.9 |
| safe-v4 | 22.2 ±6.9 | 22.2 ±6.9 | 34.0 ±6.7 | 34.0 ±6.7 |

Instr-loose (%) por tipo de instrucao:

| Tipo | instruct2-full | safe-v4 |
|---|---|---|
| comecar_com | 0 | 0 |
| evitar_palavra | 100 | 50 |
| incluir_palavra | 43 | 43 |
| incluir_todas | 0 | 0 |
| json_valido | 0 | 50 |
| max_frases | 100 | 100 |
| max_palavras | 100 | 33 |
| min_maiusculas | 0 | 0 |
| min_marcadores | 0 | 0 |
| min_palavras | 100 | 75 |
| min_placeholders | 100 | 100 |
| num_frases | 100 | 0 |
| num_itens_numerados | 0 | 0 |
| num_marcadores | 0 | 0 |
| num_paragrafos | 0 | 0 |
| palavra_repetida | 50 | 0 |
| pos_escrito | 0 | 50 |
| sem_virgula | 50 | 100 |
| terminar_com | 0 | 0 |
| titulo_colchetes | 0 | 0 |
| tudo_maiusculo | 0 | 0 |
| tudo_minusculo | 100 | 100 |
