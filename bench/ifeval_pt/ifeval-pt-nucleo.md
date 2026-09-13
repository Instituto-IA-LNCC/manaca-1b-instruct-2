Acuracia IFEval-PT (%) ± erro padrao. strict=resposta crua; loose=normalizada.

| Modelo | Prompt strict | Prompt loose | Instr strict | Instr loose |
|---|---|---|---|---|
| safe-v4 | 22.2 ±6.9 | 22.2 ±6.9 | 34.0 ±6.7 | 34.0 ±6.7 |
| instruct2-full | 27.8 ±7.5 | 27.8 ±7.5 | 40.0 ±6.9 | 40.0 ±6.9 |
| ttl-460m-chat | 8.3 ±4.6 | 8.3 ±4.6 | 18.0 ±5.4 | 18.0 ±5.4 |
| tucano-1b1-instruct | 11.1 ±5.2 | 13.9 ±5.8 | 22.0 ±5.9 | 26.0 ±6.2 |
| tucano-2b4-instruct | 13.9 ±5.8 | 13.9 ±5.8 | 26.0 ±6.2 | 26.0 ±6.2 |
| tucano2-qwen-1b5-instruct | 41.7 ±8.2 | 44.4 ±8.3 | 58.0 ±7.0 | 60.0 ±6.9 |

Instr-loose (%) por tipo de instrucao:

| Tipo | safe-v4 | instruct2-full | ttl-460m-chat | tucano-1b1-instruct | tucano-2b4-instruct | tucano2-qwen-1b5-instruct |
|---|---|---|---|---|---|---|
| comecar_com | 0 | 0 | 0 | 0 | 0 | 33 |
| evitar_palavra | 50 | 100 | 100 | 100 | 50 | 100 |
| incluir_palavra | 43 | 43 | 29 | 43 | 57 | 57 |
| incluir_todas | 0 | 0 | 100 | 100 | 100 | 100 |
| json_valido | 50 | 0 | 0 | 50 | 0 | 100 |
| max_frases | 100 | 100 | 100 | 50 | 50 | 50 |
| max_palavras | 33 | 100 | 0 | 0 | 0 | 0 |
| min_maiusculas | 0 | 0 | 0 | 0 | 0 | 100 |
| min_marcadores | 0 | 0 | 0 | 0 | 50 | 50 |
| min_palavras | 75 | 100 | 50 | 100 | 100 | 100 |
| min_placeholders | 100 | 100 | 0 | 100 | 100 | 100 |
| num_frases | 0 | 100 | 0 | 0 | 0 | 100 |
| num_itens_numerados | 0 | 0 | 0 | 0 | 0 | 100 |
| num_marcadores | 0 | 0 | 0 | 0 | 0 | 100 |
| num_paragrafos | 0 | 0 | 0 | 0 | 0 | 33 |
| palavra_repetida | 0 | 50 | 0 | 0 | 0 | 0 |
| pos_escrito | 50 | 0 | 0 | 0 | 0 | 50 |
| sem_virgula | 100 | 50 | 0 | 0 | 0 | 50 |
| terminar_com | 0 | 0 | 0 | 0 | 0 | 33 |
| titulo_colchetes | 0 | 0 | 0 | 0 | 0 | 0 |
| tudo_maiusculo | 0 | 0 | 0 | 0 | 0 | 50 |
| tudo_minusculo | 100 | 100 | 0 | 0 | 0 | 100 |
