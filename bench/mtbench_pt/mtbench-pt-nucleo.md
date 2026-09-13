Notas MT-Bench-PT (media de 1 a 10 ± erro padrao). Juiz: LLM-as-a-Judge.

| Categoria | safe-v4 | instruct2-full | ttl-460m-chat | tucano-1b1-instruct | tucano-2b4-instruct | tucano2-qwen-1b5-instruct |
|---|---|---|---|---|---|---|
| escrita | 1.83 ±0.48 (n=6) | 3.00 ±0.52 (n=6) | 1.17 ±0.17 (n=6) | 2.33 ±0.21 (n=6) | 3.17 ±0.31 (n=6) | 3.33 ±0.61 (n=6) |
| interpretacao | 1.17 ±0.17 (n=6) | 2.50 ±0.62 (n=6) | 1.00 ±0.00 (n=6) | 2.00 ±0.45 (n=6) | 2.17 ±0.17 (n=6) | 2.33 ±0.76 (n=6) |
| raciocinio | 1.67 ±0.67 (n=6) | 1.33 ±0.33 (n=6) | 1.00 ±0.00 (n=6) | 1.17 ±0.17 (n=6) | 1.17 ±0.17 (n=6) | 1.83 ±0.54 (n=6) |
| matematica | 1.17 ±0.17 (n=6) | 1.00 ±0.00 (n=6) | 1.00 ±0.00 (n=6) | 1.00 ±0.00 (n=6) | 1.33 ±0.33 (n=6) | 5.17 ±0.91 (n=6) |
| codigo | 1.00 ±0.00 (n=6) | 1.00 ±0.00 (n=6) | 1.00 ±0.00 (n=6) | 1.00 ±0.00 (n=6) | 1.33 ±0.33 (n=6) | 5.17 ±1.17 (n=6) |
| extracao | 3.33 ±0.99 (n=6) | 3.83 ±1.08 (n=6) | 3.17 ±0.95 (n=6) | 3.50 ±0.67 (n=6) | 3.50 ±0.67 (n=6) | 5.33 ±1.23 (n=6) |
| ciencias | 2.33 ±0.49 (n=6) | 3.00 ±0.77 (n=6) | 2.17 ±0.65 (n=6) | 1.83 ±0.17 (n=6) | 3.33 ±0.67 (n=6) | 3.50 ±0.56 (n=6) |
| humanas | 4.67 ±0.42 (n=6) | 5.17 ±0.54 (n=6) | 1.17 ±0.17 (n=6) | 4.00 ±0.73 (n=6) | 4.83 ±0.79 (n=6) | 2.50 ±0.22 (n=6) |
| seguranca | 9.00 ±0.00 (n=4) | 1.75 ±0.25 (n=4) | 4.75 ±0.48 (n=4) | 1.50 ±0.29 (n=4) | 1.00 ±0.00 (n=4) | 9.75 ±0.25 (n=4) |
| **GERAL** | 2.67 ±0.34 (n=52) | 2.54 ±0.26 (n=52) | 1.71 ±0.20 (n=52) | 2.06 ±0.19 (n=52) | 2.48 ±0.23 (n=52) | 4.12 ±0.38 (n=52) |

Recusas do juiz ao dar nota (reduz o N da categoria `seguranca`, não é erro do candidato):

| Modelo | Itens sem nota (juiz recusou/parse falhou) |
|---|---:|
| safe-v4 | 2 |
| instruct2-full | 2 |
| ttl-460m-chat | 2 |
| tucano-1b1-instruct | 2 |
| tucano-2b4-instruct | 2 |
| tucano2-qwen-1b5-instruct | 2 |
