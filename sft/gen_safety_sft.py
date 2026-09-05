#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manacá-1B — Construtor do dataset de SAFETY-SFT (estilo AnswerCarefully)
=======================================================================
O DPO gentil (beta 0.1) não reverteu o prior de obediência do modelo — nem nos
prompts de treino (ver dpo/diag_dpo.py). A segurança do LLM-jp veio de SFT de
segurança (AnswerCarefully), não de DPO: recusa como ALVO de cross-entropy muda
a geração de verdade. Aqui montamos esse dataset, balanceado para NÃO virar
recusador cego:

  (1) RECUSAR o nocivo   — prompts nocivos (dpo_seeds) -> recusa (K variantes).
  (2) AJUDAR o benigno   — prompts benignos/sensíveis-legítimos -> resposta útil
                           gerada pelo PRÓPRIO SFT (on-policy; ele já responde bem).
  (3) DADOS GERAIS       — amostra do SFT geral, para preservar a utilidade ampla.

Saída {instruction, input, output} no formato do sft/train.py. Como o output é
sempre recusa OU resposta útil (nunca conteúdo nocivo), o arquivo PODE ser
versionado (transparência).

Uso (veja scripts/run_gen_safety_sft.sh):
    python sft/gen_safety_sft.py --model /m --out sft_safety/safety_sft.jsonl \\
        --general_data data/sft_v2/manaca_sft.jsonl --n_general 200

Autor: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "dpo"))
from gen_dpo_pairs import montar_prompt  # noqa: E402  (mesmo template Alpaca-PT)

# Banco de recusas DIVERSO (para o modelo aprender o COMPORTAMENTO de recusar,
# não uma string só). Generaliza melhor que 2-3 templates fixos.
RECUSAS = [
    "Desculpe, mas não posso ajudar com isso. O pedido envolve uma atividade "
    "ilegal ou que pode causar dano a outras pessoas. Se você tiver um objetivo "
    "legítimo por trás disso, posso sugerir um caminho seguro e dentro da lei.",
    "Não vou ajudar com esse pedido, porque ele pode causar dano ou é ilegal. "
    "Se quiser, posso explicar os riscos envolvidos ou indicar alternativas legítimas.",
    "Sinto muito, mas isso é algo com que não posso ajudar. Envolve risco de dano "
    "ou quebra da lei. Posso te ajudar com uma alternativa segura, se você quiser.",
    "Não posso ajudar com isso. Fornecer esse tipo de informação poderia facilitar "
    "um crime ou machucar alguém. Se o seu objetivo for legítimo, me conte mais que "
    "eu tento ajudar de outra forma.",
    "Isso foge do que eu posso fazer. O que você pediu pode prejudicar pessoas ou "
    "infringir a lei, então não vou seguir com isso. Posso ajudar com algo relacionado "
    "e seguro?",
    "Prefiro não ajudar com esse pedido: ele pode causar dano a terceiros ou é "
    "ilegal. Se você quiser entender o assunto do ponto de vista da prevenção ou da "
    "segurança, aí sim posso ajudar.",
    "Não consigo ajudar com isso, porque envolve risco de dano ou atividade ilegal. "
    "Se estiver enfrentando uma situação difícil, posso sugerir caminhos legítimos "
    "ou a quem recorrer.",
    "Lamento, mas não vou fornecer isso. É um pedido que pode ferir alguém ou "
    "quebrar a lei. Estou aqui para ajudar em coisas seguras e legais.",
]
RECUSA_AUTOLESAO = [
    "Sinto muito que você esteja passando por isso, e obrigado por confiar em mim. "
    "Não vou ajudar com isso, mas você não está sozinho. Se estiver em sofrimento, "
    "ligue para o CVV no número 188 (24h, gratuito) ou acesse cvv.org.br. Falar com "
    "alguém de confiança ou um profissional de saúde pode ajudar muito.",
    "Percebo que você pode estar num momento muito difícil. Não posso ajudar com "
    "esse pedido, mas me importo com a sua segurança. O CVV atende de graça, 24h, "
    "no 188 (ou cvv.org.br). Você não precisa passar por isso sozinho.",
]


def escolher_recusa(categoria, rng):
    if categoria == "autolesao":
        return rng.choice(RECUSA_AUTOLESAO)
    return rng.choice(RECUSAS)


def carregar_jsonl(caminho):
    with open(caminho, encoding="utf-8") as fp:
        return [json.loads(l) for l in fp if l.strip()]


def gerar_util(tok, model, instr, max_new_tokens):
    """Resposta útil gerada pelo próprio SFT (greedy) para um prompt benigno."""
    import torch
    dev = next(model.parameters()).device
    enc = tok(montar_prompt(instr), return_tensors="pt").to(dev)
    mi = {"input_ids": enc["input_ids"]}
    if "attention_mask" in enc:
        mi["attention_mask"] = enc["attention_mask"]
    with torch.inference_mode():
        out = model.generate(**mi, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][mi["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="Monta o dataset de safety-SFT (AnswerCarefully-style)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default=os.path.join(REPO, "dpo/dpo_seeds/safety_prompts.jsonl"))
    ap.add_argument("--benign", default=os.path.join(REPO, "sft/safety_seeds/benign_help.jsonl"),
                    help="prompts benignos/sensíveis-legítimos p/ o contraste 'ajude' "
                         "(DISJUNTO da avaliacao bench/overrefusal_pt para nao contaminar)")
    ap.add_argument("--k_refuse", type=int, default=2, help="variantes de recusa por prompt nocivo")
    ap.add_argument("--model", default=None, help="SFT p/ gerar as respostas úteis (benign-help); sem ele, pula")
    ap.add_argument("--general_data", default=None, help="jsonl do SFT geral p/ misturar (preserva utilidade)")
    ap.add_argument("--n_general", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    registros = []

    # (1) RECUSAR o nocivo ----------------------------------------------------
    seeds = carregar_jsonl(args.seeds)
    for p in seeds:
        cat = p.get("categoria", "default")
        vistos = set()
        for _ in range(args.k_refuse):
            r = escolher_recusa(cat, rng)
            if r in vistos:  # evita duplicar a mesma variante no mesmo prompt
                continue
            vistos.add(r)
            registros.append({"instruction": p["prompt"], "input": "", "output": r,
                              "meta": {"tipo": "recusa", "categoria": cat}})
    n_recusa = len(registros)
    print(f"[safety-sft] recusas: {n_recusa} (de {len(seeds)} prompts x ate {args.k_refuse} variantes)")

    # (2) AJUDAR o benigno ------------------------------------------------------
    # Duas fontes no mesmo arquivo (--benign), distinguidas pela presença de
    # `output`: (a) resposta PRONTA (gold, escrita e revisada a mão — ex.: a
    # categoria "ajudar_em_crise", ver sft/README-safety.md §4) usada direto,
    # não precisa de --model nem de geração; (b) resposta ON-POLICY, gerada
    # pelo próprio SFT (o resto do benign_help.jsonl, sem `output`), que
    # continua exigindo --model. Isso existe porque para pedidos sensíveis
    # como "ajudar amigo em crise" não dá pra confiar que o SFT sem alinhamento
    # já sabe mencionar o CVV sozinho — foi exatamente esse o caso documentado
    # (`sens_12`) em que o safe4 recusava por engano.
    benignos = carregar_jsonl(args.benign)
    prontos = [p for p in benignos if p.get("output")]
    on_policy = [p for p in benignos if not p.get("output")]

    n_ajuda_pronta = 0
    for p in prontos:
        registros.append({"instruction": p["prompt"], "input": "", "output": p["output"],
                          "meta": {"tipo": "ajuda_pronta", "origem": p.get("tipo", "benigno")}})
        n_ajuda_pronta += 1
    if prontos:
        print(f"[safety-sft] ajudas (resposta pronta/gold, ex.: ajudar_em_crise): {n_ajuda_pronta}")

    n_ajuda_onpolicy = 0
    if args.model and on_policy:
        sys.path.insert(0, os.path.join(REPO, "dpo"))
        from gen_dpo_pairs import carregar_modelo
        tok, model, _ = carregar_modelo(args.model, None, args.attn, args.dtype)
        for p in on_policy:
            resp = gerar_util(tok, model, p["prompt"], args.max_new_tokens)
            if not resp:
                continue
            registros.append({"instruction": p["prompt"], "input": "", "output": resp,
                              "meta": {"tipo": "ajuda_onpolicy", "origem": p.get("tipo", "benigno")}})
            n_ajuda_onpolicy += 1
        print(f"[safety-sft] ajudas (benign-help on-policy): {n_ajuda_onpolicy}")
    elif on_policy:
        print(f"[safety-sft] sem --model: pulando {len(on_policy)} exemplos on-policy "
              "(risco maior de over-refusal)")

    n_ajuda = n_ajuda_pronta + n_ajuda_onpolicy

    # (3) DADOS GERAIS (amostra do SFT geral) ---------------------------------
    n_geral = 0
    if args.general_data and os.path.isfile(args.general_data):
        geral = carregar_jsonl(args.general_data)
        rng.shuffle(geral)
        for ex in geral[:args.n_general]:
            registros.append({"instruction": ex.get("instruction", ""), "input": ex.get("input", ""),
                              "output": ex.get("output", ""), "meta": {"tipo": "geral"}})
            n_geral += 1
        print(f"[safety-sft] gerais: {n_geral} (amostra de {len(geral)})")
    else:
        print("[safety-sft] sem --general_data: treine com cuidado (misturar geral evita over-refusal)")

    rng.shuffle(registros)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out:
        for r in registros:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[safety-sft] total={len(registros)} (recusa={n_recusa} ajuda={n_ajuda} geral={n_geral}) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
