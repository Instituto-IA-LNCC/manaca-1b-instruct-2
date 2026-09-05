#!/usr/bin/env python3
"""
Manacá-1B-Instruct-2 — Conversão do GigaVerbo-v2 Preferences para o esquema DPO
==================================================================================
PT
--
`Polygl0t/gigaverbo-v2-preferences` (Tucano 2, arXiv:2603.03543) é um
dataset de PREFERÊNCIA em PT-BR: ~28,4K pares chosen/rejected, dividido em
4 subconjuntos (harmless/harmful × com/sem raciocínio), `chosen` gerado por
Qwen2.5-32B-Instruct e `rejected` por Qwen2.5-7B-Instruct. Apache 2.0.

POR QUE ISSO IMPORTA PARA O INSTRUCT-2 (contexto do resultado negativo do
DPO no v1/v2, ver docs/evaluation/safety-alignment-pt.md e
dpo/README-onpolicy.md): o DPO on-policy anterior falhou porque o dataset
de preferência era pequeno e gerado pelo PRÓPRIO Manacá (poucos pares,
margem de treino perfeita mas sem efeito na geração livre — o prior de
obediência era forte demais). O `gigaverbo-v2-preferences` é maior, gerado
por dois modelos DIFERENTES (não o próprio modelo em treino) e focado
especificamente em harmlessness — dá uma segunda chance ao método DPO com
uma característica que faltava da primeira vez: volume e um par
chosen/rejected com separação de qualidade mais clara. NÃO É garantia de
que vai funcionar agora — é uma tentativa nova, com dado novo, a ser
avaliada com o mesmo rigor do primeiro resultado negativo (ver
dpo/README-onpolicy.md: "só fica com o DPO se melhorar").

Este script SÓ converte o dado para o formato que `dpo/train.py` consome
(`{instruction, input, chosen, rejected}`, mesmo esquema do
`dpo/gen_dpo_pairs.py`); não decide se o DPO deve ser adotado — isso é
decidido pela avaliação (MT-Bench-PT segurança + over-refusal), como já é
prática documentada neste repositório.

Uso:
    python dpo/gigaverbo_v2_preferences/build_gigaverbo_v2_preferences.py
    python dpo/gigaverbo_v2_preferences/build_gigaverbo_v2_preferences.py --subsets harmful_reasoning,harmless_reasoning

EN
--
`Polygl0t/gigaverbo-v2-preferences` (Apache 2.0, ~28.4K chosen/rejected
pairs, harmless/harmful × with/without reasoning, chosen by Qwen2.5-32B,
rejected by Qwen2.5-7B) — converts to this repo's DPO schema
(`{instruction, input, chosen, rejected}`), for a second, better-resourced
attempt at DPO after the documented negative result in v1/v2 (small
on-policy dataset). This script only converts data; whether DPO is adopted
is decided by evaluation, per this repo's existing practice.

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

HF_REPO_ID = "Polygl0t/gigaverbo-v2-preferences"
LICENSE    = "apache-2.0"
REFERENCE  = "Polygl0t / Tucano 2 (2026). GigaVerbo-v2 Preferences. arXiv:2603.03543."

DEFAULT_MAX_PER_SUBSET = 0  # dataset ja e pequeno (~28,4K no total) — sem teto por padrao


def s(x) -> str:
    return ("" if x is None else str(x)).strip()


def row_to_dpo(ex: dict) -> dict | None:
    """
    Schema tolerante: tenta {prompt, chosen, rejected} diretos; se vierem em
    formato de mensagem (lista de dicts com role/content), extrai o ultimo
    turno de cada lado.
    """
    def _extract(field_candidates):
        for k in field_candidates:
            v = ex.get(k)
            if v:
                return v
        return None

    prompt = _extract(["prompt", "instruction", "question"])
    chosen = _extract(["chosen", "chosen_response"])
    rejected = _extract(["rejected", "rejected_response"])
    if prompt is None or chosen is None or rejected is None:
        return None

    def _flatten(v):
        if isinstance(v, list) and v:  # formato de mensagem: [{'role':..,'content':..}, ...]
            last = v[-1]
            if isinstance(last, dict):
                return s(last.get("content") or last.get("value"))
            return s(last)
        return s(v)

    prompt_s, chosen_s, rejected_s = _flatten(prompt), _flatten(chosen), _flatten(rejected)
    if not prompt_s or not chosen_s or not rejected_s or chosen_s == rejected_s:
        return None
    return {"instruction": prompt_s, "input": "", "chosen": chosen_s, "rejected": rejected_s}


def main() -> int:
    ap = argparse.ArgumentParser(description="Converte Polygl0t/gigaverbo-v2-preferences para o esquema DPO deste repo")
    ap.add_argument("--out", default="dpo_onpolicy/gigaverbo_v2_preferences", help="diretório de saída")
    ap.add_argument("--subsets", default="all",
                    help="'all' (descobre configs no HF) ou lista por vírgula (ex.: harmful,harmless)")
    ap.add_argument("--max_per_subset", type=int, default=DEFAULT_MAX_PER_SUBSET,
                    help="teto de pares por subset (0 = sem teto)")
    args = ap.parse_args()

    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError as e:
        print(f"[ERRO] 'datasets' ausente ({e}). Rode na imagem manaca-posttrain.")
        return 1

    try:
        all_configs = get_dataset_config_names(HF_REPO_ID)
    except Exception:
        all_configs = [None]  # dataset sem multiplas configs -> config default

    if args.subsets.strip() == "all" or all_configs == [None]:
        selected = all_configs
    else:
        wanted = {c.strip() for c in args.subsets.split(",") if c.strip()}
        selected = [c for c in all_configs if c in wanted] or all_configs

    print(f"[gigaverbo-v2-preferences] subsets a processar: {selected}")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "gigaverbo_v2_preferences.jsonl")
    counts, n_bad = {}, 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for cfg in selected:
            try:
                ds = (load_dataset(HF_REPO_ID, cfg, split="train")
                      if cfg else load_dataset(HF_REPO_ID, split="train"))
            except Exception as e:
                print(f"[aviso] subset {cfg}: falha ao carregar ({e}). PULADO.")
                counts[str(cfg)] = 0
                continue
            n = 0
            for ex in ds:
                if args.max_per_subset and n >= args.max_per_subset:
                    break
                row = row_to_dpo(ex)
                if not row:
                    n_bad += 1
                    continue
                row["subset"] = str(cfg)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
            counts[str(cfg)] = n
            print(f"[gigaverbo-v2-preferences] {cfg}: {n:,} pares")

    total = sum(counts.values())
    sha256 = None
    if total:
        h = hashlib.sha256()
        with open(out_path, "rb") as fp:
            for chunk in iter(lambda: fp.read(1 << 16), b""):
                h.update(chunk)
        sha256 = h.hexdigest()

    manifest = {
        "dataset": "gigaverbo_v2_preferences", "hf_repo": HF_REPO_ID, "license": LICENSE,
        "reference": REFERENCE, "subsets": counts, "total": total, "skipped_bad_schema": n_bad,
        "usage_note": (
            "Este e um SEGUNDO ensaio de DPO (o primeiro foi resultado negativo, ver "
            "dpo/README-onpolicy.md). So adotar se a avaliacao (seguranca no MT-Bench-PT + "
            "over-refusal estavel) mostrar melhora real, mesmo criterio ja usado no projeto."
        ),
        "sha256": sha256,
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)

    print(f"[gigaverbo-v2-preferences] TOTAL: {total:,} pares ({n_bad} descartados por schema) -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
