#!/usr/bin/env python3
"""
Manacá-1B-Instruct-2 — Conversão do Tucano-SFT (TucanoBR) para o esquema Alpaca
==================================================================================
PT
--
`TucanoBR/Tucano-SFT` é o dataset de instrução usado para treinar os
próprios Tucano-Instruct (1b1/2b4). É a concatenação de três fontes já
verificadas individualmente:

  - `cnmoro/GPT4-500k-Augmented-PTBR-Clean` — MIT
  - `rhaymison/orca-math-portuguese-64k`     — Apache-2.0
  - `nicholasKluge/instruct-aira-dataset-v3` — Apache-2.0 (conversas do projeto Aira)

Todas licenças permissivas confirmadas individualmente no HuggingFace.

AVISO DE PROVENIÊNCIA (transparência, não um bloqueio): a fonte
"GPT4-500k-Augmented" foi gerada consultando um modelo GPT-4 da OpenAI —
os termos de uso da OpenAI restringem usar saídas do modelo para treinar
modelos concorrentes. Isso é uma tensão conhecida e amplamente tolerada na
comunidade de pesquisa aberta (a própria linhagem Alpaca, já usada no
corpus v1/v2 deste projeto, tem a mesma característica — instruções
geradas por um modelo proprietário). Registramos aqui pelo mesmo motivo
que o projeto já documenta a licença não-comercial do Alpaca-PT: auditoria
honesta, não omissão.

NÃO baixamos as 3 fontes separadamente (evita duplicar exemplos): o
`TucanoBR/Tucano-SFT` já é a união pronta.

Papel neste repositório: mais instrução PT-BR (nativa + traduzida) pronta,
sem custo de geração — mesma categoria do `sft/gigaverbo_v2_sft/`.

Saída: `{instruction, input, output, task}` (esquema Alpaca), consumido via
`--extra` de `sft/prepare_data.py`, igual ao gigaverbo_v2_sft/manacá-jaster.

Uso:
    python sft/tucano_sft/build_tucano_sft.py
    python sft/tucano_sft/build_tucano_sft.py --max_examples 0   # sem teto

EN
--
`TucanoBR/Tucano-SFT` is the instruction dataset used to train Tucano-
Instruct itself (concatenation of 3 individually-verified permissively
licensed sources — see above). This script converts it to this repo's
Alpaca schema, tolerant to either a chat/messages format or an
already-flat prompt/completion format, with the same best-effort
decontamination against this repo's static benchmark/seed files used in
`sft/gigaverbo_v2_sft/build_gigaverbo_v2_sft.py`.

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import unicodedata

HF_REPO_ID = "TucanoBR/Tucano-SFT"
HF_SPLIT   = "train"
LICENSE    = "apache-2.0 / mit (mix de subfontes ja verificadas — ver docstring)"
REFERENCE  = "Correa et al. (2024). Tucano. TucanoBR/Tucano-SFT (HuggingFace)."

DEFAULT_MAX_EXAMPLES = 50_000  # ver nota de escopo: essa fonte sozinha ja seria
                                # > 1/3 do total do v2 atual (141k) sem teto

DECONTAMINATION_SOURCES = [
    ("bench/ifeval_pt/prompts.jsonl", "turns"),
    ("bench/mtbench_pt/questions.jsonl", "turns"),
    ("bench/safety_pt/heldout.jsonl", "prompt"),
    ("bench/overrefusal_pt/prompts.jsonl", "prompt"),
    ("dpo/dpo_seeds/safety_prompts.jsonl", "prompt"),
    ("sft/safety_seeds/benign_help.jsonl", "prompt"),
]


def s(x) -> str:
    return ("" if x is None else str(x)).strip()


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.strip().lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text)


def load_decontamination_set(repo_root: str) -> set[str]:
    seen: set[str] = set()
    for rel_path, field in DECONTAMINATION_SOURCES:
        path = os.path.join(repo_root, rel_path)
        if not os.path.exists(path):
            print(f"[decontam] aviso: {rel_path} não encontrado — pulando.")
            continue
        n = 0
        with open(path, encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                val = row.get(field)
                text = val[0] if isinstance(val, list) and val else val
                text = s(text)
                if text:
                    seen.add(_normalize(text))
                    n += 1
        print(f"[decontam] {rel_path}: {n} prompts carregados")
    return seen


def is_contaminated(instruction: str, decontam_set: set[str]) -> bool:
    if not decontam_set:
        return False
    norm = _normalize(instruction)
    if norm in decontam_set:
        return True
    return any(norm in bench or bench in norm for bench in decontam_set if len(bench) > 20)


def _extract_role_content(msg) -> tuple[str, str]:
    if not isinstance(msg, dict):
        return "", ""
    if "role" in msg:
        return s(msg.get("role")).lower(), s(msg.get("content"))
    if "from" in msg:
        role_raw = s(msg.get("from")).lower()
        role = {"human": "user", "gpt": "assistant", "bot": "assistant"}.get(role_raw, role_raw)
        return role, s(msg.get("value"))
    return "", ""


def messages_to_alpaca(messages: list) -> dict | None:
    """Dobra conversa multi-turno em Alpaca (idêntico ao gigaverbo_v2_sft)."""
    if not messages or not isinstance(messages, list):
        return None
    parsed = [(_extract_role_content(m)) for m in messages]
    parsed = [(r, c) for r, c in parsed if r and c]
    if len(parsed) < 2 or parsed[-1][0] != "assistant":
        return None
    output = parsed[-1][1]
    last_user_idx = None
    for i in range(len(parsed) - 2, -1, -1):
        if parsed[i][0] == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        return None
    instruction = parsed[last_user_idx][1]
    prefix = {"system": "Sistema", "user": "Usuário", "assistant": "Assistente"}
    prior = parsed[:last_user_idx]
    input_text = "\n".join(f"{prefix.get(r, r.capitalize())}: {c}" for r, c in prior) if prior else ""
    return {"instruction": s(instruction), "input": s(input_text), "output": s(output)}


def row_to_alpaca(ex: dict, message_col: str | None) -> dict | None:
    """Schema tolerante: chat/messages OU já {instruction/prompt, output/completion}."""
    if message_col:
        return messages_to_alpaca(ex.get(message_col))
    instr = s(ex.get("instruction") or ex.get("prompt") or ex.get("question"))
    out = s(ex.get("output") or ex.get("completion") or ex.get("response") or ex.get("answer"))
    inp = s(ex.get("input") or ex.get("context"))
    if not instr or not out:
        return None
    return {"instruction": instr, "input": inp, "output": out}


def main() -> int:
    ap = argparse.ArgumentParser(description="Converte TucanoBR/Tucano-SFT para Alpaca {instruction,input,output}")
    ap.add_argument("--out", default="data/sft/tucano_sft", help="diretório de saída")
    ap.add_argument("--max_examples", type=int, default=DEFAULT_MAX_EXAMPLES,
                    help=f"teto de exemplos (default {DEFAULT_MAX_EXAMPLES:,}; 0 = sem teto)")
    ap.add_argument("--max_chars", type=int, default=5000,
                    help="descarta exemplos cujo prompt passa deste nº de caracteres (0 = off)")
    ap.add_argument("--repo_root", default=".", help="raiz do repo (para decontaminacao)")
    ap.add_argument("--no_decontaminate", action="store_true", help="pula a decontaminacao (NAO recomendado)")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as e:
        print(f"[ERRO] 'datasets' ausente ({e}). Rode na imagem manaca-posttrain.")
        return 1

    decontam_set: set[str] = set()
    if not args.no_decontaminate:
        decontam_set = load_decontamination_set(args.repo_root)
        print(f"[decontam] total: {len(decontam_set)} prompts normalizados")

    print(f"[tucano-sft] carregando {HF_REPO_ID} (streaming)...")
    ds = load_dataset(HF_REPO_ID, split=HF_SPLIT, streaming=True)
    first = next(iter(ds))
    message_col = next((c for c in ["messages", "conversations", "conversation", "chat"] if c in first), None)
    print(f"[tucano-sft] modo: {'chat/' + message_col if message_col else 'prompt-completion direto'}"
          f" | campos: {list(first.keys())}")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "tucano_sft.jsonl")
    n, n_contam, n_long, n_bad = 0, 0, 0, 0

    def emit(ex):
        nonlocal n, n_contam, n_long, n_bad
        row = row_to_alpaca(ex, message_col)
        if not row or not row["instruction"] or not row["output"]:
            n_bad += 1
            return False
        if is_contaminated(row["instruction"], decontam_set):
            n_contam += 1
            return False
        if args.max_chars and (len(row["instruction"]) + len(row["input"])) > args.max_chars:
            n_long += 1
            return False
        row["task"] = "tucano_sft"
        fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
        return True

    with open(out_path, "w", encoding="utf-8") as fp:
        for ex in ds:
            if args.max_examples and n >= args.max_examples:
                break
            emit(ex)

    if args.shuffle:
        with open(out_path, encoding="utf-8") as fp:
            lines = fp.readlines()
        random.Random(args.seed).shuffle(lines)
        with open(out_path, "w", encoding="utf-8") as fp:
            fp.writelines(lines)

    sha256 = None
    if n:
        h = hashlib.sha256()
        with open(out_path, "rb") as fp:
            for chunk in iter(lambda: fp.read(1 << 16), b""):
                h.update(chunk)
        sha256 = h.hexdigest()

    manifest = {
        "dataset": "tucano_sft", "hf_repo": HF_REPO_ID, "license": LICENSE, "reference": REFERENCE,
        "total": n, "skipped_bad_schema": n_bad, "skipped_contaminated": n_contam,
        "skipped_too_long": n_long, "max_examples": args.max_examples,
        "decontaminated": not args.no_decontaminate,
        "decontamination_note": (
            "Best-effort contra bench/dpo/sft/safety_seeds ESTATICOS deste repo. "
            "NAO cobre ENEM/BLUEX/OAB/ASSIN2/HateBR (carregados via HF pelo lm-eval)."
        ),
        "provenance_note": (
            "Uma das 3 subfontes (GPT4-500k-Augmented-PTBR-Clean) foi gerada via GPT-4 da "
            "OpenAI -- ver aviso de proveniencia no docstring deste script."
        ),
        "shuffled": args.shuffle, "seed": args.seed, "sha256": sha256,
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)

    print(f"[tucano-sft] TOTAL: {n:,} exemplos (descartados: {n_bad} schema, {n_contam} contaminados, "
          f"{n_long} longos) -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
