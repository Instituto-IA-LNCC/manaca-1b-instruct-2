#!/usr/bin/env python3
"""
Manacá-1B-Instruct-2 — MedPT: QA médico real e nativo em PT-BR
=========================================================================
PT
--
`AKCIT/MedPT` (D'Addario et al., 2025, arXiv:2511.11878, aceito na LREC
2026) é o maior corpus público de perguntas-e-respostas médicas REAIS e
NATIVAS em português do Brasil: 384.095 pares extraídos de interações
reais paciente-médico, cobrindo 3.200+ condições e 104 especialidades.
Licença CC BY 4.0.

POR QUE ISSO PREENCHE UMA LACUNA REAL: o projeto tem corpus jurídico
(LegalPT), enciclopédico (Wikipedia/Carolina) e crawl genérico
(ClassiCC/MADLAD/FineWeb2/HPLT2), mas nenhum QA nativo de domínio —
muito menos saúde, onde tradução do inglês perde nuance clínica e cultural
(doenças endêmicas, gírias regionais). Diferente do `synth/
generate_instructions.py`, aqui a resposta já É a resposta real de um
profissional — não precisa (nem deve) ser gerada por Claude; é conversão
determinística, no mesmo espírito do `sft/quati` e das tarefas do
`sft/jaster`.

RESSALVA DE LICENÇA (auditar antes de usar em escala): a licença CC-BY-4.0
foi confirmada via busca (arXiv + página de paper do HF), mas o acesso
direto à ficha do dataset no HuggingFace estava bloqueado no ambiente em
que este script foi escrito — o script IMPRIME o campo de licença
reportado pelos metadados do HF em tempo de execução (quando disponível)
para dupla-checagem no momento real do download.

FILTRO DE QUALIDADE: descarta pares com resposta muito curta (< 20
caracteres, sinal de resposta tipo "Vá ao médico" sem conteúdo) ou muito
longa (> 4000, provável erro de parsing), e feito o mesmo filtro
alfabético básico (`passes_sanity`) usado em outros scripts do repo.

Uso:
    python sft/medpt/build_medpt.py
    python sft/medpt/build_medpt.py --max_examples 100000 --shuffle

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

HF_REPO_ID = "AKCIT/MedPT"
LICENSE    = "cc-by-4.0"  # confirmado via arXiv/paper-page do HF; reconfirmar na ficha ao rodar
REFERENCE  = "D'Addario et al. (2025). MedPT. arXiv:2511.11878 (LREC 2026)."

DEFAULT_MAX_EXAMPLES = 0  # 0 = sem teto (dataset ja e um corte curado, 384k)
MIN_ANSWER_LEN = 20
MAX_ANSWER_LEN = 4000


def s(x) -> str:
    return ("" if x is None else str(x)).strip()


def passes_sanity(text: str, min_len: int, max_len: int) -> bool:
    if not text or len(text) < min_len or len(text) > max_len:
        return False
    n_alpha = sum(1 for c in text if c.isalpha())
    return n_alpha / max(len(text), 1) >= 0.4


def main() -> int:
    ap = argparse.ArgumentParser(description="Converte AKCIT/MedPT (QA medico real PT-BR) para Alpaca")
    ap.add_argument("--out", default="data/sft/medpt", help="diretorio de saida")
    ap.add_argument("--max_examples", type=int, default=DEFAULT_MAX_EXAMPLES,
                    help="teto de exemplos (0 = sem teto)")
    ap.add_argument("--min_answer_len", type=int, default=MIN_ANSWER_LEN)
    ap.add_argument("--max_answer_len", type=int, default=MAX_ANSWER_LEN)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as e:
        print(f"[ERRO] 'datasets' ausente ({e}). Rode na imagem manaca-posttrain.")
        return 1

    try:
        from huggingface_hub import dataset_info
        info = dataset_info(HF_REPO_ID)
        card = getattr(info, "card_data", None)
        declared = card.get("license") if hasattr(card, "get") else getattr(card, "license", None)
        print(f"[medpt] licenca declarada nos metadados do HF: "
              f"{declared or '(nao encontrada nos metadados — confira a ficha manualmente)'}")
    except Exception as e:
        print(f"[medpt] aviso: nao consegui checar a licenca via API do HF ({e}). "
              f"CONFIRME manualmente em https://huggingface.co/datasets/{HF_REPO_ID} antes de usar em escala.")

    print(f"[medpt] carregando {HF_REPO_ID} ...")
    try:
        ds = load_dataset(HF_REPO_ID, split="train")
    except Exception as e:
        print(f"[ERRO] falha ao carregar {HF_REPO_ID}: {e}")
        return 1

    cols = ds.column_names
    q_col = next((c for c in ["question", "pergunta", "query", "patient_question"] if c in cols), None)
    a_col = next((c for c in ["answer", "resposta", "response", "doctor_answer"] if c in cols), None)
    cond_col = next((c for c in ["condition", "disease", "doenca", "condicao"] if c in cols), None)
    spec_col = next((c for c in ["specialty", "especialidade"] if c in cols), None)
    print(f"[medpt] colunas detectadas: pergunta='{q_col}' resposta='{a_col}' "
          f"condicao='{cond_col}' especialidade='{spec_col}' | todas: {cols}")
    if not q_col or not a_col:
        print(f"[ERRO] colunas de pergunta/resposta nao reconhecidas em {cols}")
        return 1

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "medpt.jsonl")
    n, n_short, n_long = 0, 0, 0
    spec_dist: dict[str, int] = {}
    with open(out_path, "w", encoding="utf-8") as fp:
        for ex in ds:
            if args.max_examples and n >= args.max_examples:
                break
            q, a = s(ex.get(q_col)), s(ex.get(a_col))
            if not q:
                continue
            if not passes_sanity(a, args.min_answer_len, args.max_answer_len):
                if len(a) < args.min_answer_len:
                    n_short += 1
                else:
                    n_long += 1
                continue
            row = {"instruction": q, "input": "", "output": a, "task": "medpt_qa"}
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            if spec_col:
                spec = s(ex.get(spec_col)) or "desconhecida"
                spec_dist[spec] = spec_dist.get(spec, 0) + 1

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

    top_specialties = dict(sorted(spec_dist.items(), key=lambda kv: -kv[1])[:15])
    manifest = {
        "dataset": "medpt_qa", "hf_repo": HF_REPO_ID, "license": LICENSE, "reference": REFERENCE,
        "license_note": "Confirmado via busca/arXiv, nao via ficha do HF direto -- reconfirmar ao rodar (ver log).",
        "total": n, "skipped_short_answer": n_short, "skipped_long_answer": n_long,
        "top_specialties": top_specialties, "shuffled": args.shuffle, "seed": args.seed, "sha256": sha256,
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)

    print(f"[medpt] TOTAL: {n:,} exemplos ({n_short:,} descartados curtos, {n_long:,} descartados longos) -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
