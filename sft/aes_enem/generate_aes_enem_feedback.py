#!/usr/bin/env python3
"""
Manacá-1B-Instruct-2 — Feedback de redação ENEM grounded na nota real (via Claude)
=====================================================================================
PT
--
`kamel-usp/aes_enem_dataset` (MIT) tem redações REAIS de estudantes
brasileiros no formato ENEM (dissertativo-argumentativo) com notas
oficiais atribuídas por avaliadores humanos, por competência (C1-C5,
0-200 cada, total 0-1000). O manacá-jaster já cobre ENEM só como múltipla
escolha (`conv_enem`); isso ensina uma capacidade diferente: avaliar e dar
FEEDBACK qualitativo sobre um texto dissertativo.

POR QUE PRECISA DO CLAUDE (diferente de MedPT/Aya/HealthQA-BR, que são
conversão direta): o dataset NÃO tem feedback textual, só notas numéricas
por competência. Uma nota sozinha ("C1: 160, C2: 120...") não ensina o
modelo a EXPLICAR o porquê. O Claude gera a justificativa qualitativa,
mas GROUNDED nas notas reais (nunca inventa uma nota diferente da real) --
mesmo princípio de "não invente além do que foi dado" do
`synth/generate_instructions.py`.

SCHEMA INCERTO (o dataset tem múltiplos configs com nomes de coluna
possivelmente diferentes -- JBCS2025, PROPOR2024, sourceAWithGraders,
sourceB): o script descobre as configs dinamicamente e tenta detectar
colunas de texto e nota de forma tolerante (nota única 0-1000, ou 5 notas
por competência C1-C5). Reporta no manifest o que foi de fato encontrado
-- auditar antes de confiar em escala.

Uso:
    export ANTHROPIC_API_KEY=sk-ant-...
    python sft/aes_enem/generate_aes_enem_feedback.py --dry-run
    python sft/aes_enem/generate_aes_enem_feedback.py --config sourceAWithGraders --max_examples 500

Saída: `data/sft/aes_enem/aes_enem_feedback.jsonl` (Alpaca:
`{instruction, input, output, task}`) + manifest com custo estimado.

EN
--
kamel-usp/aes_enem_dataset (MIT) has real Brazilian ENEM-style essays with
official human-assigned competency grades but no qualitative feedback.
Uses Claude to generate feedback grounded in the REAL grades (never
inventing a different score) -- teaches essay evaluation/feedback, a
capability distinct from the multiple-choice ENEM task already in
manacá-jaster.

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

HF_REPO_ID = "kamel-usp/aes_enem_dataset"
LICENSE    = "mit"
REFERENCE  = "kamel-usp (2025/2026). AES-ENEM Dataset."

DEFAULT_MODEL = os.environ.get("MODEL", "claude-opus-5")
DEFAULT_MAX_TOKENS = 1500
DEFAULT_MAX_EXAMPLES = 500

PRICING_SNAPSHOT = {
    "claude-opus-5":    (5.00, 25.00),
    "claude-sonnet-5":  (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

COMPETENCIAS = [
    "C1 (domínio da norma culta)", "C2 (compreensão do tema e uso de conhecimentos)",
    "C3 (seleção e organização de argumentos)", "C4 (mecanismos linguísticos de coesão)",
    "C5 (proposta de intervenção)",
]

INSTRUCTION = ("Avalie a redação a seguir segundo os critérios do ENEM (5 competências, "
               "0 a 200 pontos cada) e justifique a nota de cada competência.")

SYSTEM_PROMPT = """Você escreve o FEEDBACK qualitativo de uma correção de redação do ENEM que \
já foi avaliada por um corretor humano real. As notas por competência (C1 a C5, 0-200 cada) \
JÁ FORAM DADAS -- sua única tarefa é escrever a JUSTIFICATIVA de cada nota, com base no texto \
da redação. NÃO invente uma nota diferente da fornecida; use as notas exatamente como estão.

Regras obrigatórias:
- Para cada uma das 5 competências, escreva 1-3 frases explicando por que a redação recebeu \
aquela nota especificamente (pontos fortes E fracos observados no texto).
- Cite trechos ou características concretas da redação (não seja genérico).
- Termine com a nota total (soma das 5 competências) e um resumo de 1 frase.
- Responda em português do Brasil, em texto corrido (não JSON)."""


def s(x) -> str:
    return ("" if x is None else str(x)).strip()


def detect_grade_columns(first_row: dict) -> tuple[str | None, list[str] | None]:
    """Retorna (coluna_nota_unica, [colunas_c1..c5]) -- o que existir."""
    single_candidates = ["score", "nota", "nota_final", "grade", "total_score", "final_grade"]
    single = next((c for c in single_candidates if c in first_row), None)
    comp_patterns = [
        ["c1", "c2", "c3", "c4", "c5"],
        ["comp1", "comp2", "comp3", "comp4", "comp5"],
        ["competencia1", "competencia2", "competencia3", "competencia4", "competencia5"],
        ["nota_c1", "nota_c2", "nota_c3", "nota_c4", "nota_c5"],
        ["grades_1", "grades_2", "grades_3", "grades_4", "grades_5"],
    ]
    comp_cols = next((p for p in comp_patterns if all(c in first_row for c in p)), None)
    if not comp_cols and "grades" in first_row:
        # 'grades' pode ser uma lista/dict com as 5 notas
        val = first_row.get("grades")
        if isinstance(val, (list, tuple)) and len(val) == 5:
            comp_cols = "grades_list"  # sentinela: tratado especial no loop
    return single, comp_cols


def build_grades_text(ex: dict, single_col: str | None, comp_cols) -> str | None:
    if comp_cols == "grades_list":
        vals = ex.get("grades")
        if not (isinstance(vals, (list, tuple)) and len(vals) == 5):
            return None
        return "\n".join(f"{COMPETENCIAS[i]}: {vals[i]}" for i in range(5))
    if comp_cols:
        vals = [ex.get(c) for c in comp_cols]
        if any(v is None for v in vals):
            return None
        return "\n".join(f"{COMPETENCIAS[i]}: {vals[i]}" for i in range(5))
    if single_col and ex.get(single_col) is not None:
        return f"Nota total: {ex.get(single_col)} (de 0 a 1000)"
    return None


def call_with_retry(client, max_retries=5, base_delay=1.0, **kwargs):
    import anthropic
    import random as _random
    last_exc = None
    for attempt in range(max_retries):
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            last_exc = e
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                last_exc = e
            else:
                raise
        delay = min(base_delay * (2 ** attempt) + _random.uniform(0, 1), 30.0)
        print(f"[retry] tentativa {attempt + 1}/{max_retries} apos {delay:.1f}s")
        time.sleep(delay)
    raise last_exc


def main() -> int:
    ap = argparse.ArgumentParser(description="Gera feedback de redacao ENEM via Claude, grounded na nota real")
    ap.add_argument("--out", default="data/sft/aes_enem", help="diretorio de saida")
    ap.add_argument("--config", default=None, help="config do HF (default: descobre e usa a 1a disponivel)")
    ap.add_argument("--max_examples", type=int, default=DEFAULT_MAX_EXAMPLES)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="mostra o prompt de sistema e o 1o exemplo, sem chamar a API")
    args = ap.parse_args()

    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError as e:
        print(f"[ERRO] 'datasets' ausente ({e}). Rode na imagem manaca-posttrain.")
        return 1

    if args.config:
        cfg = args.config
    else:
        try:
            configs = get_dataset_config_names(HF_REPO_ID)
        except Exception as e:
            print(f"[ERRO] falha ao descobrir configs de {HF_REPO_ID}: {e}")
            return 1
        if not configs:
            print(f"[ERRO] nenhuma config encontrada em {HF_REPO_ID}")
            return 1
        cfg = configs[0]
        print(f"[aes-enem] configs disponiveis: {configs} | usando: {cfg}")

    try:
        ds = load_dataset(HF_REPO_ID, cfg, split="train")
    except Exception as e:
        print(f"[ERRO] falha ao carregar {HF_REPO_ID} ({cfg}): {e}")
        return 1

    cols = ds.column_names
    essay_col = next((c for c in ["essay", "text", "redacao", "essay_text", "corpus"] if c in cols), None)
    if not essay_col:
        print(f"[ERRO] coluna de redacao nao reconhecida em {cols}")
        return 1
    first_row = ds[0]
    single_col, comp_cols = detect_grade_columns(first_row)
    print(f"[aes-enem] config={cfg} | coluna redacao='{essay_col}' | "
          f"nota unica='{single_col}' | competencias='{comp_cols}' | colunas: {cols}")
    if not single_col and not comp_cols:
        print(f"[ERRO] nenhuma coluna de nota reconhecida em {cols}. "
              "Ajuste detect_grade_columns() apos auditar a ficha real.")
        return 1

    if args.dry_run:
        grades_text = build_grades_text(first_row, single_col, comp_cols)
        print("\n[dry-run] nenhuma chamada de API sera feita.\n")
        print("=" * 60)
        print("SYSTEM PROMPT:\n" + SYSTEM_PROMPT)
        print("=" * 60)
        print(f"1o EXEMPLO — redacao (300 char): {s(first_row.get(essay_col))[:300]}")
        print(f"notas detectadas: {grades_text}")
        return 0

    try:
        import anthropic
    except ImportError as e:
        print(f"[ERRO] 'anthropic' ausente ({e}). Rode: pip install -r requirements/corpus.txt")
        return 1

    client = anthropic.Anthropic()
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "aes_enem_feedback.jsonl")

    n, n_no_grades, n_api_fail, in_tok, out_tok = 0, 0, 0, 0, 0
    with open(out_path, "w", encoding="utf-8") as fp:
        for i, ex in enumerate(ds):
            if args.max_examples and n >= args.max_examples:
                break
            essay = s(ex.get(essay_col))
            grades_text = build_grades_text(ex, single_col, comp_cols)
            if not essay or not grades_text:
                n_no_grades += 1
                continue
            user_msg = f"Redação:\n{essay}\n\nNotas já atribuídas pelo corretor:\n{grades_text}"
            try:
                resp = call_with_retry(
                    client, model=args.model, max_tokens=args.max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                )
            except Exception as e:
                print(f"[aviso] exemplo {i}: falha na API ({e}). PULADO.")
                n_api_fail += 1
                continue
            in_tok += resp.usage.input_tokens
            out_tok += resp.usage.output_tokens
            feedback = next((b.text for b in resp.content if b.type == "text"), "").strip()
            if not feedback:
                n_api_fail += 1
                continue
            fp.write(json.dumps({"instruction": INSTRUCTION, "input": essay, "output": feedback,
                                 "task": "aes_enem_feedback"}, ensure_ascii=False) + "\n")
            n += 1
            if (n) % 50 == 0:
                print(f"[aes-enem] {n} exemplos gravados...")

    price_in, price_out = PRICING_SNAPSHOT.get(args.model, (None, None))
    est_cost = (round(in_tok / 1e6 * price_in + out_tok / 1e6 * price_out, 4)
                if price_in is not None else None)

    manifest = {
        "dataset": "aes_enem_feedback", "hf_repo": HF_REPO_ID, "config": cfg,
        "license": LICENSE, "reference": REFERENCE, "model": args.model,
        "essay_column": essay_col, "single_grade_column": single_col, "competency_columns": comp_cols,
        "total": n, "skipped_no_grades": n_no_grades, "skipped_api_failure": n_api_fail,
        "tokens_input": in_tok, "tokens_output": out_tok, "estimated_cost_usd": est_cost,
        "estimated_cost_note": "Chamadas sincronas, SEM desconto de Batches. Conferir preco vigente.",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)

    print(f"\n[aes-enem] TOTAL: {n:,} exemplos -> {out_path}")
    print(f"[aes-enem] tokens: {in_tok:,} entrada / {out_tok:,} saida"
          + (f" | custo estimado: US${est_cost}" if est_cost is not None else ""))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[INFO] Interrompido pelo usuario.")
        sys.exit(130)
