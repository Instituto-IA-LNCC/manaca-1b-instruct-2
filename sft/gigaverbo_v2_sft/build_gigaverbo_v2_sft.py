#!/usr/bin/env python3
"""
Manacá-1B-Instruct-2 — Conversão do GigaVerbo-v2 SFT (Polygl0t) para o esquema Alpaca
========================================================================================
PT
--
`Polygl0t/gigaverbo-v2-sft` é um dataset PRONTO de instrução em PT-BR do
projeto Tucano 2 (arXiv:2603.03543): ~4,09M exemplos / ~2,1B tokens / 4,7 GB,
Apache-2.0, 100% gerado por LLM (não é tradução — ataca direto o problema já
documentado do corpus v1/v2, onde Alpaca-PT+OPUS-100 traduzidos são ~64% do
mix), filtrado por juiz LLM (nota < 3 descartada) e decontaminado pelos
autores contra OS BENCHMARKS DELES. Organizado em configs por categoria de
tarefa (function_call, summarization, code, structured, translation, math,
math_cot, rewriting e mais — descobertas dinamicamente, não fixadas aqui).

VEREDITO DE UTILIDADE (por que este script existe):
    Sim, é útil — e é o maior salto de alavancagem disponível hoje para o
    instruct-2: dados nativos (não traduzidos), sintéticos e QUALIFICADOS,
    numa escala (~4M) que nenhum pipeline de geração sintética própria deste
    projeto (ver corpus/scripts/, que ainda dependem de um passo de geração
    via LLM professor) alcançaria com orçamento de pesquisa. Preenche
    categorias que o manacá-jaster mal cobre hoje (code, math, function_call,
    structured output).

O QUE ESTE SCRIPT FAZ (e as decisões que exige):

  1. DESCOBRE as configs dinamicamente via `datasets.get_dataset_config_names`
     — não fixamos a lista de 12 categorias (a fonte é a autoridade, e uma
     lista hardcoded ficaria desatualizada silenciosamente).

  2. CONVERTE conversa multi-turno -> Alpaca `{instruction, input, output}`:
     o `sft/train.py` deste repo é um treinador COMPLETION-ONLY de turno
     único (marcadores `### Instrução:` / `### Resposta:`), não suporta
     multi-turno nativamente. A conversão dobra todos os turnos anteriores
     ao último (system + pares usuário/assistente) dentro de `input` como
     transcrição, e usa a ÚLTIMA pergunta do usuário como `instruction` e a
     ÚLTIMA resposta do assistente como `output`. Isso preserva TODO exemplo
     (em vez de só os de 1 turno), ao custo de não treinar nos turnos
     intermediários — uma limitação conhecida, não um bug; ver
     "Próximos passos" no fim deste docstring.

  3. AMOSTRA por config (`--max_per_task`, default 20.000 != 0): diferente
     do manacá-jaster (tarefas de ~1-50k linhas), aqui uma única config pode
     ter centenas de milhares de linhas — sem teto, o mix de instruct-2
     ficaria dominado por esta única fonte (o v2 inteiro tem 141k exemplos
     NO TOTAL). Ajuste com critério, não deixe em 0 sem pensar no orçamento
     de compute do treino.

  4. DECONTAMINA (best-effort) contra os PRÓPRIOS benchmarks deste repo —
     bench/ifeval_pt/prompts.jsonl, bench/mtbench_pt/questions.jsonl,
     bench/safety_pt/heldout.jsonl, bench/overrefusal_pt/prompts.jsonl,
     dpo/dpo_seeds/safety_prompts.jsonl, sft/safety_seeds/benign_help.jsonl
     — por normalização + correspondência exata/substring. A decontaminação
     dos autores foi contra OS BENCHMARKS DELES, não contra os nossos.
     LIMITAÇÃO HONESTA: ENEM/BLUEX/OAB/ASSIN2/HateBR (carregados
     dinamicamente do HF pelo lm-eval, não são arquivos estáticos deste
     repo) NÃO são cobertos por esta checagem — ver "Próximos passos".

Saída: `{instruction, input, output, task}` (mesmo esquema jaster/prepare_data.py)
em `data/sft/gigaverbo_v2_sft/<config>.jsonl` + combinado
`gigaverbo_v2_sft.jsonl` + `manifest.json` (contagens + sha256 por config).

Uso no mix de treino (igual ao manacá-jaster, via `--extra`):
    python sft/prepare_data.py --sources alpaca,aya,oasst --out data/sft_v3 \\
        --shuffle --dedup \\
        --extra data/sft/gigaverbo_v2_sft/gigaverbo_v2_sft.jsonl \\
                data/sft_v2/jaster/manaca_jaster.jsonl

Próximos passos (fora do escopo deste script, registrados para o backlog):
    - Multi-turno de verdade: dar a `sft/train.py` um chat template com roles
      (ver recomendação já discutida para o instruct-2), o que elimina a
      necessidade de dobrar turnos em `input` e usa os ~4M exemplos por
      inteiro em vez de só o último par.
    - Decontaminação completa: rodar contra os datasets de exame do lm-eval
      (ENEM/BLUEX/OAB/ASSIN2/HateBR) carregados via HF, não só os arquivos
      estáticos deste repo.

EN
--
`Polygl0t/gigaverbo-v2-sft` is a READY-MADE PT-BR instruction dataset from
the Tucano 2 project: ~4.09M examples / ~2.1B tokens, Apache-2.0, fully
LLM-generated (not translated), quality-filtered and decontaminated by its
authors (against their own benchmarks). This script discovers its configs
dynamically, folds multi-turn conversations into the single-turn Alpaca
schema this repo's trainer consumes, samples per config (default cap
20,000 — this source alone could otherwise dwarf the whole v1/v2 mix), and
re-decontaminates against this repo's own static benchmark/seed files
(best-effort; exam datasets loaded dynamically by lm-eval are not covered).

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

HF_REPO_ID = "Polygl0t/gigaverbo-v2-sft"
LICENSE    = "apache-2.0"
REFERENCE  = "Polygl0t / Tucano 2 (2026). GigaVerbo-v2 SFT. arXiv:2603.03543."

DEFAULT_MAX_PER_TASK = 20_000  # ver justificativa no docstring (item 3)

# Arquivos estáticos deste repo usados na decontaminação best-effort (item 4).
# Cada entrada: (caminho relativo ao repo, campo do texto do prompt).
# "turns" é lista (usamos o 1º elemento); os demais são string direta.
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
    """
    Normalização para comparação de decontaminação: minúsculas, sem acentos
    (NFKD + remoção de marcas combinantes) e espaços colapsados. Sem essa
    dobra de acentos, uma variante re-acentuada do mesmo prompt escaparia da
    checagem por igualdade/substring — limitação lexical ainda existe para
    paráfrases de verdade (não é dedup semântico), mas cobre a forma mais
    comum de "quase-igual" (mesma frase, acentuação/caixa diferente).
    """
    text = unicodedata.normalize("NFKD", text.strip().lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text)


def load_decontamination_set(repo_root: str) -> set[str]:
    """
    Carrega e normaliza os prompts dos benchmarks/seeds ESTÁTICOS deste repo.
    Tolerante: arquivo ausente é pulado com aviso (não derruba o build).
    """
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
    """Correspondência exata ou substring (nos dois sentidos) após normalização."""
    if not decontam_set:
        return False
    norm = _normalize(instruction)
    if norm in decontam_set:
        return True
    return any(norm in bench or bench in norm for bench in decontam_set if len(bench) > 20)


# ── Conversão: conversa multi-turno -> Alpaca (item 2 do docstring) ───────────

def _extract_role_content(msg) -> tuple[str, str]:
    """Schema tolerante: {'role','content'} (OpenAI/Qwen) ou {'from','value'} (ShareGPT)."""
    if not isinstance(msg, dict):
        return "", ""
    if "role" in msg:
        role, content = s(msg.get("role")), s(msg.get("content"))
    elif "from" in msg:
        role_raw = s(msg.get("from")).lower()
        role = {"human": "user", "gpt": "assistant", "bot": "assistant"}.get(role_raw, role_raw)
        content = s(msg.get("value"))
    else:
        return "", ""
    return role.lower(), content


def messages_to_alpaca(messages: list) -> dict | None:
    """
    Dobra uma conversa multi-turno em {instruction, input, output}.

    - Último turno deve ser do assistente (senão descarta: conversa incompleta).
    - Última pergunta do usuário -> instruction.
    - Tudo antes disso (system + pares usuário/assistente anteriores) ->
      transcrição em `input`, prefixada por papel.
    - Conversas de 1 único par (user, assistant) -> input fica vazio (caso comum).
    """
    if not messages or not isinstance(messages, list):
        return None
    parsed = [(_extract_role_content(m)) for m in messages]
    parsed = [(r, c) for r, c in parsed if r and c]
    if len(parsed) < 2 or parsed[-1][0] != "assistant":
        return None

    output = parsed[-1][1]
    # Procura a ultima mensagem de usuario antes da resposta final.
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


def conv_gigaverbo(ds, message_col: str):
    for ex in ds:
        row = messages_to_alpaca(ex.get(message_col))
        if row and row["instruction"] and row["output"]:
            yield row


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Converte Polygl0t/gigaverbo-v2-sft (chat) para Alpaca {instruction,input,output}")
    ap.add_argument("--tasks", default="all",
                    help="'all' (descobre configs no HF) ou lista por vírgula de configs específicas")
    ap.add_argument("--out", default="data/sft/gigaverbo_v2_sft", help="diretório de saída")
    ap.add_argument("--max_per_task", type=int, default=DEFAULT_MAX_PER_TASK,
                    help=f"teto de exemplos por config (default {DEFAULT_MAX_PER_TASK:,}; 0 = sem teto)")
    ap.add_argument("--max_chars", type=int, default=5000,
                    help="descarta exemplos cujo prompt (instrucao+entrada) passa deste nº de caracteres (0 = off)")
    ap.add_argument("--repo_root", default=".",
                    help="raiz do repo (para localizar os arquivos de decontaminacao)")
    ap.add_argument("--no_decontaminate", action="store_true",
                    help="pula a decontaminacao contra os benchmarks/seeds estaticos deste repo (NAO recomendado)")
    ap.add_argument("--shuffle", action="store_true", help="embaralha o mix combinado (semente fixa)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        from datasets import get_dataset_config_names, load_dataset
    except ImportError as e:
        print(f"[ERRO] 'datasets' ausente ({e}). Rode na imagem manaca-posttrain.")
        return 1

    decontam_set: set[str] = set()
    if not args.no_decontaminate:
        decontam_set = load_decontamination_set(args.repo_root)
        print(f"[decontam] total: {len(decontam_set)} prompts normalizados")
    else:
        print("[decontam] DESATIVADA via --no_decontaminate — nao recomendado.")

    print(f"[gigaverbo-v2-sft] descobrindo configs de {HF_REPO_ID} ...")
    try:
        all_configs = get_dataset_config_names(HF_REPO_ID)
    except Exception as e:
        print(f"[ERRO] falha ao descobrir configs de {HF_REPO_ID}: {e}")
        return 1
    print(f"[gigaverbo-v2-sft] {len(all_configs)} configs encontradas: {all_configs}")

    if args.tasks.strip() == "all":
        selected = all_configs
    else:
        wanted = {t.strip() for t in args.tasks.split(",") if t.strip()}
        selected = [c for c in all_configs if c in wanted]
        missing = wanted - set(selected)
        if missing:
            print(f"[aviso] configs pedidas mas nao encontradas no HF: {sorted(missing)}")

    os.makedirs(args.out, exist_ok=True)
    combined_path = os.path.join(args.out, "gigaverbo_v2_sft.jsonl")
    counts, sha, skip_contam, skip_long = {}, {}, {}, {}

    def sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as fp:
            for chunk in iter(lambda: fp.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()

    with open(combined_path, "w", encoding="utf-8") as fout:
        for name in selected:
            try:
                print(f"[gigaverbo-v2-sft] {name}: carregando (streaming)...")
                ds = load_dataset(HF_REPO_ID, name, split="train", streaming=True)
            except Exception as e:
                print(f"[aviso] {name}: falha ao carregar ({type(e).__name__}: {e}). PULADA.")
                counts[name] = 0
                continue

            try:
                first = next(iter(ds))
            except StopIteration:
                print(f"[aviso] {name}: config vazia. PULADA.")
                counts[name] = 0
                continue
            message_col = next(
                (c for c in ["messages", "conversations", "conversation", "chat"] if c in first),
                None,
            )
            if message_col is None:
                print(f"[aviso] {name}: nenhuma coluna de conversa reconhecida em {list(first.keys())}. PULADA.")
                counts[name] = 0
                continue

            per_path = os.path.join(args.out, f"{name}.jsonl")
            n, n_contam, n_long = 0, 0, 0
            with open(per_path, "w", encoding="utf-8") as fp:
                for row in conv_gigaverbo(ds, message_col):
                    if args.max_per_task and n >= args.max_per_task:
                        break
                    if is_contaminated(row["instruction"], decontam_set):
                        n_contam += 1
                        continue
                    if args.max_chars and (len(row["instruction"]) + len(row["input"])) > args.max_chars:
                        n_long += 1
                        continue
                    row["task"] = f"gigaverbo_v2_sft_{name}"
                    line = json.dumps(row, ensure_ascii=False)
                    fp.write(line + "\n")
                    fout.write(line + "\n")
                    n += 1
            counts[name] = n
            skip_contam[name] = n_contam
            skip_long[name] = n_long
            if n:
                sha[name] = sha256(per_path)
            print(f"[gigaverbo-v2-sft] {name}: {n:,} exemplos "
                  f"(descartados: {n_contam} contaminados, {n_long} longos) -> {per_path}")

    if args.shuffle:
        with open(combined_path, encoding="utf-8") as fp:
            lines = fp.readlines()
        random.Random(args.seed).shuffle(lines)
        with open(combined_path, "w", encoding="utf-8") as fp:
            fp.writelines(lines)

    total = sum(counts.values())
    if total:
        sha["gigaverbo_v2_sft.jsonl"] = sha256(combined_path)
    manifest = {
        "dataset": "gigaverbo_v2_sft", "hf_repo": HF_REPO_ID, "license": LICENSE,
        "reference": REFERENCE, "configs": counts, "skipped_contaminated": skip_contam,
        "skipped_too_long": skip_long, "total": total, "max_per_task": args.max_per_task,
        "decontaminated": not args.no_decontaminate,
        "decontamination_note": (
            "Best-effort contra bench/dpo/sft/safety_seeds ESTATICOS deste repo. "
            "NAO cobre ENEM/BLUEX/OAB/ASSIN2/HateBR (carregados via HF pelo lm-eval)."
        ),
        "shuffled": args.shuffle, "seed": args.seed, "sha256": sha,
    }
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)

    print(f"[gigaverbo-v2-sft] TOTAL: {total:,} exemplos -> {combined_path}")
    if total == 0:
        print("[gigaverbo-v2-sft] AVISO: nenhum exemplo gerado (verifique rede/config; --tasks).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
