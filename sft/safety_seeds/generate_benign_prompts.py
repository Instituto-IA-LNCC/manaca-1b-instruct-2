#!/usr/bin/env python3
"""
Manacá-1B-Instruct-2 — Geração de prompts benignos-mas-sensíveis (escala o lado "ajuda")
============================================================================================
PT
--
O safety-SFT do Manacá está em 411 exemplos contra 1.800 do AnswerCarefully
v2 (LLM-jp) — ver `sft/README-safety.md` §4-5. Os itens 1-2 do plano
(`mine_harmful_prompts.py`, `mine_refusal_bank.py`) escalam o lado RECUSA.
Este script escala o lado que faltava: **AJUDA** — hoje travado em 55
prompts (`sft/safety_seeds/benign_help.jsonl`), todos escritos à mão.

POR QUE GERAÇÃO VIA CLAUDE AQUI É SEGURA (diferente do lado nocivo, que
exige auditoria antes de usar): estamos gerando **pedidos benignos**
(fraseado sensível, mas o pedido em si é legítimo — proteção digital,
saúde não-emergencial, autodefesa, direitos legais, ficção com temas
pesados). O pior caso de um prompt malfeito aqui é um exemplo de baixa
qualidade, não um exemplo perigoso — por isso este script NÃO exige
`--sample-audit` obrigatório como os mineradores do lado nocivo (ainda que
uma conferência rápida nunca seja má ideia).

IMPORTANTE — o que este script NÃO faz: gera só os **prompts**, não as
respostas. As respostas continuam on-policy (geradas pelo próprio SFT via
`sft/gen_safety_sft.py --model ...`), exatamente como os 50 prompts
originais — EXCETO para categorias onde já se sabe que o modelo sem
alinhamento falha (o único caso disso hoje é "ajudar em crise", que já tem
resposta pronta e não passa por este script). Se uma nova categoria gerada
aqui também se mostrar problemática on-policy, o tratamento é o mesmo já
usado para a crise: escrever e revisar uma resposta pronta à parte.

CATEGORIAS (inspiradas no vão que separa "recusa" de "ajuda" — o tipo de
pedido que soa alarmante mas é legítimo, na linha do XSTest em inglês, que
não tem versão em PT-BR pronta): segurança digital, saúde não-emergencial,
autodefesa pessoal, ficção/criativo com temas pesados, direitos legais
cotidianos, finanças e golpes, conflitos e relacionamentos, prevenção de
acidentes domésticos, primeiros socorros, denúncia de abuso/assédio.

DECONTAMINAÇÃO: deduplica contra `sft/safety_seeds/benign_help.jsonl`
(não repete o que já existe) E contra `bench/overrefusal_pt/prompts.jsonl`
(o benign_help precisa ficar DISJUNTO da avaliação de over-refusal, senão
contamina a métrica — mesmo cuidado já documentado no arquivo original).

Uso:
    export ANTHROPIC_API_KEY=sk-ant-...
    python sft/safety_seeds/generate_benign_prompts.py --dry-run          # sem custo
    python sft/safety_seeds/generate_benign_prompts.py                   # gera de verdade
    python sft/safety_seeds/generate_benign_prompts.py --categories seguranca_digital,autodefesa

Saída: `sft/safety_seeds/benign_help_generated.jsonl` (schema igual ao
original: `{id, tipo, prompt}`, sem `output` -> fica on-policy) + manifest
com custo estimado. Junte com o original:

    cat sft/safety_seeds/benign_help.jsonl \\
        sft/safety_seeds/benign_help_generated.jsonl \\
        > sft/safety_seeds/benign_help_v2.jsonl

EN
--
Scales the "help the benign" side of the safety-SFT (stuck at 55 hand-
written prompts) by generating NEW PROMPTS (not responses — those stay
on-policy, same design as the original 50) across categories inspired by
the XSTest over-refusal taxonomy (no ready PT-BR version exists). Safe to
automate (benign requests, not harmful), deduplicated against both the
existing file and the over-refusal eval set (to keep it disjoint, per the
project's existing convention).

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone

DEFAULT_MODEL = os.environ.get("MODEL", "claude-opus-5")
DEFAULT_PER_CATEGORY = 10
DEFAULT_MAX_TOKENS = 2000

# Preço aproximado por 1M tokens (input/output), USD -- SEM desconto de
# Batches (este script usa chamadas sincronas, volume pequeno). Mesma
# fonte/ressalva do synth/generate_instructions.py: conferir preco vigente.
PRICING_SNAPSHOT = {
    "claude-opus-5":    (5.00, 25.00),
    "claude-sonnet-5":  (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# (categoria, descricao do tema para o prompt de geracao)
CATEGORIES = [
    ("seguranca_digital", "proteger contas, dispositivos e dados pessoais de ataques "
     "(sem pedir para ATACAR ninguem -- e sobre SE PROTEGER)"),
    ("saude_nao_emergencial", "duvidas de saude cotidianas, primeiros socorros basicos, "
     "interacao de remedios, sem substituir diagnostico medico"),
    ("autodefesa_pessoal", "como se proteger fisicamente em situacoes de risco (assalto, "
     "perseguicao), sem pedir para agredir ninguem primeiro"),
    ("ficcao_temas_pesados", "pedidos de escrita criativa (romance policial, thriller, "
     "roteiro) que mencionam crime, violencia ou substancias APENAS como parte da trama"),
    ("direitos_legais_cotidianos", "duvidas sobre direitos trabalhistas, do consumidor, "
     "de inquilino, sem pedir para burlar a lei"),
    ("financas_e_golpes", "como reconhecer e se proteger de golpes financeiros, sem pedir "
     "para aplicar um golpe"),
    ("conflitos_e_relacionamentos", "como lidar com conflitos, ciume, termino, ou suspeita "
     "de traicao de forma saudavel, sem pedir para invadir privacidade de ninguem"),
    ("prevencao_acidentes_domesticos", "como prevenir acidentes com criancas, produtos "
     "quimicos domesticos, ou eletricidade em casa"),
    ("denuncia_abuso_assedio", "como reconhecer e denunciar assedio ou abuso (trabalho, "
     "relacionamento, escola), buscando ajuda apropriada"),
    ("seguranca_no_transito", "duvidas sobre direcao segura, o que fazer em acidentes, "
     "primeiros socorros no transito"),
]

SYSTEM_PROMPT = """Você gera exemplos de PEDIDOS BENIGNOS mas de fraseado sensível, para um \
dataset de treino de segurança de um assistente de IA em português do Brasil. O objetivo é \
ensinar o assistente a NÃO recusar pedidos legítimos só porque soam alarmantes.

Regras obrigatórias:
- Cada pedido deve ser algo que uma pessoa real e bem-intencionada perguntaria.
- O pedido deve ser CLARAMENTE legítimo dentro do tema dado -- não deve conter nenhum \
pedido real de dano, ataque, ou informação perigosa.
- Varie o fraseado: algumas perguntas diretas, outras mais informais/coloquiais, outras \
com um pouco de urgência ou preocupação genuína.
- Responda em português do Brasil.
- Responda APENAS com um array JSON de strings, sem markdown, sem texto antes ou depois: \
["pedido 1", "pedido 2", ...]"""


def s(x) -> str:
    return ("" if x is None else str(x)).strip()


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.strip().lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text)


def load_existing_prompts(paths: list[str], field_map: dict[str, str]) -> set[str]:
    """field_map: {caminho: nome_do_campo} (alguns arquivos usam 'prompt', outros 'turns')."""
    seen = set()
    for path in paths:
        if not os.path.isfile(path):
            print(f"[aviso] {path} nao encontrado -- pulando na deduplicacao.")
            continue
        field = field_map.get(path, "prompt")
        n = 0
        with open(path, encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
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
        print(f"[dedup] {path}: {n} prompts carregados")
    return seen


def parse_json_array(raw_text: str) -> list[str] | None:
    text = raw_text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        arr = json.loads(match.group(0))
        return [s(x) for x in arr if s(x)]
    except json.JSONDecodeError:
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
    ap = argparse.ArgumentParser(description="Gera prompts benignos-mas-sensiveis (escala o lado 'ajuda' do safety-SFT)")
    ap.add_argument("--out", default="sft/safety_seeds/benign_help_generated.jsonl")
    ap.add_argument("--categories", default="all",
                    help="'all' ou lista por virgula (ex.: seguranca_digital,autodefesa_pessoal)")
    ap.add_argument("--per_category", type=int, default=DEFAULT_PER_CATEGORY,
                    help=f"quantos prompts pedir por categoria por chamada (default {DEFAULT_PER_CATEGORY})")
    ap.add_argument("--calls_per_category", type=int, default=1,
                    help="quantas chamadas (rodadas) por categoria, para mais volume/diversidade")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--existing_benign", default="sft/safety_seeds/benign_help.jsonl")
    ap.add_argument("--overrefusal_eval", default="bench/overrefusal_pt/prompts.jsonl",
                    help="deduplicar contra a avaliacao de over-refusal (manter disjunto)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="mostra os prompts de sistema sem chamar a API")
    args = ap.parse_args()

    selected = CATEGORIES if args.categories.strip() == "all" else [
        c for c in CATEGORIES if c[0] in {x.strip() for x in args.categories.split(",")}
    ]
    print(f"[gen-benign] {len(selected)} categorias selecionadas: {[c[0] for c in selected]}")

    if args.dry_run:
        print("\n[dry-run] nenhuma chamada de API sera feita.\n")
        print("=" * 60)
        print("SYSTEM PROMPT:\n" + SYSTEM_PROMPT)
        for cat, desc in selected[:3]:
            print("=" * 60)
            print(f"[{cat}] Gere {args.per_category} pedidos benignos sobre: {desc}")
        return 0

    existing = load_existing_prompts(
        [args.existing_benign, args.overrefusal_eval],
        {args.overrefusal_eval: "prompt"},
    )
    print(f"[dedup] total: {len(existing)} prompts existentes (benign_help + overrefusal eval)")

    try:
        import anthropic
    except ImportError as e:
        print(f"[ERRO] 'anthropic' ausente ({e}). Rode: pip install -r requirements/corpus.txt")
        return 1

    client = anthropic.Anthropic()
    rng = random.Random(args.seed)

    seen_norm = set(existing)
    generated: list[dict] = []
    counts: dict[str, int] = {}
    in_tok, out_tok = 0, 0
    next_id = 56  # continua depois de bh_55 (ver sft/README-safety.md SS5)

    for cat, desc in selected:
        cat_kept = 0
        for call_n in range(args.calls_per_category):
            user_msg = f"Gere {args.per_category} pedidos benignos sobre: {desc}"
            try:
                resp = call_with_retry(
                    client, model=args.model, max_tokens=args.max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                )
            except Exception as e:
                print(f"[aviso] {cat} (chamada {call_n}): falha na API ({e}). PULADA.")
                continue

            in_tok += resp.usage.input_tokens
            out_tok += resp.usage.output_tokens
            text = next((b.text for b in resp.content if b.type == "text"), "")
            prompts = parse_json_array(text)
            if not prompts:
                print(f"[aviso] {cat} (chamada {call_n}): resposta nao-parseavel. PULADA.")
                continue

            for p in prompts:
                norm = _normalize(p)
                if norm in seen_norm:
                    continue
                seen_norm.add(norm)
                generated.append({"id": f"bh_{next_id:03d}", "tipo": cat, "prompt": p})
                next_id += 1
                cat_kept += 1
        counts[cat] = cat_kept
        print(f"[gen-benign] {cat}: {cat_kept} prompts novos")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fp:
        for r in generated:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")

    price_in, price_out = PRICING_SNAPSHOT.get(args.model, (None, None))
    est_cost = (round(in_tok / 1e6 * price_in + out_tok / 1e6 * price_out, 4)
                if price_in is not None else None)

    manifest = {
        "dataset": "benign_help_generated", "model": args.model,
        "categories": counts, "total": len(generated),
        "tokens_input": in_tok, "tokens_output": out_tok,
        "estimated_cost_usd": est_cost,
        "estimated_cost_note": "Chamadas sincronas, SEM desconto de Batches. Conferir preco vigente.",
        "deduplicated_against": [args.existing_benign, args.overrefusal_eval],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(os.path.dirname(args.out) or ".", "benign_help_generated_manifest.json"),
              "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)

    print(f"\n[gen-benign] TOTAL: {len(generated)} prompts novos -> {args.out}")
    print(f"[gen-benign] tokens: {in_tok:,} entrada / {out_tok:,} saida"
          + (f" | custo estimado: US${est_cost}" if est_cost is not None else ""))
    print(f"[gen-benign] junte com o original: cat {args.existing_benign} {args.out} > "
          f"sft/safety_seeds/benign_help_v2.jsonl")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[INFO] Interrompido pelo usuario.")
        sys.exit(130)
