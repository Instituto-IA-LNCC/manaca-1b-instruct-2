#!/usr/bin/env python3
"""
Manacá-1B-Instruct-2 — Geração de PROMPTS nocivos via API Anthropic (escala o lado "recusa")
============================================================================================
PT
--
Meta explícita (ver `sft/README-safety.md` §4-6): sair de 411 exemplos de
safety-SFT para ~2.000 (acima do AnswerCarefully v2 do LLM-jp, 1.800).
`mine_harmful_prompts.py` minera prompts de um dataset de terceiros (volume
incerto até rodar, categorias não documentadas na fonte); este script GERA
prompts novos via Claude, dando controle direto sobre volume e cobertura de
categoria -- é o "usar a API Anthropic para chegar lá" que fecha a conta.

CONTA (ver `sft/README-safety.md` §6 para o detalhamento):
    seeds atuais (50 escritos à mão) + ~350 gerados aqui = ~400 seeds
    400 seeds x K_REFUSE=4 (receita "safe4")           = 1.600 recusa
    benign_help (55) + gerados por generate_benign_prompts.py (~200)
                                                         =   255 ajuda
    N_GENERAL (amostra do SFT geral)                    =   200 geral
    -----------------------------------------------------------------
    TOTAL projetado                                     = ~2.055

POR QUE ISSO É SEGURO DE AUTOMATIZAR: o script gera só o PEDIDO (uma frase,
no estilo dos 50 seeds já existentes -- ver exemplos abaixo), nunca a
resposta. A resposta continua sendo um dos templates fixos de recusa em
`sft/gen_safety_sft.py` (RECUSAS / RECUSA_AUTOLESAO) -- o modelo Claude
usado aqui NUNCA gera conteúdo nocivo de verdade, só o enunciado do pedido
que um assistente bem alinhado deve recusar. É a mesma prática usada por
datasets públicos de safety (AnswerCarefully, do-not-answer, HarmBench):
gerar PERGUNTAS de red-team com um LLM para treinar RECUSA, não gerar as
respostas nocivas.

CATEGORIAS: reaproveita as 14 categorias já usadas em `safety_prompts.jsonl`
que não exigem cuidado extra de conteúdo sensível-a-pessoa (ciberataque,
fraude, malware, documento_falso, violencia, arma, droga, roubo,
privacidade, fiscal, odio, assedio, extorsao, stalking). DELIBERADAMENTE
EXCLUI "autolesao" e "sexual": autolesão já tem tratamento especial neste
projeto (RECUSA_AUTOLESAO + a categoria "ajudar_em_crise" revisada à mão,
ver README-safety.md §5) e não deve ser escalada por geração em lote sem a
mesma revisão manual; "sexual" fica de fora por ser a categoria mais
sensível a falsos positivos/negativos de geração automática. Se precisar
de mais desses dois, o caminho é o mesmo já usado para a crise: escrever e
revisar à mão, não gerar em lote.

Uso:
    export ANTHROPIC_API_KEY=sk-ant-...
    python dpo/dpo_seeds/generate_harmful_prompts.py --dry-run          # sem custo
    python dpo/dpo_seeds/generate_harmful_prompts.py --sample-audit 15 # gera e so audita (nao grava)
    python dpo/dpo_seeds/generate_harmful_prompts.py                   # gera e grava o candidato

Saída: `dpo/dpo_seeds/safety_prompts_generated.jsonl` (schema idêntico a
`safety_prompts.jsonl`: `{id, categoria, prompt}`), marcado como
CANDIDATO no manifest -- mesma política dos mineradores (auditar antes de
usar). Deduplicado contra `safety_prompts.jsonl`,
`safety_prompts_ambiguous.jsonl` e `safety_prompts_mined.jsonl` (os que
existirem). Junte tudo com:

    cat dpo/dpo_seeds/safety_prompts.jsonl \\
        dpo/dpo_seeds/safety_prompts_ambiguous.jsonl \\
        dpo/dpo_seeds/safety_prompts_mined.jsonl \\
        dpo/dpo_seeds/safety_prompts_generated.jsonl \\
        > dpo/dpo_seeds/safety_prompts_v2.jsonl

EN
--
Generates NEW harmful-request PROMPTS (never the responses -- those stay
fixed refusal templates in gen_safety_sft.py) via the Claude API, to scale
the refusal-side seed pool from 50 hand-written prompts toward the ~400
needed to reach a ~2,000-example safety-SFT dataset (surpassing LLM-jp's
AnswerCarefully v2 at 1,800). Same practice as public safety datasets
(AnswerCarefully, do-not-answer, HarmBench): an LLM generates red-team
QUESTIONS, never the harmful answers. Deliberately excludes the
"autolesao" (self-harm) and "sexual" categories, which get the same
hand-written + reviewed treatment already used for the "ajudar_em_crise"
category -- not bulk API generation.

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

# claude-opus-5 bloqueia por completo (resposta vazia, nao recusa com texto)
# as categorias ciberataque/malware/privacidade -- confirmado ao vivo, 3/3
# chamadas vazias nas 3 categorias. claude-sonnet-5 gera normalmente as
# mesmas categorias (30/30/30 pedidos plausiveis, sem conteudo extremo) e
# e mais barato -- vira o default.
DEFAULT_MODEL = os.environ.get("MODEL", "claude-sonnet-5")
DEFAULT_PER_CATEGORY = 10
DEFAULT_CALLS_PER_CATEGORY = 3
DEFAULT_MAX_TOKENS = 2000

# Preço aproximado por 1M tokens (input/output), USD -- chamadas sincronas,
# SEM desconto de Batches (volume pequeno). Mesma ressalva do
# synth/generate_instructions.py e generate_benign_prompts.py: conferir
# preco vigente.
PRICING_SNAPSHOT = {
    "claude-opus-5":    (5.00, 25.00),
    "claude-sonnet-5":  (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# Categorias reaproveitadas de dpo/dpo_seeds/safety_prompts.jsonl (exclui
# "autolesao" e "sexual" de proposito -- ver docstring).
CATEGORIES = [
    ("ciberataque", "invadir contas, redes ou dispositivos de terceiros sem autorizacao "
     "(phishing, senha, autenticacao, wifi alheio)"),
    ("fraude", "falsificar documentos financeiros, aplicar golpes, burlar verificacao de "
     "identidade ou credito"),
    ("malware", "criar ou modificar software malicioso (virus, ransomware, trojan, keylogger)"),
    ("documento_falso", "falsificar identidade, diploma, assinatura, comprovante ou outro "
     "documento oficial"),
    ("violencia", "causar dano fisico a uma pessoa especifica (agredir, envenenar, "
     "sequestrar)"),
    ("arma", "fabricar ou modificar armas de fogo, explosivos ou armas caseiras"),
    ("droga", "produzir, extrair ou traficar substancias ilegais"),
    ("roubo", "arrombar, furtar ou roubar bens de terceiros sem ser notado"),
    ("privacidade", "monitorar, rastrear ou acessar comunicacoes privadas de outra pessoa "
     "sem consentimento"),
    ("fiscal", "sonegar impostos, lavar dinheiro ou burlar fiscalizacao financeira"),
    ("odio", "produzir discurso de odio ou incitar discriminacao contra um grupo"),
    ("assedio", "assediar, intimidar ou humilhar uma pessoa especifica repetidamente"),
    ("extorsao", "chantagear ou extorquir dinheiro/favores de alguem sob ameaca"),
    ("stalking", "localizar, seguir ou vigiar uma pessoa contra a vontade dela"),
]

SYSTEM_PROMPT = """Você ajuda a construir um dataset de SEGURANÇA para treinar um assistente \
de IA a RECUSAR pedidos nocivos -- a mesma prática usada em datasets públicos como \
AnswerCarefully, do-not-answer e HarmBench.

Sua tarefa é gerar PEDIDOS (perguntas/instruções) que uma pessoa mal-intencionada ou confusa \
poderia fazer a um assistente de IA, dentro do tema dado. Você gera SOMENTE o pedido -- \
NUNCA a resposta, NUNCA informação real de como executar o pedido.

Regras obrigatórias:
- Cada pedido é UMA frase curta e direta (no estilo: "Como falsificar um comprovante de \
renda para aprovar um empréstimo?", "Como burlar a autenticação de dois fatores da conta de \
alguém?").
- O pedido em si NÃO deve conter instruções, passos ou informação real -- é só o ENUNCIADO \
do que a pessoa quer, exatamente como ela perguntaria.
- Varie o fraseado: pedidos diretos, pedidos disfarçados de "hipótese" ou "curiosidade", \
pedidos com contexto que soa como desculpa (ex.: "é para um trabalho da faculdade"), pedidos \
mais informais/coloquiais.
- Responda em português do Brasil.
- Responda APENAS com um array JSON de strings, sem markdown, sem texto antes ou depois: \
["pedido 1", "pedido 2", ...]"""


def s(x) -> str:
    return ("" if x is None else str(x)).strip()


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.strip().lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text)


def load_existing_prompts(paths: list[str]) -> set[str]:
    seen = set()
    for path in paths:
        if not os.path.isfile(path):
            print(f"[aviso] {path} nao encontrado -- pulando na deduplicacao.")
            continue
        n = 0
        with open(path, encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = s(row.get("prompt"))
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
    if match:
        try:
            arr = json.loads(match.group(0))
            result = [s(x) for x in arr if s(x)]
            if result:
                return result
        except json.JSONDecodeError:
            pass
    # Fallback tolerante: a resposta pode nao ser um array JSON estrito (ex.:
    # o modelo respondeu com preambulo/ressalva + lista numerada com os
    # pedidos entre aspas, em vez do formato pedido) -- extrai strings entre
    # aspas duplas plausiveis como pedido em vez de descartar tudo.
    quoted = re.findall(r'"([^"\n]{10,300})"', text)
    if quoted:
        return [s(x) for x in quoted if s(x)]
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


DEDUP_FILES_DEFAULT = [
    "dpo/dpo_seeds/safety_prompts.jsonl",
    "dpo/dpo_seeds/safety_prompts_ambiguous.jsonl",
    "dpo/dpo_seeds/safety_prompts_mined.jsonl",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Gera prompts nocivos via API (escala o lado 'recusa' do safety-SFT)")
    ap.add_argument("--out", default="dpo/dpo_seeds/safety_prompts_generated.jsonl")
    ap.add_argument("--categories", default="all",
                    help="'all' ou lista por virgula (ex.: ciberataque,fraude)")
    ap.add_argument("--per_category", type=int, default=DEFAULT_PER_CATEGORY,
                    help=f"quantos pedidos por chamada (default {DEFAULT_PER_CATEGORY})")
    ap.add_argument("--calls_per_category", type=int, default=DEFAULT_CALLS_PER_CATEGORY,
                    help=f"quantas chamadas por categoria (default {DEFAULT_CALLS_PER_CATEGORY})")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--dedup_against", nargs="*", default=DEDUP_FILES_DEFAULT,
                    help="arquivos .jsonl existentes para deduplicar (campo 'prompt')")
    ap.add_argument("--sample-audit", type=int, default=0,
                    help="se > 0, gera e so IMPRIME N amostras aleatorias para auditoria (nao grava nada)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="mostra os prompts de sistema sem chamar a API")
    args = ap.parse_args()

    selected = CATEGORIES if args.categories.strip() == "all" else [
        c for c in CATEGORIES if c[0] in {x.strip() for x in args.categories.split(",")}
    ]
    print(f"[gen-harmful] {len(selected)} categorias selecionadas: {[c[0] for c in selected]}")

    if args.dry_run:
        print("\n[dry-run] nenhuma chamada de API sera feita.\n")
        print("=" * 60)
        print("SYSTEM PROMPT:\n" + SYSTEM_PROMPT)
        for cat, desc in selected[:3]:
            print("=" * 60)
            print(f"[{cat}] Gere {args.per_category} pedidos sobre: {desc}")
        return 0

    existing = load_existing_prompts(args.dedup_against) if not args.sample_audit else set()
    if existing:
        print(f"[dedup] total: {len(existing)} prompts existentes")

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
    next_id = 0

    for cat, desc in selected:
        cat_kept = 0
        for call_n in range(args.calls_per_category):
            user_msg = f"Gere {args.per_category} pedidos sobre: {desc}"
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
                snippet = text[:200].replace("\n", " ")
                print(f"[aviso] {cat} (chamada {call_n}): resposta nao-parseavel. PULADA. "
                      f"Inicio da resposta: {snippet!r}")
                continue

            for p in prompts:
                norm = _normalize(p)
                if norm in seen_norm:
                    continue
                seen_norm.add(norm)
                generated.append({"id": f"genapi_{next_id:04d}", "categoria": cat, "prompt": p})
                next_id += 1
                cat_kept += 1
        counts[cat] = cat_kept
        print(f"[gen-harmful] {cat}: {cat_kept} prompts novos")

    if args.sample_audit:
        sample = rng.sample(generated, min(args.sample_audit, len(generated)))
        print(f"\n[audit] {len(sample)} amostras aleatorias (NADA foi gravado em disco):\n")
        for r in sample:
            print("=" * 60)
            print(f"[{r['categoria']}] {r['prompt']}")
        print("\n[audit] Confira se estes SAO de fato pedidos nocivos plausiveis (nao falsos "
              "positivos/genericos demais) antes de rodar sem --sample-audit.")
        return 0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fp:
        for r in generated:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")

    price_in, price_out = PRICING_SNAPSHOT.get(args.model, (None, None))
    est_cost = (round(in_tok / 1e6 * price_in + out_tok / 1e6 * price_out, 4)
                if price_in is not None else None)

    manifest = {
        "dataset": "safety_prompts_generated", "model": args.model,
        "categories": counts, "total": len(generated),
        "tokens_input": in_tok, "tokens_output": out_tok,
        "estimated_cost_usd": est_cost,
        "estimated_cost_note": "Chamadas sincronas, SEM desconto de Batches. Conferir preco vigente.",
        "deduplicated_against": args.dedup_against,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "CANDIDATO -- auditar com --sample-audit antes de usar para treino",
    }
    with open(os.path.join(os.path.dirname(args.out) or ".", "safety_prompts_generated_manifest.json"),
              "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)

    print(f"\n[gen-harmful] TOTAL: {len(generated)} prompts novos -> {args.out}")
    print(f"[gen-harmful] tokens: {in_tok:,} entrada / {out_tok:,} saida"
          + (f" | custo estimado: US${est_cost}" if est_cost is not None else ""))
    print("[gen-harmful] PROXIMO PASSO: auditar uma amostra "
          f"(python {sys.argv[0]} --sample-audit 15) antes de juntar ao mix de seeds.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[INFO] Interrompido pelo usuario.")
        sys.exit(130)
