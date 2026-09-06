#!/usr/bin/env python3
"""
Manacá-1B-Instruct-2 — SFT a partir de personas brasileiras (NVIDIA Nemotron)
=================================================================================
PT
--
`nvidia/Nemotron-Personas-Brazil` (CC BY 4.0) é 6 milhões de personas
sintéticas em português, ancoradas em distribuições demográficas reais do
IBGE (idade, gênero, escolaridade, ocupação, região, cultura, hobbies,
objetivos). Diferente de tudo já usado no projeto, não é texto cru nem QA
pronto — é matéria-prima para gerar instrução DIVERSA e culturalmente
brasileira via API do Claude, evitando o problema comum de SFT sintético
ficar genérico/"americanizado".

COMO FUNCIONA: para cada persona sorteada, pede ao Claude para gerar UM
pedido plausível que essa pessoa faria a um assistente de IA (coerente com
o perfil: idade, ocupação, região, objetivos) E uma resposta útil e
grounded no que um assistente real responderia -- tudo em uma única
chamada, schema JSON (mesmo padrão do `synth/generate_instructions.py`,
mas condicionado à persona em vez de a uma passagem de corpus).

POR QUE CHAMADAS SÍNCRONAS (não Batches API): teto de personas por rodada
é da ordem de milhares/dezenas de milhares (`--num_personas`), não milhões
-- volume compatível com chamadas diretas, mesmo padrão de
`sft/safety_seeds/generate_benign_prompts.py` e
`dpo/dpo_seeds/generate_harmful_prompts.py`.

AMOSTRAGEM: `Nemotron-Personas-Brazil` tem 6M linhas -- carregar tudo na
RAM não é viável. Usa reservoir sampling (Algorithm R) em UMA passada de
streaming, sem viés de posição no arquivo.

DECONTAMINAÇÃO: mesmo mecanismo do `synth/generate_instructions.py`
(arquivos estáticos deste repo: bench/, dpo/dpo_seeds/, sft/safety_seeds/).

Uso:
    export ANTHROPIC_API_KEY=sk-ant-...
    python sft/personas_br/generate_personas_sft.py --dry-run           # sem custo
    python sft/personas_br/generate_personas_sft.py --num_personas 2000

Saída: `data/sft/personas_br/personas_br.jsonl` (schema Alpaca:
`{instruction, input, output, task}`) + manifest com custo estimado.

EN
--
Turns NVIDIA's Nemotron-Personas-Brazil (6M synthetic PT-BR personas
anchored in real IBGE demographics, CC BY 4.0) into diverse, culturally
grounded SFT instruction+response pairs via the Claude API -- one call per
sampled persona, same JSON pattern as synth/generate_instructions.py but
conditioned on a persona description instead of a corpus passage.

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

HF_REPO_ID = "nvidia/Nemotron-Personas-Brazil"
LICENSE    = "cc-by-4.0"
REFERENCE  = "NVIDIA (2026). Nemotron-Personas-Brazil."

DEFAULT_MODEL = os.environ.get("MODEL", "claude-opus-5")
DEFAULT_NUM_PERSONAS = 2000
# CUSTO REAL confirmado ao vivo: sem `output_config.effort` explicito, o
# thinking adaptativo (ligado por padrao em claude-sonnet-5/claude-opus-5
# quando o param `thinking` e omitido) consome boa parte do orcamento de
# max_tokens em raciocinio invisivel -- numa rodada real de 500 personas,
# a MEDIA de output foi ~1191 tokens/chamada (quase o teto), MESMO incluindo
# as 452 chamadas que so deveriam responder {"skip": true} (deveria ser
# quase de graca). Isso tambem provavelmente inflou a taxa de skip: o
# modelo fica sem espaco de token pra escrever uma resposta boa. Fix:
# --effort default "low" (tarefa simples, nao precisa de raciocinio
# profundo) + teto um pouco mais alto pra nao truncar a resposta visivel.
DEFAULT_MAX_TOKENS = 2000

PRICING_SNAPSHOT = {
    "claude-opus-5":    (5.00, 25.00),
    "claude-sonnet-5":  (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

DECONTAMINATION_SOURCES = [
    ("bench/ifeval_pt/prompts.jsonl", "turns"),
    ("bench/mtbench_pt/questions.jsonl", "turns"),
    ("bench/safety_pt/heldout.jsonl", "prompt"),
    ("bench/overrefusal_pt/prompts.jsonl", "prompt"),
    ("dpo/dpo_seeds/safety_prompts.jsonl", "prompt"),
    ("sft/safety_seeds/benign_help.jsonl", "prompt"),
]

SYSTEM_PROMPT = """Você gera UM exemplo de treino de instrução para um assistente de IA em \
português do Brasil, a partir da descrição de uma persona (pessoa fictícia, mas com perfil \
demográfico realista).

Sua tarefa:
1. Gere um PEDIDO (pergunta, tarefa ou pedido de ajuda) que essa persona especificamente \
faria a um assistente de IA -- coerente com a idade, ocupação, região, escolaridade, \
objetivos e interesses dela. Não mencione a persona no pedido; o pedido deve soar como algo \
que uma pessoa real digitaria, na primeira pessoa, sem se descrever.
2. Gere uma RESPOSTA útil, precisa e apropriada ao pedido -- a resposta que um bom assistente \
de IA daria.

Regras obrigatórias:
- Tudo em português do Brasil.
- O pedido deve ser plausível e específico (não genérico como "me ajude com algo") -- reflita \
a vida real e as circunstâncias da persona.
- NÃO use emojis na resposta, nem quando o contexto for informal (ex.: mensagem de WhatsApp, \
post de rede social) -- transmita o mesmo tom só com palavras.
- Se a persona não render um pedido plausível e específico, responda {"skip": true}.
- Responda APENAS com um objeto JSON válido, sem markdown, sem texto antes ou depois, no \
formato exato: {"instruction": "...", "output": "..."}"""


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
            continue
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
    return seen


def is_contaminated(instruction: str, decontam_set: set[str]) -> bool:
    if not decontam_set:
        return False
    norm = _normalize(instruction)
    if norm in decontam_set:
        return True
    return any(norm in bench or bench in norm for bench in decontam_set if len(bench) > 20)


# Faixas Unicode de emoji (pictogramas, emoticons, dingbats, simbolos
# diversos, bandeiras). Conta CARACTERES de emoji, nao "sequencias" (um
# emoji composto com ZWJ conta como varios) -- suficiente para detectar
# presenca, sem precisar de uma lib externa so pra isso.
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001F5FF"  # simbolos/pictogramas diversos
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F680-\U0001F6FF"   # transporte/mapas
    "\U0001F900-\U0001F9FF"   # simbolos/pictogramas suplementares
    "\U0001FA70-\U0001FAFF"   # simbolos/pictogramas extendidos-A
    "\U00002600-\U000026FF"   # simbolos diversos
    "\U00002700-\U000027BF"   # dingbats
    "\U00002B00-\U00002BFF"   # simbolos/setas diversas (inclui ⭐ estrela)
    "\U0001F1E6-\U0001F1FF"   # indicadores regionais (bandeiras)
    "]"
)
# O tokenizer do manaca-1b-base e um SentencePiece Unigram treinado do zero
# SO no corpus PT-BR do projeto (character_coverage=0.9995, byte_fallback=
# True -- ver tokenizer/scripts/02_train_spm.py do manaca-1b-base). Emoji
# nao tem token dedicado: cada um vira ~4 tokens de fallback por byte UTF-8,
# sem NENHUMA exposicao no pre-treino. Nao e so "excesso que soa artificial"
# -- e token fora de distribuicao, caro (4x mais tokens que o normal) e
# arriscado (o modelo pode gerar so parte da sequencia de bytes e produzir
# UTF-8 invalido). Por isso o teto e ZERO, nao um limite de densidade.
MAX_EMOJIS_PER_OUTPUT = 0


def count_emojis(text: str) -> int:
    return len(_EMOJI_RE.findall(text))


def reservoir_sample(iterable, k: int, rng: random.Random) -> list:
    """Amostragem por reservatório (Algorithm R): 1 passada, sem carregar tudo na RAM."""
    sample: list = []
    for i, item in enumerate(iterable):
        if i < k:
            sample.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                sample[j] = item
    return sample


def persona_to_text(row: dict) -> str:
    """Concatena os campos textuais de uma linha de persona (schema tolerante)."""
    candidates = ["persona", "persona_description", "description", "text",
                  "skills_and_expertise", "professional_persona", "hobbies_and_interests",
                  "culture_and_values", "life_story", "goals_and_plans"]
    parts = [s(row.get(c)) for c in candidates if s(row.get(c))]
    if not parts:
        # fallback: concatena qualquer valor string do dict
        parts = [s(v) for v in row.values() if isinstance(v, str) and s(v)]
    return "\n".join(parts)


def parse_json_obj(raw_text: str) -> dict | None:
    text = raw_text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
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


_FATAL_MSG_MARKERS = ("credit", "balance", "quota", "insufficient")


def _is_fatal_credit_error(exc: Exception) -> bool:
    """Detecta erro de saldo/creditos esgotados (HTTP 400/403 da Anthropic
    com mensagem mencionando credito/saldo/quota) -- NAO um erro generico de
    requisicao invalida, que deve continuar sendo tratado como falha pontual
    (PULADA) e nao interromper a rodada inteira."""
    import anthropic
    if not isinstance(exc, anthropic.APIStatusError):
        return False
    status = getattr(exc, "status_code", None)
    if status not in (400, 403):
        return False
    msg = str(exc).lower()
    return any(marker in msg for marker in _FATAL_MSG_MARKERS)


def load_manifest(out_dir: str) -> dict:
    path = os.path.join(out_dir, "manifest.json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fp:
                return json.load(fp)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_manifest_atomic(out_dir: str, manifest: dict) -> None:
    path = os.path.join(out_dir, "manifest.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Gera SFT (instrucao+resposta) a partir de personas brasileiras via Claude")
    ap.add_argument("--out", default="data/sft/personas_br", help="diretorio de saida")
    ap.add_argument("--num_personas", type=int, default=DEFAULT_NUM_PERSONAS,
                    help=f"quantas personas amostrar/processar (default {DEFAULT_NUM_PERSONAS})")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"],
                    help="output_config.effort -- 'low' evita gasto de thinking em excesso numa tarefa "
                         "simples (default 'low'; ver nota de custo no README)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="mostra o prompt de sistema e uma persona amostrada, sem chamar a API")
    ap.add_argument("--overwrite", action="store_true",
                    help="permite sobrescrever um lote existente em --out cujo status/parametros "
                         "nao permitem retomada (sem isso, o comando para com erro em vez de "
                         "truncar silenciosamente um lote ja pago e completo)")
    args = ap.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rng = random.Random(args.seed)

    try:
        from datasets import load_dataset
    except ImportError as e:
        print(f"[ERRO] 'datasets' ausente ({e}). Rode na imagem manaca-posttrain.")
        return 1

    print(f"[personas-br] streaming {HF_REPO_ID} para reservoir-sample de {args.num_personas:,} personas...")
    try:
        ds = load_dataset(HF_REPO_ID, split="train", streaming=True)
        personas = reservoir_sample(ds, args.num_personas, rng)
    except Exception as e:
        print(f"[ERRO] falha ao carregar/amostrar {HF_REPO_ID}: {e}")
        return 1
    print(f"[personas-br] {len(personas):,} personas amostradas")

    if args.dry_run:
        print("\n[dry-run] nenhuma chamada de API sera feita.\n")
        print("=" * 60)
        print("SYSTEM PROMPT:\n" + SYSTEM_PROMPT)
        if personas:
            print("=" * 60)
            print("EXEMPLO DE PERSONA (1a amostrada):\n" + persona_to_text(personas[0])[:1500])
        return 0

    decontam = load_decontamination_set(repo_root)
    print(f"[personas-br] decontaminacao: {len(decontam):,} prompts estaticos carregados")

    try:
        import anthropic
    except ImportError as e:
        print(f"[ERRO] 'anthropic' ausente ({e}). Rode: pip install -r requirements/corpus.txt")
        return 1

    client = anthropic.Anthropic()
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "personas_br.jsonl")

    # ── Retomada: uma rodada anterior pode ter parado no meio (creditos
    # esgotados, Ctrl-C, container derrubado). O reservoir sample e
    # deterministico (mesma seed + mesmo num_personas + mesmo stream), entao
    # "personas[i]" e igual em toda rodada -- basta gravar ate onde
    # chegamos (processed_idx) e, ao retomar, pular os indices ja
    # tentados e ABRIR O ARQUIVO EM MODO APPEND (nunca sobrescrever o que
    # ja foi pago e gravado).
    prev = load_manifest(args.out)
    can_resume = (
        prev.get("status") in ("in_progress", "stopped_fatal_error")
        and prev.get("personas_sampled") == len(personas)
        and prev.get("seed") == args.seed
        and prev.get("model") == args.model
    )
    processed_idx = prev.get("processed_idx", -1) if can_resume else -1
    n = prev.get("total", 0) if can_resume else 0
    n_skip_model = prev.get("skipped_model_decidiu", 0) if can_resume else 0
    n_skip_unparseable = prev.get("skipped_json_invalido", 0) if can_resume else 0
    n_skip_contam = prev.get("skipped_contaminado", 0) if can_resume else 0
    n_skip_emoji_heavy = prev.get("skipped_emoji_pesado", 0) if can_resume else 0
    n_skip_api_fail = prev.get("skipped_api_failure", 0) if can_resume else 0
    in_tok = prev.get("tokens_input", 0) if can_resume else 0
    out_tok = prev.get("tokens_output", 0) if can_resume else 0
    if can_resume:
        print(f"[personas-br] RETOMANDO: {processed_idx + 1:,}/{len(personas):,} personas ja "
              f"tentadas, {n:,} exemplos ja gravados em {out_path}")

    SAVE_EVERY = 20  # checkpoint do manifest a cada N personas tentadas

    def save_manifest(status: str) -> None:
        price_in, price_out = PRICING_SNAPSHOT.get(args.model, (None, None))
        est_cost = (round(in_tok / 1e6 * price_in + out_tok / 1e6 * price_out, 4)
                    if price_in is not None else None)
        save_manifest_atomic(args.out, {
            "dataset": "personas_br", "hf_repo": HF_REPO_ID, "license": LICENSE, "reference": REFERENCE,
            "model": args.model, "seed": args.seed, "personas_sampled": len(personas),
            "status": status, "processed_idx": last_completed_idx, "total": n,
            "skipped_model_decidiu": n_skip_model, "skipped_json_invalido": n_skip_unparseable,
            "skipped_contaminado": n_skip_contam, "skipped_emoji_pesado": n_skip_emoji_heavy,
            "skipped_api_failure": n_skip_api_fail,
            "tokens_input": in_tok, "tokens_output": out_tok, "estimated_cost_usd": est_cost,
            "estimated_cost_note": "Chamadas sincronas, SEM desconto de Batches. Conferir preco vigente.",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })

    # Trava de seguranca: um lote anterior em --out so pode ser retomado
    # (append) se can_resume for True (status + seed/model/personas_sampled
    # batendo exatamente); qualquer outro caso (lote 'completed', ou
    # parametros diferentes) cairia em modo "w" e apagaria silenciosamente
    # exemplos JA PAGOS. Confirmado ao vivo (mesmo bug na versao "irma" deste
    # script, synth/generate_instructions_local.py, que apagou um lote de
    # 891 exemplos ao rodar de novo com --seed diferente sobre o mesmo
    # --out). Falha cedo em vez de truncar sem avisar.
    if not can_resume and not args.overwrite and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"[ERRO] {out_path} ja existe (lote anterior: status={prev.get('status')!r}, "
              f"total={prev.get('total')}, seed={prev.get('seed')}) e os parametros deste comando "
              "nao permitem retomada -- rodar assim SOBRESCREVERIA exemplos ja pagos. Use "
              "--out <outro-diretorio> para um lote novo separado (preservando o anterior), ou "
              "--overwrite se a intencao e mesmo descartar o que esta em " + out_path + ".")
        return 1

    last_completed_idx = processed_idx
    stop_reason = None
    file_mode = "a" if can_resume else "w"
    with open(out_path, file_mode, encoding="utf-8") as fp:
        for i, row in enumerate(personas):
            if i <= processed_idx:
                continue
            persona_text = persona_to_text(row)
            if not persona_text:
                last_completed_idx = i
                continue
            try:
                resp = call_with_retry(
                    client, model=args.model, max_tokens=args.max_tokens,
                    system=SYSTEM_PROMPT,
                    output_config={"effort": args.effort},
                    messages=[{"role": "user", "content": f"Persona:\n{persona_text[:3000]}"}],
                )
            except Exception as e:
                if _is_fatal_credit_error(e):
                    print(f"\n[personas-br] ERRO FATAL na persona {i} (provavel esgotamento de "
                          f"creditos/quota): {e}")
                    print(f"[personas-br] parando aqui -- {n:,} exemplos ja gravados em {out_path}. "
                          "Rode o MESMO comando de novo apos resolver o credito para retomar.")
                    stop_reason = "fatal_api_error"
                    break
                print(f"[aviso] persona {i}: falha na API ({e}). PULADA.")
                n_skip_api_fail += 1
                last_completed_idx = i
                continue

            in_tok += resp.usage.input_tokens
            out_tok += resp.usage.output_tokens
            text = next((b.text for b in resp.content if b.type == "text"), "")
            obj = parse_json_obj(text)
            if not obj or obj.get("skip"):
                n_skip_model += 1
                last_completed_idx = i
                continue
            instr, out = s(obj.get("instruction")), s(obj.get("output"))
            if not instr or not out:
                n_skip_unparseable += 1
                last_completed_idx = i
                continue
            if is_contaminated(instr, decontam):
                n_skip_contam += 1
                last_completed_idx = i
                continue
            if count_emojis(out) > MAX_EMOJIS_PER_OUTPUT:
                n_skip_emoji_heavy += 1
                last_completed_idx = i
                continue

            fp.write(json.dumps({"instruction": instr, "input": "", "output": out, "task": "personas_br"},
                                 ensure_ascii=False) + "\n")
            fp.flush()
            n += 1
            last_completed_idx = i
            if (i + 1) % 100 == 0:
                print(f"[personas-br] {i + 1}/{len(personas)} processadas, {n} gravadas")
            if (i + 1) % SAVE_EVERY == 0:
                save_manifest("in_progress")

    save_manifest("stopped_fatal_error" if stop_reason else "completed")

    price_in, price_out = PRICING_SNAPSHOT.get(args.model, (None, None))
    est_cost = (round(in_tok / 1e6 * price_in + out_tok / 1e6 * price_out, 4)
                if price_in is not None else None)
    print(f"\n[personas-br] TOTAL: {n:,} exemplos -> {out_path}")
    print(f"[personas-br] tokens: {in_tok:,} entrada / {out_tok:,} saida"
          + (f" | custo estimado: US${est_cost}" if est_cost is not None else ""))
    if stop_reason:
        print(f"[personas-br] PARADO ANTES DO FIM (motivo: {stop_reason}) -- rode o mesmo comando "
              "de novo para retomar de onde parou.")
        return 2
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[INFO] Interrompido pelo usuario.")
        sys.exit(130)
