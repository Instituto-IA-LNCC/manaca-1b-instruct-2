# =============================================================================
# Manacá-1B-Instruct — pós-treino (SFT) reprodutível. Atalhos do que VALEU.
# -----------------------------------------------------------------------------
# Só os alvos do instruct v1 e v2 (o que funcionou). A base (pré-treino/aval.)
# está em github.com/Instituto-IA-LNCC/manaca-1b-base.
#
#   cp .env.example .env    # ajuste CKPT_DIR, DATA_DIR, HF_TOKEN, GPUS_PER_NODE=2
#   make build-posttrain    # imagem GPU (trl/accelerate/deepspeed)
#   make sft-data           # dados v1  |  make sft-data-v2  -> dados v2
#   make sft ARGS="..."     # SFT v1    |  make sft-v2       -> SFT v2
# =============================================================================
SHELL := /bin/bash
DC := docker compose
.PHONY: help env build-posttrain sft-data sft-data-v2 sft sft-v2 sft-safety dpo corpus-legalpt corpus-classiccpt corpus-carolina corpus-all gigaverbo-v2-sft-data corpus-madlad400 corpus-fineweb2 corpus-hplt2 corpus-portuguese-pd corpus-all-raw tucano-sft-data quati-data dpo-gigaverbo-v2-preferences dpo-reward-aira dpo-preferences-data synth-run synth-submit synth-collect synth-dry-run
.RECIPEPREFIX := >

help:  ## Lista os alvos
> @grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

env:  ## Cria .env a partir do exemplo, se faltar
> @[ -f .env ] || { cp .env.example .env && echo "[env] .env criado a partir de .env.example — ajuste os caminhos"; }

build-posttrain: env  ## Constrói a imagem GPU de SFT (Fase 3)
> $(DC) build posttrain

sft-data:  ## Dados de SFT v1 (manaca-jaster + alpaca + oasst) -> data/sft
> ./scripts/docker/run_posttrain.sh bash -lc 'python sft/jaster/build_jaster.py --tasks default --out data/sft/jaster --shuffle && python sft/prepare_data.py --sources alpaca,oasst --out data/sft --shuffle --dedup --extra data/sft/jaster/manaca_jaster.jsonl'

sft-data-v2:  ## Dados de SFT v2 (OASST corrigido + aya + traducao + sumarizacao + jaster completo; filtro de longos) -> data/sft_v2
> ./scripts/docker/run_posttrain.sh bash -lc 'python sft/jaster/build_jaster.py --tasks all --out data/sft_v2/jaster --shuffle --max_chars 5000 --max_per_task 8000 && python sft/prepare_data.py --sources alpaca,aya,oasst,translation,summarization --out data/sft_v2 --shuffle --dedup --max_chars 5000 --extra data/sft_v2/jaster/manaca_jaster.jsonl'

sft:  ## SFT (Fase 3a): make sft ARGS="..."  (DETACH=1 para segundo plano)
> ./scripts/docker/run_sft.sh sft $(ARGS)

sft-v2:  ## SFT v2 (full FT) -> checkpoints/manaca-1b-instruct-v2-full (NAO sobrescreve a v1). DETACH=1 p/ 2o plano
> ./scripts/docker/run_sft.sh sft --model_name_or_path menezesbruno/manaca-1b-base --data_files data/sft_v2/manaca_sft.jsonl --full_finetuning --gradient_accumulation_steps 128 --output_dir /workspace/checkpoints/manaca-1b-instruct-v2-full $(ARGS)

sft-safety:  ## Safety-SFT (LoRA sobre o v2, DDP): recusa o nocivo sem virar recusador cego -> instruct-v2-safe (o "safe"). Ver docs/evaluation/safety-alignment-pt.md
> ACCELERATE_CONFIG=$${ACCELERATE_CONFIG:-configs/accelerate_config_ddp.yaml} ./scripts/docker/run_sft.sh sft --model_name_or_path /workspace/checkpoints/manaca-1b-instruct-v2-full --data_files sft_safety/safety_sft.jsonl --output_dir /workspace/checkpoints/manaca-1b-instruct-v2-safe --lora_r 32 --lora_alpha 64 --num_train_epochs 3 --learning_rate 1e-4 --gradient_accumulation_steps 8 --save_steps 200 $(ARGS)

dpo:  ## DPO on-policy (RESULTADO NEGATIVO documentado; nao mudou a geracao). Ver dpo/README-onpolicy.md
> ACCELERATE_CONFIG=$${ACCELERATE_CONFIG:-configs/accelerate_config_ddp.yaml} ./scripts/docker/run_sft.sh dpo $(ARGS)

corpus-legalpt:  ## Aquisicao do LegalPT (menos tesemo_v2, ja no Tier 1 do base) -> $WORK_DIR/raw/legalpt
> ./scripts/docker/run_posttrain.sh python corpus/scripts/01_acquire_legalpt.py $(ARGS)

corpus-classiccpt:  ## Aquisicao do ClassiCC-PT (filtrado por edu_score/toxic_score) -> $WORK_DIR/raw/classiccpt
> ./scripts/docker/run_posttrain.sh python corpus/scripts/02_acquire_classiccpt.py $(ARGS)

corpus-carolina:  ## Aquisicao do Carolina Corpus v1.2 (registro/estilo, CC BY-NC-SA) -> $WORK_DIR/raw/carolina
> ./scripts/docker/run_posttrain.sh python corpus/scripts/03_acquire_carolina.py $(ARGS)

corpus-all:  ## Roda os 3 scripts de aquisicao em sequencia (LegalPT -> ClassiCC-PT -> Carolina)
> $(MAKE) corpus-legalpt
> $(MAKE) corpus-classiccpt
> $(MAKE) corpus-carolina

gigaverbo-v2-sft-data:  ## Converte Polygl0t/gigaverbo-v2-sft (Apache-2.0, ~4M ex.) -> data/sft/gigaverbo_v2_sft (Alpaca)
> ./scripts/docker/run_posttrain.sh python sft/gigaverbo_v2_sft/build_gigaverbo_v2_sft.py --shuffle $(ARGS)

corpus-madlad400:  ## Aquisicao do MADLAD-400/pt (Apache-2.0, livre do base) -> $WORK_DIR/raw/madlad400
> ./scripts/docker/run_posttrain.sh python corpus/scripts/04_acquire_madlad400.py $(ARGS)

corpus-fineweb2:  ## Aquisicao do FineWeb-2/por_Latn (ODC-By, livre do base) -> $WORK_DIR/raw/fineweb2
> ./scripts/docker/run_posttrain.sh python corpus/scripts/05_acquire_fineweb2.py $(ARGS)

corpus-hplt2:  ## Aquisicao do HPLT 2.0/pt (CC0, livre do base) -> $WORK_DIR/raw/hplt2
> ./scripts/docker/run_posttrain.sh python corpus/scripts/06_acquire_hplt2.py $(ARGS)

corpus-portuguese-pd:  ## Aquisicao do Portuguese-PD (dominio publico, registro historico) -> $WORK_DIR/raw/portuguese_pd
> ./scripts/docker/run_posttrain.sh python corpus/scripts/07_acquire_portuguese_pd.py $(ARGS)

corpus-all-raw:  ## Roda os 7 scripts de aquisicao bruta em sequencia (01-07)
> $(MAKE) corpus-legalpt
> $(MAKE) corpus-classiccpt
> $(MAKE) corpus-carolina
> $(MAKE) corpus-madlad400
> $(MAKE) corpus-fineweb2
> $(MAKE) corpus-hplt2
> $(MAKE) corpus-portuguese-pd

tucano-sft-data:  ## Converte TucanoBR/Tucano-SFT (Apache-2.0/MIT) -> data/sft/tucano_sft (Alpaca)
> ./scripts/docker/run_posttrain.sh python sft/tucano_sft/build_tucano_sft.py --shuffle $(ARGS)

quati-data:  ## Constroi a tarefa de relevancia Quati (CC BY 4.0, nativo) -> data/sft/quati (Alpaca)
> ./scripts/docker/run_posttrain.sh python sft/quati/build_quati_task.py --shuffle $(ARGS)

dpo-gigaverbo-v2-preferences:  ## Converte Polygl0t/gigaverbo-v2-preferences (Apache-2.0) -> dpo_onpolicy/gigaverbo_v2_preferences
> ./scripts/docker/run_posttrain.sh python dpo/gigaverbo_v2_preferences/build_gigaverbo_v2_preferences.py $(ARGS)

dpo-reward-aira:  ## Converte nicholasKluge/reward-aira-dataset (Apache-2.0) -> dpo_onpolicy/reward_aira
> ./scripts/docker/run_posttrain.sh python dpo/reward_aira/build_reward_aira.py $(ARGS)

dpo-preferences-data:  ## Roda os 2 conversores de preferencia (2a tentativa de DPO, ver dpo/README-onpolicy.md)
> $(MAKE) dpo-gigaverbo-v2-preferences
> $(MAKE) dpo-reward-aira

synth-dry-run:  ## Mostra os prompts de geracao sintetica sem gastar (make synth-dry-run ARGS="--source X --max-examples 5")
> ./scripts/docker/run_posttrain.sh python synth/generate_instructions.py run --dry-run $(ARGS)

synth-run:  ## Gera instrucao sintetica via Batches API: submete+espera+coleta (make synth-run ARGS="--source X --max-examples N")
> ./scripts/docker/run_posttrain.sh python synth/generate_instructions.py run $(ARGS)

synth-submit:  ## So submete o(s) lote(s) e sai (nao espera ate 24h) -- ver synth/README.md
> ./scripts/docker/run_posttrain.sh python synth/generate_instructions.py submit $(ARGS)

synth-collect:  ## Busca resultados de um lote ja submetido (make synth-collect ARGS="--state data/synth/X/batch_state.json")
> ./scripts/docker/run_posttrain.sh python synth/generate_instructions.py collect $(ARGS)
