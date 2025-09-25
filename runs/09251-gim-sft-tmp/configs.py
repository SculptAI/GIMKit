import pathlib

import torch


EXP_NAME = pathlib.Path(__file__).resolve().parent.name

ARTIFACTS_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "artifacts" / EXP_NAME
FINAL_MODEL_DIR = ARTIFACTS_DIR / "sft-gim"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 0

BASE_MODEL_NAME = "unsloth/Qwen3-4B-Instruct-2507"
MAX_SEQ_LENGTH = 4096

DATASET_NAME = "Ki-Seki/GIM-SFT"
DATASET_LEN = 100_000
TRAIN_SPLIT = 0.98
TRAIN_SIZE = int(DATASET_LEN * TRAIN_SPLIT)

NUM_GPUS = torch.cuda.device_count()
MICRO_BSZ = 8
GRAD_ACCUM = 4
GLOBAL_BSZ = MICRO_BSZ * GRAD_ACCUM * NUM_GPUS

ESTIMATED_STEPS = (TRAIN_SIZE // GLOBAL_BSZ) * 1
WARMUP_STEPS = max(1, ESTIMATED_STEPS // 20)
SAVE_STEPS = max(1, ESTIMATED_STEPS // 10)
EVAL_STEPS = SAVE_STEPS
