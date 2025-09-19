import pathlib

import torch


EXP_NAME = pathlib.Path(__file__).resolve().parent.name
EXP_DESC = "Masked tag id starts from 0; same as the initial run configuration"

ARTIFACTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "artifacts" / EXP_NAME
FINAL_MODEL_DIR = ARTIFACTS_DIR / "sft-gim"

RANDOM_SEED = 0

DATASET_LEN = 50000
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
