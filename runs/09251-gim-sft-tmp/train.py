from unsloth import FastModel  # noqa: I001

import logging
import os

import configs

from datasets import concatenate_datasets, load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from gimkit import Query, guide
from datasets import Dataset


# ─── General Setup ────────────────────────────────────────────────────────────

os.environ["SWANLAB_EXP_NAME"] = configs.EXP_NAME
os.environ["SWANLAB_LOG_DIR"] = str(configs.ARTIFACTS_DIR / "swanlog")

logging.basicConfig(
    filename=configs.ARTIFACTS_DIR / "training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.info("Training configurations:")
for key, value in vars(configs).items():
    if not key.startswith("__"):
        logging.info(f"{key} = {value}")

# ─── Load Model And Tokenizer ─────────────────────────────────────────────────

model, tokenizer = FastModel.from_pretrained(
    model_name=configs.BASE_MODEL_NAME,
    max_seq_length=configs.MAX_SEQ_LENGTH,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,
    token=None,
)

model = FastModel.get_peft_model(
    model,
    r=32,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=configs.RANDOM_SEED,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="qwen3-instruct",
)

# ─── Load Dataset ─────────────────────────────────────────────────────────────

# fmt: off
high_subsets = [        # 23294 in total
    "gsm8k_reasoning",  # 1254
    "hk_o1aw",          # 14363
    "lima",             # 1030
    "o1_journey",       # 327
    "process_bench",    # 1179
    "uhgeval",          # 5141
]
mid_subsets = [         # 437113 in total
    "cnn_daily_mail",   # 287113
    "magpie_reasoning", # 150000
]
low_subsets = [         # 2697422 in total
    "kaist_cot",        # 1837928
    "numina_math",      # 859494
]
# fmt: on


def _concat_subsets(subsets: list[str]) -> Dataset:
    return concatenate_datasets(
        [load_dataset(configs.DATASET_NAME, subset, split="train") for subset in subsets]
    )


logging.info("Loading and preparing dataset...")
high_dataset = _concat_subsets(high_subsets)
if configs.DATASET_LEN - len(high_dataset) > 0:
    _rest_len = configs.DATASET_LEN - len(high_dataset)
    _mid_len = int(_rest_len * 0.6)
    _low_len = _rest_len - _mid_len
    mid_dataset = (
        _concat_subsets(mid_subsets).shuffle(seed=configs.RANDOM_SEED).select(range(_mid_len))
    )
    low_dataset = (
        _concat_subsets(low_subsets).shuffle(seed=configs.RANDOM_SEED).select(range(_low_len))
    )

    assert len(high_dataset) + len(mid_dataset) + len(low_dataset) == configs.DATASET_LEN
    dataset = concatenate_datasets([high_dataset, mid_dataset, low_dataset])
    logging.info(
        f"Dataset sizes: high {len(high_dataset)}, mid {len(mid_dataset)}, low {len(low_dataset)}"
    )

else:
    dataset = high_dataset.shuffle(seed=configs.RANDOM_SEED).select(range(configs.DATASET_LEN))
    logging.info(f"Dataset sizes: high {len(dataset)}")

dataset = (
    dataset.shuffle(seed=configs.RANDOM_SEED)
    .map(
        lambda example: {
            "text": tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": example["gim_query"]},
                    {"role": "assistant", "content": example["gim_response"]},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
        },
        num_proc=os.cpu_count(),
    )
    .select_columns(["text"])
)

# ─── Training ─────────────────────────────────────────────────────────────────

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset.select(range(configs.TRAIN_SIZE)),
    eval_dataset=dataset.select(range(configs.TRAIN_SIZE, configs.DATASET_LEN)),
    args=SFTConfig(
        output_dir=configs.ARTIFACTS_DIR,
        dataset_text_field="text",
        per_device_train_batch_size=configs.MICRO_BSZ,
        gradient_accumulation_steps=configs.GRAD_ACCUM,
        eval_strategy="steps",
        eval_steps=configs.EVAL_STEPS,
        num_train_epochs=1,  # Set this for 1 full training run.
        max_steps=-1,
        warmup_steps=configs.WARMUP_STEPS,
        learning_rate=2e-4,  # Reduce to 2e-5 for long training runs
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_steps=configs.SAVE_STEPS,
        optim="adamw_8bit",
        weight_decay=0.01,
        seed=configs.RANDOM_SEED,
        report_to="swanlab",
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)

trainer_stats = trainer.train()

# ─── Inference ────────────────────────────────────────────────────────────────

messages = [{"role": "user", "content": str(Query(f"This is an {guide()} text."))}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,  # Must add for generation
)

response = model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    max_new_tokens=256,  # Increase for longer outputs!
    temperature=0.7,
    top_p=0.8,
    top_k=20,  # For non thinking
)
logging.info("Request: " + text)
logging.info("Response: " + tokenizer.decode(response[0]))

# ─── Save Model ───────────────────────────────────────────────────────────────

model.save_pretrained_merged(configs.FINAL_MODEL_DIR, tokenizer, save_method="merged_16bit")
