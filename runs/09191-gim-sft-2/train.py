from unsloth import FastModel  # noqa: I001

import logging
import os

import configs

from datasets import concatenate_datasets, load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import get_chat_template, train_on_responses_only


# ─── General Setup ────────────────────────────────────────────────────────────

os.environ["SWANLAB_EXP_NAME"] = configs.EXP_NAME
os.environ["SWANLAB_DESCRIPTION"] = configs.EXP_DESC

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
    model_name="unsloth/Qwen3-4B-Instruct-2507",
    max_seq_length=4096,
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

dataset = load_dataset("Ki-Seki/MaskedIO")
dataset = concatenate_datasets(list(dataset.values()))
dataset = (
    dataset.shuffle(seed=configs.RANDOM_SEED)
    .select(range(configs.DATASET_LEN))
    .map(
        lambda example: {
            "text": tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": example["m_input"]},
                    {"role": "assistant", "content": example["m_output"]},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
        }
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
        warmup_steps=configs.WARMUP_STEPS,
        eval_strategy="steps",
        eval_steps=configs.EVAL_STEPS,
        num_train_epochs=1,  # Set this for 1 full training run.
        max_steps=-1,
        learning_rate=2e-4,  # Reduce to 2e-5 for long training runs
        logging_steps=1,
        save_steps=configs.SAVE_STEPS,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
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

messages = [
    {
        "role": "user",
        "content": '<|M_INPUT|>This is an <|MASKED id="m_1"|><|/MASKED|> text.<|/M_INPUT|>',
    }
]
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
