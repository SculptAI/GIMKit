from unsloth import FastModel  # noqa: I001
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from transformers import TextStreamer

from datasets import concatenate_datasets, load_dataset

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
    random_state=3407,
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
    dataset.shuffle(seed=0)
    .select(range(1000))
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
    train_dataset=dataset.select(range(800)),
    eval_dataset=dataset.select(range(800, 1000)),
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # Use GA to mimic batch size!
        warmup_steps=5,
        num_train_epochs=1,  # Set this for 1 full training run.
        max_steps=-1,
        learning_rate=2e-4,  # Reduce to 2e-5 for long training runs
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
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

messages = [{"role": "user", "content": "<|M_INPUT|>Nothing<|/M_INPUT|>"}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,  # Must add for generation
)

response = model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    max_new_tokens=1000,  # Increase for longer outputs!
    temperature=0.7,
    top_p=0.8,
    top_k=20,  # For non thinking
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)
print("\n\n" + tokenizer.decode(response[0]))

# ─── Save Model ───────────────────────────────────────────────────────────────

model.save_pretrained_merged("MaskedLLM", tokenizer, save_method="merged_16bit")
