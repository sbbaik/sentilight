from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_messages(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], str]:
    if len(messages) < 3 or messages[-1].get("role") != "assistant":
        raise ValueError("expected messages ending with assistant")
    return messages[:-1], str(messages[-1]["content"])


def apply_chat_template(tokenizer, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class ChatSFTDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_seq_len: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        row = self.rows[index]
        prompt_messages, assistant_text = split_messages(row["messages"])
        prompt = apply_chat_template(self.tokenizer, prompt_messages)
        target = assistant_text + (self.tokenizer.eos_token or "<|im_end|>")
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        target_ids = self.tokenizer(target, add_special_tokens=False).input_ids
        input_ids = (prompt_ids + target_ids)[: self.max_seq_len]
        labels = ([-100] * len(prompt_ids) + target_ids)[: self.max_seq_len]
        if len(input_ids) < 2:
            input_ids = input_ids + [self.tokenizer.eos_token_id]
            labels = labels + [self.tokenizer.eos_token_id]
        return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def make_collator(tokenizer):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def collate(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids, labels, attention = [], [], []
        for item in batch:
            pad = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * pad)
            labels.append(item["labels"] + [-100] * pad)
            attention.append(item["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }

    return collate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tune local Qwen3-0.6B for SentiLight JSON lighting output")
    parser.add_argument("--base-model", type=Path, default=Path("models/qwen"))
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=512)
    parser.add_argument("--eval-strategy", choices=["no", "epoch"], default="epoch")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    train_rows = load_jsonl(args.train_jsonl)
    if args.max_train_rows and args.max_train_rows < len(train_rows):
        random.shuffle(train_rows)
        train_rows = train_rows[: args.max_train_rows]
    val_rows = load_jsonl(args.val_jsonl)
    if args.max_eval_rows and args.max_eval_rows < len(val_rows):
        val_rows = val_rows[: args.max_eval_rows]

    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model), local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(args.base_model),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = ChatSFTDataset(train_rows, tokenizer, args.max_seq_len)
    val_dataset = ChatSFTDataset(val_rows, tokenizer, args.max_seq_len)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=20,
        save_strategy="epoch",
        eval_strategy=args.eval_strategy,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if args.eval_strategy != "no" else None,
        data_collator=make_collator(tokenizer),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    summary = {"base_model": str(args.base_model), "train_rows": len(train_rows), "val_rows": len(val_rows), "output_dir": str(args.output_dir)}
    (args.output_dir / "sentilight_training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
