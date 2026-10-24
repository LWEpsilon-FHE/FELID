import argparse
import json
import time
from itertools import chain
from pathlib import Path

import torch
from datasets import concatenate_datasets, load_dataset, load_from_disk
from safetensors.torch import load_file
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertConfig, BertForMaskedLM

from attention import KernelBertForMaskedLM, freeze_parameters


def load_corpus(path, tokenizer, config):
    path = Path(path)
    if path.is_dir():
        return load_from_disk(str(path))
    if path.is_file():
        corpus = load_dataset("text", data_files=str(path), split="train")
    else:
        sources = []
        for name, variant, group in config["sources"]:
            source = load_dataset(name, variant, split="train", trust_remote_code=True)
            source = source.select(range(min(len(source), config["max_entries"])))
            source = source.select_columns("text")
            if group > 1:
                source = source.map(
                    lambda batch: {"text": ["".join(batch["text"][i:i + group]) for i in range(0, len(batch["text"]) - group + 1, group)]},
                    batched=True, batch_size=1024,
                )
            sources.append(source)
        corpus = concatenate_datasets(sources).shuffle()
        corpus = corpus.select(range(min(len(corpus), config["max_entries"])))
    def tokenize(batch):
        return tokenizer(batch["text"], return_attention_mask=False, return_token_type_ids=False, truncation=False)
    tokens = corpus.map(tokenize, batched=True, batch_size=1024, remove_columns=corpus.column_names)
    def pack(batch):
        tokens = list(chain.from_iterable(batch["input_ids"]))
        length = config["sequence_length"]
        return {"input_ids": [tokens[i:i + length] for i in range(0, len(tokens) - length + 1, length)]}
    packed = tokens.map(pack, batched=True, batch_size=1024).shuffle()
    packed = packed.select(range(min(len(packed), config["max_sequences"])))
    if not path.is_file():
        packed.save_to_disk(str(path))
    return packed


class MaskTokens:
    def __init__(self, tokenizer, probability):
        self.tokenizer = tokenizer
        self.probability = probability

    def __call__(self, examples):
        inputs = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
        labels = inputs.clone()
        count = round(self.probability * inputs.shape[1])
        locations = torch.argsort(torch.randint_like(inputs, inputs.shape[1]))[:, :count]
        selected = torch.zeros_like(inputs, dtype=torch.bool).scatter_(1, locations, True)
        labels[~selected] = -100
        masked = torch.zeros_like(inputs, dtype=torch.bool).scatter_(1, locations[:, :round(0.8 * count)], True)
        inputs[masked] = self.tokenizer.mask_token_id
        replaced = torch.zeros_like(inputs, dtype=torch.bool).scatter_(1, locations[:, round(0.8 * count):round(0.9 * count)], True)
        words = torch.randint(len(self.tokenizer), inputs.shape)
        inputs[replaced] = words[replaced]
        return {"input_ids": inputs, "labels": labels}


def hidden_mse(student, teacher, input_ids, pad_token_id):
    mask = (input_ids != pad_token_id).unsqueeze(-1).to(student.dtype)
    denominator = (mask.sum() * student.shape[-1]).clamp_min(1)
    return ((student - teacher).pow(2) * mask).sum() / denominator


def logits_kl(student, teacher, input_ids, pad_token_id, temperature):
    mask = (input_ids != pad_token_id).to(student.dtype)
    divergence = F.kl_div(
        F.log_softmax(student / temperature, dim=-1),
        F.softmax(teacher / temperature, dim=-1), reduction="none",
    ).sum(-1)
    return (divergence * mask).sum() / mask.sum().clamp_min(1) * temperature ** 2


def load_student(checkpoint):
    source = Path(checkpoint)
    config_file = source / "model_config.json"
    if config_file.is_file():
        config = BertConfig.from_json_file(str(config_file))
        config._attn_implementation = "eager"
        model = KernelBertForMaskedLM(config)
        model.load_state_dict(load_file(str(source / "model.safetensors")), strict=True)
    else:
        model = KernelBertForMaskedLM.from_pretrained(checkpoint, attn_implementation="eager")
    freeze_parameters(model)
    return model


def make_optimizer(model, config):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if any(key in name for key in ("bias", "LayerNorm.weight", "norm")) else decay).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": config["weight_decay"]}, {"params": no_decay, "weight_decay": 0}],
        lr=config["learning_rate"], betas=tuple(config["betas"]), eps=config["epsilon"],
    )


def save_model(model, tokenizer, path):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def train(config, data_path, output_path, checkpoint, teacher_checkpoint, stages, device):
    microbatch = config["microbatch_size"]
    if microbatch <= 0 or config["batch_size"] <= 0 or config["batch_size"] % microbatch:
        raise ValueError("batch_size must be a positive multiple of microbatch_size")
    accumulation = config["batch_size"] // microbatch
    if any(steps <= 0 or steps % accumulation for steps in stages):
        raise ValueError("Each stage must contain a positive multiple of the accumulation interval")
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
    tokenizer.model_max_length = 10 ** 30
    corpus = load_corpus(data_path, tokenizer, config)
    tokenizer.model_max_length = config["sequence_length"]
    if len(corpus) < microbatch:
        raise ValueError("The corpus must contain at least one microbatch")
    model = load_student(checkpoint).to(device).eval()
    teacher = BertForMaskedLM.from_pretrained(teacher_checkpoint, attn_implementation="eager").to(device).eval()
    teacher.requires_grad_(False)
    if model.config.vocab_size != len(tokenizer) or teacher.config.vocab_size != len(tokenizer):
        raise ValueError("Student, teacher and tokenizer must share a vocabulary")
    if model.config.hidden_size != teacher.config.hidden_size:
        raise ValueError("Student and teacher hidden sizes must match")
    loader = DataLoader(
        corpus, batch_size=microbatch, shuffle=True, drop_last=True,
        collate_fn=MaskTokens(tokenizer, config["mlm_probability"]),
        num_workers=config["workers"], pin_memory=device.type == "cuda",
        persistent_workers=config["workers"] > 0,
    )
    amp = device.type == "cuda" and config["precision"] != "float32"
    dtype = getattr(torch, config["precision"])
    output = Path(output_path)
    start = time.monotonic()
    completed = 0
    for steps in stages:
        optimizer = make_optimizer(model, config)
        scaler = torch.amp.GradScaler("cuda", enabled=amp and dtype == torch.float16)
        optimizer.zero_grad(set_to_none=True)
        batches = iter(loader)
        for step in range(1, steps + 1):
            try:
                batch = next(batches)
            except StopIteration:
                batches = iter(loader)
                batch = next(batches)
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            factor = min((step - 1) / max(config["warmup_steps"], 1), 1) if config["warmup_steps"] else 1
            for group in optimizer.param_groups:
                group["lr"] = config["learning_rate"] * factor
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                student = model(**batch, output_hidden_states=True, return_dict=True)
                with torch.no_grad():
                    target = teacher(input_ids=batch["input_ids"], output_hidden_states=True, return_dict=True)
                mse = hidden_mse(student.hidden_states[-1], target.hidden_states[-1].to(student.hidden_states[-1]), batch["input_ids"], tokenizer.pad_token_id)
                kl = logits_kl(student.logits, target.logits.to(student.logits), batch["input_ids"], tokenizer.pad_token_id, config["temperature"])
                loss = config["mlm_weight"] * student.loss + config["hidden_mse_weight"] * mse + config["logits_kl_weight"] * kl
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {completed + step}")
            scaler.scale(loss / accumulation).backward()
            if step % accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if step == 1 or step % config["log_every"] == 0 or step == steps:
                print(f"step={completed + step} loss={loss.item():.6f} mlm={student.loss.item():.6f} hidden_mse={mse.item():.6f} logits_kl={kl.item():.6f} seconds={time.monotonic() - start:.2f}", flush=True)
            if config["save_every"] and step % config["save_every"] == 0:
                save_model(model, tokenizer, output / "checkpoint")
        completed += steps
        save_model(model, tokenizer, output / f"step_{completed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--data")
    parser.add_argument("--output")
    parser.add_argument("--checkpoint")
    parser.add_argument("--teacher")
    parser.add_argument("--steps", type=int, nargs="+")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    train(
        config, args.data or config["data"], args.output or config["output"],
        args.checkpoint or config["checkpoint"], args.teacher or config["teacher"],
        args.steps or config["stages"], torch.device(args.device),
    )


if __name__ == "__main__":
    main()
