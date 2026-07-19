#!/usr/bin/env python3
"""Minimal, dependency-light LoRA SFT: fine-tune a base model on the SOMA
distillation corpus (verified prompt->gold pairs). Pure transformers + peft +
torch (no trl), so it is robust to the container's bleeding-edge transformers.

Loss is computed only on the assistant completion (the prompt is masked to -100),
standard instruction SFT. Saves the LoRA adapter to --out.
"""
import argparse
import json

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model


class SFTData(Dataset):
    def __init__(self, path, tok, max_len=1024):
        self.rows = [json.loads(x) for x in open(path)]
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        m = self.rows[i]["messages"]
        user, asst = m[0], m[1]
        # full render (user + assistant), and prompt-only render to find the mask boundary
        def render(msgs, gen):
            try:
                enc = self.tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=gen,
                                                   return_dict=True, enable_thinking=False)
            except TypeError:
                enc = self.tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=gen,
                                                   return_dict=True)
            ids = enc["input_ids"]
            return list(ids[0]) if ids and isinstance(ids[0], (list, tuple)) else list(ids)
        full = render([user, asst], False)
        prompt = render([user], True)
        full = full[: self.max_len]
        labels = list(full)
        for j in range(min(len(prompt), len(full))):
            labels[j] = -100                                   # mask the prompt: loss only on the answer
        return {"input_ids": full, "labels": labels}


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    ids, labs, mask = [], [], []
    for b in batch:
        p = maxlen - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * p)
        labs.append(b["labels"] + [-100] * p)
        mask.append([1] * len(b["input_ids"]) + [0] * p)
    return (torch.tensor(ids), torch.tensor(labs), torch.tensor(mask))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    lora = LoraConfig(r=args.r, lora_alpha=args.r * 2, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = SFTData(args.train, tok)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    steps = len(dl) * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * steps), steps)

    getattr(model, "train")()
    step = 0
    for ep in range(args.epochs):
        run = 0.0
        for ids, labs, mask in dl:
            ids, labs, mask = ids.to(model.device), labs.to(model.device), mask.to(model.device)
            out = model(input_ids=ids, attention_mask=mask, labels=labs)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            run += out.loss.item()
            step += 1
            if step % 20 == 0:
                print(f"  epoch {ep + 1} step {step}/{steps} loss {run / 20:.4f}", flush=True)
                run = 0.0
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("saved adapter ->", args.out, flush=True)


if __name__ == "__main__":
    main()
