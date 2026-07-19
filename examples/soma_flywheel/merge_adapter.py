#!/usr/bin/env python3
"""Merge a LoRA adapter into base weights → a standalone model dir the serving
stack (sglang/vllm) can load directly. Used by the orchestrator after the gain
gate accepts the finetune."""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cpu")
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    merged.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.out)
    print("merged ->", args.out)


if __name__ == "__main__":
    main()
