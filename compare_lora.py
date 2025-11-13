# demo_compare.py
import torch, datasets
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from lora import LoRAInjector

def load_batch(tokenizer, seq=128, bs=8):
    text = "Hello world! " * seq
    inp = tokenizer(text, return_tensors="pt").input_ids[:,:seq]
    batch = {"input_ids": inp.repeat(bs,1), "attention_mask": torch.ones_like(inp).repeat(bs,1)}
    return batch

def compare():
    device="cuda"
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model_base = GPT2LMHeadModel.from_pretrained("gpt2")
    batch = load_batch(tok)

    inj_none = LoRAInjector(model_base)
    base = inj_none.profile(batch)
    print("Fine-tune baseline:", base)

    model_lora = GPT2LMHeadModel.from_pretrained("gpt2")
    inj = LoRAInjector(model_lora, r=4, alpha=16, dropout=0.05)
    inj.add_lora()
    lora = inj.profile(batch)
    print("LoRA r=4          :", lora)

    speed = round(lora['fw_ms']/base['fw_ms'],3), round(lora['bw_ms']/base['bw_ms'],3)
    mem   = round(lora['peak_mem_MB']/base['peak_mem_MB'],3)
    print(f"\nОтносительно FT: FW×{speed[0]}, BW×{speed[1]},  VRAM×{mem}")
