# ===========================
# Import required libraries
# ===========================
import transformers
import torch.nn.functional as F
import torch
from huggingface_hub import notebook_login
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from prompt_template import dataset_prompts_and_instructions

# ===========================
# Configurations
# ===========================

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_INPUT_TOKENS = 3500
MAX_NEW_TOKENS = 300

# ===========================
# Determine which dataset to test on
# ===========================
data = "quality_short"
dataset = f"dataset/{data}.jsonl"
print(f"We will be testing on dataset {dataset}\n")

# ===========================
# Check the inputs are correct
# ===========================
inputs = pd.read_json(dataset, lines=True, orient="records")

length = len(inputs)
assert length == 1000

# ===========================
# Dataset name mapping
# ===========================

def normalize_dataset_name(name: str) -> str:
    name_lower = name.lower()
    mapping = {
        "cnli": "cnli",
        "coqa": "coqa",
        "narrativeqa": "narrative_qa",
        "narrative_qa": "narrative_qa",
        "qasper": "qasper",
        "quality": "quality",
    }
    key = ''.join(ch for ch in name_lower if ch.isalpha() or ch == '_')
# ===========================
# Construct prompt
# ===========================

def build_prompt(row) -> str:
    ds_key = normalize_dataset_name(row["dataset"])
    cfg = dataset_prompts_and_instructions[ds_key]

    full_prompt = cfg["prompt"].format(
              context=row["base_ctx"],
        instruction=cfg["instruction"],
        question=row["question"],
    )

# ===========================
# Build the model
# ===========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# ===================================================
# Generate output and compute logits, entropies
# ===================================================
def generate_with_llama3(prompt: str) -> str:
  messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    # TODO: Generate output and compute logits, entropies

# ===========================
# Concurrent running
# ===========================

def run_solver_job(df, engine_func, max_workers: int = 4):
    prompts = [build_prompt(row) for _, row in df.iterrows()]
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for res in tqdm(executor.map(engine_func, prompts), total=len(prompts)):
            results.append(res)
    return results

# ===========================
# Run SLM
# ===========================
outputs = {}

print("Running Llama-3 (small model)...")
outputs["llama3_pred"] = run_solver_job(inputs,partial(generate_with_llama3),max_workers=2)