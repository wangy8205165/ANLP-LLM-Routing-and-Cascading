'''
This script is to run SLM (llama 3.1B-Instruct) and LLM (GPT-5) on the datasets
The results will be used as the baseline to compare  with AutoMix (Threshold & POMDP)
'''

# ===========================
# Import required libraries
# ===========================

import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from tqdm import tqdm
from huggingface_hub import notebook_login
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
import os
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
parser = argparse.ArgumentParser(description="Please enter the the name of dataset to test")
parser.add_argument("--dataset", type=str, choices=["cnli", "coqa", "narrative_qa", "qasper", "quality"], required=True,help="Enter Dataset")
args = parser.parse_args()
print(f"We will be testing on dataset {args.dataset}\n")
dataset = f"dataset/{args.dataset}.jsonl"

# ===========================
# Check the inputs are correct
# ===========================
inputs = pd.read_json(dataset, lines=True, orient="records")

inputs = inputs.sample(20)

length = len(inputs)
if args.dataset == "narrative_qa":
    assert length == 15772, "Incorrect Dataset!"
elif args.dataset == "cnli":
    assert length == 8228, "Incorrect Dataset!"
elif args.dataset == "coqa":
    assert length == 7849, "Incorrect Dataset!"
elif args.dataset == "quality":
    assert length == 4600, "Incorrect Dataset!"
elif args.dataset =="qasper":
    assert length == 4271, "Incorrect Dataset!"

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
    return mapping.get(key, key)

# ===========================
# Load the local llama 3 model
# ===========================

# Build the pipeline
def build_llm_pipeline(model_id=MODEL_ID, device=DEVICE):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    return pipeline
llama3_pipeline = build_llm_pipeline(MODEL_ID, DEVICE)

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
    return full_prompt

# ===========================
# Llama-3 inference
# ===========================

def generate_with_llama3(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    outputs = llama3_pipeline(
        messages,
        max_new_tokens=max_new_tokens,
        # temperature=temperature,
    )
    text = outputs[0]["generated_text"][-1]["content"]
    return text.strip()

# ===========================
# ChatGPT inference
# ===========================

from openai import OpenAI
MAX_NEW_TOKENS = 300
api_key = "your openai api token"
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def generate_with_gpt(prompt: str, model_name: str = "gpt-5",max_tokens: int = MAX_NEW_TOKENS) -> str:
    response = client.responses.create(
        model=model_name,
        reasoning={"effort": "low"},
        instructions="You are a helpful assistant.",
        input=prompt,
        # max_output_tokens = max_tokens
    )
    return response.output_text


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
# Run SLM and LLM baseline
# ===========================
outputs = {}

print("Running Llama-3 (small model)...")
outputs["llama3_pred"] = run_solver_job(inputs,partial(generate_with_llama3),max_workers=2)

print("Running ChatGPT (large model)...")
inputs["gpt_pred"] = run_solver_job(inputs,partial(generate_with_gpt, model_name="gpt-5"),max_workers=8)

# ===========================
# Save the results
# ===========================

import json
output_path = f"outputs/baseline_output_{args.dataset}.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(outputs, f, ensure_ascii=False, indent=4)