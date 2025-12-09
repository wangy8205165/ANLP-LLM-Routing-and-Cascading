# LLM Routing and Cascading

This project implements an intelligent routing system for LLM cascading, where a Small Language Model (SLM) and Large Language Model (LLM) are strategically used based on confidence scores to optimize performance-cost trade-offs. The system uses self-verification confidence to decide whether to keep the SLM's answer or route to the more expensive LLM.

## Overview

### Problem Statement

In production NLP systems, there's a fundamental trade-off between model performance and computational cost. Large Language Models (LLMs) like GPT-5 deliver superior accuracy but are expensive to run, while Small Language Models (SLMs) like Llama-3.1 are cost-effective but may struggle with complex queries. The challenge is: **how can we automatically decide when to use the expensive LLM versus the cheaper SLM to maximize performance while minimizing cost?**

### Solution: AutoMix-Style Intelligent Routing

This project implements an **AutoMix-style routing system** that intelligently routes queries between a Small Language Model (SLM) and Large Language Model (LLM) based on the SLM's self-assessed confidence. The key innovation is using **self-verification**—having the SLM evaluate its own answer quality—to make routing decisions without requiring external supervision.

### How It Works

The system operates in three main stages:

1. **Confidence Estimation**: For each query, the SLM (Llama-3.1 8B-Instruct) generates an answer and then performs self-verification to estimate its confidence in that answer. The confidence is discretized into bins: `{0.0, 0.25, 0.5, 0.75, 1.0}`.

2. **Policy Learning**: A discrete routing policy is learned from training data by:
   - Estimating average performance (accuracy/F1) of both SLM and LLM (GPT-5) for each confidence bin
   - Computing reward functions: `R_keep(v) = P_slm(v) - λ·C_slm` and `R_route(v) = P_llm(v) - λ·(C_slm + C_llm)`
   - Selecting the action (`keep` or `route`) that maximizes reward for each confidence bin
   - The parameter λ controls the performance-cost trade-off

3. **Routing Execution**: At inference time, the system:
   - Gets SLM's answer and confidence score
   - Looks up the policy action for the nearest confidence bin
   - If `keep`: returns SLM's answer (cost = 5 units)
   - If `route`: queries LLM and returns its answer (cost = 25 units)

### Key Features

- **Self-Verification Based**: Uses the SLM's own confidence assessment, eliminating the need for external confidence models or labeled training data
- **Discrete Policy Learning**: Learns optimal routing decisions per confidence bin using a small training set (typically 50 samples)
- **Cost-Aware Optimization**: Explicitly optimizes the performance-cost trade-off through the λ parameter
- **Multi-Dataset Support**: Works across diverse NLP tasks (classification, QA, reading comprehension) with appropriate metrics
- **Comprehensive Evaluation**: Measures both performance (accuracy/F1) and cost-effectiveness (ΔIBC metric)

### Models and Datasets

- **Small Language Model**: Llama-3.1 8B-Instruct (local deployment)
- **Large Language Model**: GPT-5 (via OpenAI API)
- **Datasets**: CNLI, CoQA, NarrativeQA, QASPER, QuALITY

### Benefits

- **Cost Reduction**: Achieves near-LLM performance at a fraction of the cost by only using the expensive LLM when necessary
- **Automatic Adaptation**: Policy automatically adapts to dataset characteristics and cost constraints
- **Interpretable**: Discrete policies are easy to understand and debug
- **Scalable**: Can be trained on small datasets and applied to large-scale inference

## Requirements

- Python 3.8+
- Required packages:
  - `pandas`
  - `numpy`
  - `transformers` (for Llama-3.1)
  - `torch`
  - `openai` (for GPT-5 API)
  - `tqdm`
  - `huggingface_hub`

## Installation

```bash
# Clone the repository
cd ANLP-LLM-Routing-and-Cascading

# Install dependencies
pip install pandas numpy transformers torch openai tqdm huggingface_hub

# Set up OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

## Project Structure

```
.
├── dataset/                    # Input datasets (JSONL format)
│   ├── cnli_short.jsonl
│   ├── coqa_short.jsonl
│   ├── narrative_qa_short.jsonl
│   ├── qasper_short.jsonl
│   └── quality_short.jsonl
├── outputs/                    # Baseline predictions from SLM and LLM
│   ├── baseline_output_cnli_short.json
│   ├── baseline_output_coqa_short.json
│   └── ...
├── verification/               # Self-verification confidence scores
│   ├── self_ver_cnli_short.json
│   ├── self_ver_coqa_short.json
│   └── ...
├── router_data/                # Combined data for router training
│   ├── router_data_cnli_short.jsonl
│   └── ...
├── policy/                     # Learned routing policies
│   ├── policy_cnli_short.jsonl
│   └── ...
├── pomdp/                      # Routing results after applying policies
│   ├── results_cnli_short.json
│   └── ...
├── threshold_router_outputs/   # Threshold-based router results
└── [Python scripts and notebooks]
```

## Workflow

The complete pipeline consists of the following steps:

1. **Generate Baseline Predictions** → 2. **Self-Verification** → 3. **Prepare Router Data** → 4. **Train Router** → 5. **Apply Router** → 6. **Evaluate**

### Step 1: Generate Baseline Predictions

Run both SLM (Llama-3.1) and LLM (GPT-5) on the dataset to get baseline predictions.

```bash
python run_solver.py --dataset cnli
```

**Output:** `outputs/baseline_output_{dataset}.json`

**File:** `run_solver.py`
- Loads Llama-3.1 8B-Instruct model locally
- Calls GPT-5 via OpenAI API
- Generates predictions for all samples in parallel
- Saves predictions in JSON format with keys `llama3_pred` and `gpt_pred`

### Step 2: Self-Verification (External)

Generate self-verification confidence scores for SLM predictions. This step is typically done in a notebook (`AutoMix_Self_Verification.ipynb`).

**Output:** `verification/self_ver_{dataset}.json`

The verification file contains:
```json
{
  "slm_ver_confidence": [0.0, 0.25, 0.5, 0.75, 1.0, ...]
}
```

Confidence values are discretized to bins: `{0.0, 0.25, 0.5, 0.75, 1.0}`

### Step 3: Prepare Router Data

Combine dataset, baseline predictions, and verification scores into a unified format for router training.

```bash
python prepare_router_data.py --dataset cnli_short
```

**Output:** `router_data/router_data_{dataset}.jsonl`

**File:** `prepare_router_data.py`
- Merges dataset, baseline outputs, and verification results
- Computes per-sample performance metrics (`perf_slm`, `perf_llm`)
- Creates router training data with all necessary fields

Each record contains:
- `id`, `dataset`, `gold_output`
- `slm_pred`, `llm_pred`
- `slm_confidence` (from self-verification)
- `perf_slm`, `perf_llm` (accuracy or F1 score)

### Step 4: Train Router

Train a discrete routing policy based on confidence bins.

```bash
python train_router_discrete.py --dataset cnli_short --lambda 0.01 --c_slm 1 --c_llm 60 --train_size 50
```

**Arguments:**
- `--dataset`: Dataset name (e.g., `cnli_short`)
- `--lambda`: Trade-off coefficient λ for reward = perf - λ * cost (default: 0.01)
- `--c_slm`: Cost of SLM (default: 1.0)
- `--c_llm`: Cost of LLM, incremental over SLM (default: 60.0)
- `--train_size`: Number of samples for training (default: 50)
- `--seed`: Random seed for train/test split (default: 42)

**Output:** `policy/policy_{dataset}.jsonl`

**File:** `train_router_discrete.py`
- Splits router data into train/test sets
- Estimates average performance per confidence bin
- Builds discrete policy: for each bin, chooses `keep` (use SLM) or `route` (use LLM)
- Policy decision: `keep` if `R_keep(v) >= R_route(v)`, else `route`
  - `R_keep(v) = P_slm(v) - λ * C_slm`
  - `R_route(v) = P_llm(v) - λ * (C_slm + C_llm)`

Policy format:
```json
{
  "0.0": "route",
  "0.25": "keep",
  "0.5": "keep",
  "0.75": "keep",
  "1.0": "route"
}
```

### Step 5: Apply Router

Apply the learned policy to route samples and generate final answers.

```bash
python apply_router.py --dataset cnli_short
```

**Output:** `pomdp/results_{dataset}.json`

**File:** `apply_router.py`
- Loads policy and router data
- For each sample, finds nearest confidence bin and applies policy action
- If `keep`: uses SLM prediction (cost = 5)
- If `route`: uses LLM prediction (cost = 25)
- Saves results with routing decisions and final answers

### Step 6: Evaluate

Evaluate routing performance and compute cost-benefit metrics.

```bash
python eval.py --dataset cnli_short --cost 18.32
```

**Arguments:**
- `--dataset`: Dataset name
- `--cost`: Average cost per sample (required for F1 datasets)

**File:** `eval.py`
- Computes accuracy (for CNLI) or F1 score (for CoQA, QASPER, NarrativeQA)
- Calculates ΔIBC (delta Improvement-Benefit-Cost) metric
- Compares routing performance against baseline SLM and LLM

**Metrics:**
- **Accuracy** (CNLI): Exact match after label normalization
- **F1 Score** (CoQA, QASPER, NarrativeQA): Token-level F1 with normalization
- **ΔIBC**: Percentage improvement in benefit-cost ratio over baseline

## File Documentation

### Core Scripts

#### `run_solver.py`
Generates baseline predictions from both SLM and LLM models.
- **Purpose:** Create baseline outputs for comparison
- **Input:** Dataset JSONL file
- **Output:** JSON file with `llama3_pred` and `gpt_pred` arrays
- **Dependencies:** `prompt_template.py` for prompt formatting

#### `prepare_router_data.py`
Prepares unified router training data from multiple sources.
- **Purpose:** Combine dataset, predictions, and verification into training format
- **Input:** Dataset, baseline outputs, verification results
- **Output:** Router data JSONL with performance metrics
- **Dependencies:** `metric_compute.py` for performance calculation

#### `train_router_discrete.py`
Trains a discrete routing policy based on confidence bins.
- **Purpose:** Learn optimal routing decisions per confidence bin
- **Input:** Router data JSONL
- **Output:** Policy JSON mapping confidence bins to actions
- **Key Parameters:** λ (trade-off), costs (C_slm, C_llm), train_size

#### `apply_router.py`
Applies learned policy to route samples.
- **Purpose:** Execute routing decisions and generate final answers
- **Input:** Policy JSON, router data JSONL
- **Output:** Results JSON with routing decisions and final answers
- **Metrics:** Computes average cost per sample

#### `eval.py`
Evaluates routing performance and computes metrics.
- **Purpose:** Measure routing effectiveness and cost-benefit
- **Input:** POMDP results JSON
- **Output:** Accuracy/F1 scores and ΔIBC metric
- **Dependencies:** `metric_compute.py` for metric computation

### Utility Files

#### `metric_compute.py`
Metric computation utilities for different datasets.
- **Functions:**
  - `normalize_cnli_label()`: Normalizes CNLI labels (Entailment/Contradiction/Not mentioned)
  - `normalize_quality_label()`: Extracts A/B/C/D from QuALITY answers
  - `compute_accuracy()`: Computes accuracy with optional normalization
  - `f1_score()`: Computes token-level F1 score with text normalization
  - `compute_average_f1()`: Computes mean and variance of F1 scores

#### `prompt_template.py`
Dataset-specific prompt templates and instructions.
- **Purpose:** Define prompt formats for each dataset
- **Datasets:** CNLI, CoQA, NarrativeQA, QASPER, QuALITY
- **Format:** Each dataset has `instruction`, `prompt` template, and `truncation_message`

#### `util.py`
Utility script for dataset preprocessing (example: slicing QuALITY to 1000 samples).

#### `openai_api.py`
Example script demonstrating OpenAI API usage (not used in main pipeline).

### Notebooks

- `AutoMix_Run_Solver.ipynb`: Interactive notebook for running solver
- `AutoMix_Self_Verification.ipynb`: Self-verification confidence generation
- `AutoMix_Threshold_Router.ipynb`: Threshold-based router experiments
- `test.ipynb`: Testing and experimentation

## Supported Datasets

1. **CNLI** (`cnli_short`): Chinese Natural Language Inference
   - Metric: Accuracy
   - Labels: Entailment, Contradiction, Not mentioned

2. **CoQA** (`coqa_short`): Conversational Question Answering
   - Metric: F1 Score
   - Format: Free-form answers

3. **NarrativeQA** (`narrative_qa_short`): Reading comprehension on stories
   - Metric: F1 Score
   - Format: Free-form answers

4. **QASPER** (`qasper_short`): Question Answering on Scientific Papers
   - Metric: F1 Score
   - Format: Free-form answers (or "unanswerable")

5. **QuALITY** (`quality_short`): Multiple-choice reading comprehension
   - Metric: Accuracy
   - Format: A/B/C/D choices

## Output Formats

### Baseline Output (`outputs/baseline_output_{dataset}.json`)
```json
{
  "llama3_pred": ["answer1", "answer2", ...],
  "gpt_pred": ["answer1", "answer2", ...]
}
```

### Router Data (`router_data/router_data_{dataset}.jsonl`)
Each line is a JSON object:
```json
{
  "id": "sample_id",
  "dataset": "cnli_short",
  "gold_output": "Entailment",
  "slm_pred": "Entailment",
  "llm_pred": "Entailment",
  "slm_confidence": 0.75,
  "perf_slm": 1.0,
  "perf_llm": 1.0
}
```

### Policy (`policy/policy_{dataset}.jsonl`)
```json
{
  "0.0": "route",
  "0.25": "keep",
  "0.5": "keep",
  "0.75": "keep",
  "1.0": "route"
}
```

### Results (`pomdp/results_{dataset}.json`)
Array of result objects:
```json
[
  {
    "id": "sample_id",
    "dataset": "cnli_short",
    "slm_confidence": 0.75,
    "action": "keep",
    "gold_answer": "Entailment",
    "final_answer": "Entailment"
  },
  ...
]
```

## Example Usage

Complete workflow example for CNLI dataset:

```bash
# 1. Generate baseline predictions
python run_solver.py --dataset cnli

# 2. (Self-verification done separately, e.g., in notebook)
# Output: verification/self_ver_cnli_short.json

# 3. Prepare router data
python prepare_router_data.py --dataset cnli_short

# 4. Train router
python train_router_discrete.py --dataset cnli_short --lambda 0.01 --c_slm 1 --c_llm 60 --train_size 50

# 5. Apply router
python apply_router.py --dataset cnli_short

# 6. Evaluate
python eval.py --dataset cnli_short --cost 18.32
```

## Cost Model

- **SLM (Llama-3.1)**: Cost = 5 units per sample
- **LLM (GPT-5)**: Cost = 25 units per sample (incremental)
- **Average Cost**: Computed as `(n_keep * 5 + n_route * 25) / total_samples`

## Evaluation Metrics

### Accuracy (CNLI, QuALITY)
Exact match after label normalization.

### F1 Score (CoQA, QASPER, NarrativeQA)
Token-level F1 computed as:
- Normalize text (lowercase, remove punctuation)
- Compute precision and recall on token overlap
- F1 = 2 * (precision * recall) / (precision + recall)

### ΔIBC (Delta Improvement-Benefit-Cost)
Measures improvement in benefit-cost ratio:
```
IBC_model = (P_model - P_slm) / (C_model - C_slm)
IBC_baseline = (P_llm - P_slm) / (C_llm - C_slm)
ΔIBC = ((IBC_model - IBC_baseline) / IBC_baseline) * 100%
```

Where:
- `P_model`: Performance of routing system
- `P_slm`, `P_llm`: Baseline SLM and LLM performance
- `C_model`: Average cost of routing system
- `C_slm`, `C_llm`: Costs of SLM and LLM

## Notes

- Self-verification confidence is discretized to bins: `{0.0, 0.25, 0.5, 0.75, 1.0}`
- The routing policy is deterministic: each confidence bin maps to a single action
- Training uses a small subset (default 50 samples) to estimate bin statistics
- The system assumes SLM confidence correlates with answer quality

## License

See LICENSE file for details.

## Project Information

This is a project for 11711 ANLP (Advanced Natural Language Processing) at Carnegie Mellon University.
