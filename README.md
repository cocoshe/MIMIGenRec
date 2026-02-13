
<p align="center">
  <img src="assets/MIMIGenRec_logo.png" alt="MIMIGenRec logo"/>
</p>

<p align="center"><strong>A Flexible Framework for Generative Recommendation</strong></p>



**MIMIGenRec** (Modular, Integrated, Mutable, Interchangeable GenRec) is a flexible training framework for **generative recommendation** models.

**🌟 Highlights**

- **LlamaFactory integration**: SFT and LoRA for a wide range of custom models via simple YAML configs; support for backends such as [Unsloth](https://github.com/unslothai/unsloth); built-in experiment monitors (e.g. WandB) for logging and comparison.
- **TRL integration**: Tight integration with [TRL](https://github.com/huggingface/trl) and the Hugging Face ecosystem; multi-GPU and multi-node training with [Accelerate](https://github.com/huggingface/accelerate), flexible [DeepSpeed](https://github.com/microsoft/DeepSpeed) configs (ZeRO-2/3, etc.), and easy to design custom rewards (e.g. NDCG, HR) for policy optimization.
- **Flexible Trie design**: Constrained decoding over SIDs via a Trie, which is flexible to build constrained logits processor for beam search when rollout.

## Overview

| Stage | Framework | Description |
|-------|-----------|-------------|
| **SFT** | LlamaFactory | `llamafactory-cli train` with YAML configs; supports 0.5B / 1.5B / 3B and multiple data sizes |
| **RL** | TRL  | Custom `MIMIGenRec` model wrapper + `GRPOTrainer`; ranking rewards (e.g. NDCG) for policy optimization |

## Quick Start

### 1. Environment & Dependencies

**Install the current library**:

```bash
pip install -e .
```

This installs LlamaFactory in editable mode with all dependencies (PyTorch, transformers, TRL, accelerate, etc. per `pyproject.toml`). The `llamafactory-cli` (and `lmf`) commands will be available after install.

- Optional: set `HF_ENDPOINT` (e.g. `https://hf-mirror.com`) if you use a mirror.
- Optional: set `WANDB_API_KEY` and `WANDB_PROJECT` for experiment logging.

>**Tested with packages**. If the pipeline fails due to version incompatibilities, please align your environment with the versions below:

| Package      | Version   |
|-------------|-----------|
| Python      | 3.12.12   |
| torch       | 2.8.0+cu128 |
| transformers| 4.57.1    |
| trl         | 0.24.0    |
| accelerate  | 1.11.0    |
| peft        | 0.17.1    |
| datasets    | 4.0.0     |

To print your current environment versions (run inside your env):

```bash
python -c "
import sys
for p in ['torch', 'transformers', 'trl', 'accelerate', 'peft', 'datasets']:
    try:
        m = __import__(p); print(p, getattr(m, '__version__', '?'))
    except Exception as e: print(p, 'not installed')
print('python', sys.version.split()[0])
"
```

### 2. Data Preparation

> If you want to test on prepared dataset, you can skip to `5. SFT training` section.

#### 2.1 Download Amazon data (example: `Industrial_and_Scientific`)

```bash
cd ./data
wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Industrial_and_Scientific_5.json.gz
gunzip Industrial_and_Scientific_5.json.gz
wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Industrial_and_Scientific.json.gz
gunzip meta_Industrial_and_Scientific.json.gz
```

#### 2.2 Filter and preprocess
```
bash amazon18_data_process.sh
```

Then we got:

- `Industrial_and_Scientific.item2id`
- `Industrial_and_Scientific.user2id`
- `Industrial_and_Scientific.review.json`
- `Industrial_and_Scientific.item.json`
- `Industrial_and_Scientific.inter.json`
- `Industrial_and_Scientific.test.inter`
- `Industrial_and_Scientific.valid.inter`
- `Industrial_and_Scientific.train.inter`

#### 2.3 Encode item text to embeddings

Please follow "Encode item text to embeddings" in [MiniOneRec](https://github.com/AkaliKong/MiniOneRec#2-data-preparation):

Then we got:

- `Industrial_and_Scientific.emb-qwen-td.npy`

### 3. SID Construction

Please follow 3.1, 3.2 of "SID Construction" in [MiniOneRec](https://github.com/AkaliKong/MiniOneRec#3-sid-construction)
to generate indices.

Then we got:

- `Industrial_and_Scientific.index.json`

### 4. Preprocess for SFT + RL

By default the category is `Industrial_and_Scientific` with raw data under `data/Amazon18`. Run:

```bash
bash preprocess_data_sft_rl.sh
```

This runs `preprocess_data_sft_rl.py` and writes SFT/RL data and `new_tokens.json` to `data/Industrial_and_Scientific/`. You can change `DATA_DIR`, `CATEGORY`, `OUTPUT_DIR`, `TASK4_SAMPLE`, and `SEED` in the script.

Then we got:

```
data/Industrial_and_Scientific/
├── new_tokens.json      # SID vocabulary for LlamaFactory add_tokens_list
├── id2sid.json          # item_id -> [sid1, sid2, sid3] (same format as source index)
├── sft/
│   ├── train.json
│   ├── valid.json
│   └── test.json
└── rl/
    ├── train.json
    ├── valid.json
    └── test.json
```


### 5. SFT training

```bash
bash sft.sh
```

- Default: 8 GPUs (`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`), 0.5B config.
- Edit `sft.sh` to change GPUs, WANDB project, and comment/uncomment the relevant `llamafactory-cli train` lines to switch 0.5B / 1.5B / 3B and dsz (e.g. dsz0 / dsz2 / dsz3).

### 6. RL training

After SFT and once you have a checkpoint, run RL with TRL:

```bash
bash trl_trainer.sh
```

- In `trl_trainer.sh` set:
  - `MODEL_PATH`: path to the SFT checkpoint (e.g. `saves/qwen2.5-0.5b/full/industry-sft-dsz0`)
  - `DATA_DIR`: RL data directory (e.g. `data/amazon_industry/rl`)
  - `INDEX_PATH`: category index file (e.g. `data/amazon_industry/Industrial_and_Scientific.index.json`)
  - `OUTPUT_DIR`: RL output directory
- The script launches `trl_trainer.py` via `accelerate` + DeepSpeed, using `MIMIGenRec` and `GRPOTrainer` with rewards from `rewards/ranking_reward` (e.g. NDCG rule reward).

### 7. Evaluate

Set your trained model in `evaluate.sh`.

`exp_name`: Your model path
`test_data_path`: test json path
`output_dir`: path to save results


```bash
bash evaluate.sh
```


---

## SFT/RL with Custom Dataset

### 1. Prepare with Custom Dataset

You must **first convert items to SIDs** (e.g. via [MiniOneRec](https://github.com/AkaliKong/MiniOneRec) SID construction), then prepare **`new_tokens.json`**, **`id2sid.json`** and the **SFT / RL datasets**. The layout and formats are described below.


**Example directory structure:**

```
data/Industrial_and_Scientific/
├── new_tokens.json      # SID vocabulary
├── id2sid.json          # item_id -> [sid1, sid2, sid3]
├── sft/
│   ├── train.json       # SFT training set
│   ├── valid.json       # SFT validation set
│   └── test.json        # SFT test set
└── rl/
    ├── train.json       # RL training set
    ├── valid.json       # RL validation set
    └── test.json        # RL test set
```

#### 1.1 `new_tokens.json`

- **Format**: A JSON array of strings. Each string is a **semantic ID (SID) token** (e.g. `"<a_100>"`, `"<b_230>"`, `"<c_0>"`).

Example:
```json
[
  "<a_100>",
  "<a_102>",
  "<a_105>",
  "<a_106>",
  "<a_108>",
  "<a_109>",
  "<a_111>",
  "<a_115>",
  "<a_116>",
  "<a_118>",
  "<a_11>",
  ......
]
```

#### 1.2 `id2sid.json`

This is used to build Trie for constrained beam search. Each candidate (item) is represented by SID: the value is the concatenation of the three tokens in the array (e.g. `<a_102><b_178><c_228>`).

Example:
```json
{
    "3681": [
        "<a_102>",
        "<b_178>",
        "<c_228>"
    ],
    "3682": [
        "<a_135>",
        "<b_237>",
        "<c_165>"
    ]
}
```

#### 1.3 SFT data (`sft/train.json`, `valid.json`, `test.json`)

Example:

```json
{
    "system": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.",
    "instruction": "Can you predict the next possible item that the user may expect?",
    "input": "The user has interacted with items <a_14><b_221><c_27>, <a_58><b_86><c_2>, <a_221><b_23><c_236>, <a_102><b_164><c_35> in chronological order. Can you predict the next possible item that the user may expect?",
    "output": "<a_58><b_138><c_72>"
}
```


#### 1.4 RL data (`rl/train.json`, `valid.json`, `test.json`)

Example:

```json
{
    "data_source": "Industrial_and_Scientific",
    "prompt": [
      {
        "role": "system",
        "content": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
      },
      {
        "role": "user",
        "content": "Can you predict the next possible item the user may expect, given the following chronological interaction history: <a_46><b_127><c_11>, <a_109><b_82><c_159>, <a_215><b_255><c_82>, <a_74><b_21><c_124>, <a_128><b_195><c_181>, <a_42><b_119><c_86>, <a_61><b_31><c_174>, <a_61><b_21><c_4>, <a_87><b_177><c_42>, <a_100><b_108><c_21>"
      }
    ],
    "ability": "seq_rec",
    "reward_model": {
      "style": "rule",
      "ground_truth": "<a_206><b_91><c_113>"
    },
    "extra_info": {
      "split": "test",
      "index": 3643,
      "task": "task1_sid_sft"
    }
}
```

### 2. SFT data Registry

You must register your SFT dataset in **`data/dataset_info.json`**, pointing `file_name` to the JSON under `data/` (e.g. `Industrial_and_Scientific/sft/train.json`) and mapping columns as below:

```json
  "Industrial_and_Scientific_train": {
    "file_name": "Industrial_and_Scientific/sft/train.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "system": "system"
    }
  },
  "Industrial_and_Scientific_valid": {
    "file_name": "Industrial_and_Scientific/sft/valid.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output",
      "system": "system"
    }
  }
```

### 3. Create your SFT yaml config

The SFT config, for example: **`examples/train_full/Industrial_and_Scientific/industry_rec_full_sft_0.5b_dsz0.yaml`**. Use it (or copy and edit) to run SFT with LlamaFactory.

**Ensure `dataset` and `eval_dataset` match the keys you added in `data/dataset_info.json`, and `add_tokens_list` points to your `new_tokens.json`.**

Then run:

```bash
llamafactory-cli train PATH_TO_YOUR_YAML.yaml
```

(or use `bash sft.sh` after uncommenting the corresponding line).

## Citation & acknowledgments

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — SFT training framework  
- [TRL](https://github.com/huggingface/trl) — Reinforcement learning training
- [MiniOneRec](https://github.com/AkaliKong/MiniOneRec) -  First fully open-source generative recommendation framework
