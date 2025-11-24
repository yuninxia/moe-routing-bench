# Complete PERFT-MoE Fine-Tuning Guide (Single H100)

## Repository Layout
- `commonsense/`: training + evaluation entrypoints for commonsense benchmarks (`finetune.py`, `commonsense_evaluate.py`, Slurm templates).
- `math/`: arithmetic-focused training/eval scripts mirroring the commonsense folder.
- `olmoe_modification/`: patched `OlmoeAdapter*` config/model registered when the train scripts import them.
- `mixtral_modification/`: Mixtral support (ignore unless you fine-tune Mixtral).
- `utils.py`: shared adapter initialisation utilities (LoRA / Parallel adapters).

## Environment Setup
Run everything from `3rdparty/PERFT-MoE/`. The adapters rely on Transformers ≥4.38 with `trust_remote_code`.
```bash
conda create -n perft python=3.10 -y
conda activate perft
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
pip install "datasets<3.0" transformers fire safetensors tqdm wandb
huggingface-cli login  # required to pull allenai/OLMoE-1B-7B-0924
```

## Data Preparation
All dataset utilities live under `commonsense/` and rely on Hugging Face datasets. Install a recent `datasets` release and allow remote code execution (`export HF_DATASETS_ALLOW_CODE=1`) because Social-IQa, ARC, and OpenBookQA ship custom loaders. Each record consumed by `finetune.py` must expose `instruction`, `input`, and `output` (for example `{"instruction": "Answer the question.", "input": "Question + options", "output": "C"}`).

**Download individual corpora and cache raw JSON:**
```bash
cd 3rdparty/PERFT-MoE
export HF_DATASETS_ALLOW_CODE=1
python commonsense/download_boolq.py
python commonsense/download_piqa.py
python commonsense/download_social_i_qa.py
python commonsense/download_hellaswag.py
python commonsense/download_winogrande.py
python commonsense/download_arc_challenge.py
python commonsense/download_arc_easy.py
python commonsense/download_openbookqa.py
```
Each command writes to `commonsense/raw/<dataset>_train.json` so you can inspect or customise the sources.

**Merge into the Commonsense170K training file:**
```bash
python commonsense/build_commonsense_170k.py --output commonsense/commonsense_170k.json
```

**Export evaluation fixtures:**
```bash
python commonsense/build_eval_sets.py
```
This generates `commonsense/dataset/<benchmark>/test.json`, matching the format expected by `commonsense/commonsense_evaluate.py`. Create analogous files under `math/dataset/` if you plan to run Math50K evaluations.

## Train OLMoE-1B-7B on One H100 (PERFT-R, BF16)
`finetune.py` loads BF16 weights onto the available GPU. The example below launches a PERFT-R run with LoRA rank 16 and Top-2/4 routing; adjust the hyperparameters to match other settings from the paper.
```bash
cd 3rdparty/PERFT-MoE
mkdir -p checkpoints log
CUDA_VISIBLE_DEVICES=0 python commonsense/finetune.py \
  --base_model allenai/OLMoE-1B-7B-0924 \
  --data_path commonsense_170k.json \
  --output_dir checkpoints/OLMoE-1B-7B-0924.top2of4_r16 \
  --batch_size 16 --micro_batch_size 4 \
  --num_epochs 3 --learning_rate 1e-5 \
  --cutoff_len 256 --val_set_size 120 \
  --eval_step 80 --save_step 80 \
  --shared_routing_adapter True \
  --shared_routing_adapter_num_experts 4 \
  --shared_routing_adapter_num_experts_per_tok 2 \
  --adapter_type LoRA --lora_r 16 --lora_alpha 32 --dropout 0.05
```
`batch_size` divided by `micro_batch_size` controls gradient accumulation (16/2 → 8). Use `--shared_routing_adapter_num_experts 2` and `--shared_routing_adapter_num_experts_per_tok 1` for Top-1/2. Swap `--adapter_type Parallel_Adapter --hidden_dim <d>` to reproduce the parallel-adapter configs.

## Evaluate
`commonsense/commonsense_evaluate.py` reloads the adapter weights and writes per-sample results to `commonsense/experiment/`.
```bash
CUDA_VISIBLE_DEVICES=0 python commonsense/commonsense_evaluate.py \
  --dataset boolq \
  --base_model allenai/OLMoE-1B-7B-0924 \
  --peft_model checkpoints/OLMoE-1B-7B-0924.top2of4_r16 \
  --name OLMoE-1B-7B.top2of4_r16 \
  --batch_size 16 --max_new_tokens 4
```
Repeat for each benchmark listed in the `--dataset` choices; logs are appended in `log/`.

## Tips
- Set `TOKENIZERS_PARALLELISM=false` to silence tokenizer warnings.
- If memory is tight, lower `--micro_batch_size` or truncate `--cutoff_len`.
- `math/finetune.py` and `math/math_evaluate.py` share the same flags when you move on to Math50K.
