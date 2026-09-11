# BanglaVLM — Comprehensive Project Analysis Report

*Auto-generated from Drive-saved artifacts. All numbers below are measured, not estimated.*

## 1. Project Overview

This project fine-tuned `HuggingFaceTB/SmolVLM2-2.2B-Instruct` with LoRA for Bengali Visual Question Answering, across three experiments:
- **Experiment 1**: LoRA fine-tuning on Bangla-Bayanno alone → **BanglaVLM-v1**
- **Experiment 2**: LoRA fine-tuning on Bangla-Bayanno + ChitroJera combined → **BanglaVLM-v2**
- **Experiment 3**: Cross-dataset evaluation of Base SmolVLM2, BanglaVLM-v1, and BanglaVLM-v2 on Bangla-Bayanno, ChitroJera, and the fully **unseen** BanglaVerse dataset, to test generalization.

## 2. Datasets

- **Bangla-Bayanno**: 53,817 QA pairs over 4,673 images. Role: Training + evaluation (Exp1, Exp2, Exp3).
- **ChitroJera**: 12,231 QA pairs over 12,231 images. Role: Training + evaluation (Exp2, Exp3).
- **BanglaVerse**: 1,143 QA pairs over 1,143 images. Role: Evaluation ONLY — never used in training (cross-dataset generalization test, Exp3).

## 3. Training Resource Constraints

Training ran on a free-tier Google Colab T4 GPU (16GB), which made full-dataset fine-tuning infeasible (an early full-dataset run was projected at 175+ hours). To fit realistic session limits, training used a capped, seeded subset:

- **Experiment 1 (Bangla-Bayanno)**: used 7,000 of 53,817 available cleaned QA pairs (**13.01%**) for training, capped at 1000 optimizer steps.
- **Experiment 2 (Bangla-Bayanno + ChitroJera (combined))**: used 7,000 of 65,628 available cleaned QA pairs (**10.67%**) for training, capped at 1000 optimizer steps.

This is a deliberate, documented trade-off — not a data-quality shortcut. Validation and test sets were **not** subsampled from the training cap; they reflect the full cleaned image-level split.

## 4. Training Dynamics

- **experiment1**: reached step 1000, final train loss **0.1889**, final val loss **0.2016**.
- **experiment2**: reached step 1000, final train loss **0.1931**, final val loss **0.2105**.

## 5. All Model × Dataset Combinations Run

| experiment   | model         | dataset                     | task                           |   n_examples |
|:-------------|:--------------|:----------------------------|:-------------------------------|-------------:|
| Exp1         | Base SmolVLM2 | Bangla-Bayanno              | inference (baseline, sampled)  |          300 |
| Exp1         | BanglaVLM-v1  | Bangla-Bayanno              | TRAINING                       |          nan |
| Exp1         | BanglaVLM-v1  | Bangla-Bayanno              | inference (full test set)      |         5004 |
| Exp2         | BanglaVLM-v2  | Bangla-Bayanno + ChitroJera | TRAINING                       |          nan |
| Exp2         | Base SmolVLM2 | Bangla-Bayanno              | inference (evaluation)         |         5004 |
| Exp2         | Base SmolVLM2 | ChitroJera                  | inference (evaluation)         |         1224 |
| Exp2         | BanglaVLM-v1  | Bangla-Bayanno              | inference (evaluation)         |         5004 |
| Exp2         | BanglaVLM-v1  | ChitroJera                  | inference (evaluation)         |         1224 |
| Exp2         | BanglaVLM-v2  | Bangla-Bayanno              | inference (evaluation)         |         5004 |
| Exp2         | BanglaVLM-v2  | ChitroJera                  | inference (evaluation)         |         1224 |
| Exp3         | Base SmolVLM2 | Bangla-Bayanno              | inference (cross-dataset eval) |          500 |
| Exp3         | Base SmolVLM2 | ChitroJera                  | inference (cross-dataset eval) |          500 |
| Exp3         | Base SmolVLM2 | BanglaVerse (unseen)        | inference (cross-dataset eval) |          500 |
| Exp3         | BanglaVLM-v1  | Bangla-Bayanno              | inference (cross-dataset eval) |          500 |
| Exp3         | BanglaVLM-v1  | ChitroJera                  | inference (cross-dataset eval) |          500 |
| Exp3         | BanglaVLM-v1  | BanglaVerse (unseen)        | inference (cross-dataset eval) |          500 |
| Exp3         | BanglaVLM-v2  | Bangla-Bayanno              | inference (cross-dataset eval) |          500 |
| Exp3         | BanglaVLM-v2  | ChitroJera                  | inference (cross-dataset eval) |          500 |
| Exp3         | BanglaVLM-v2  | BanglaVerse (unseen)        | inference (cross-dataset eval) |          500 |

## 6. Time Spent Per Stage

| stage                 |   experiment1 |   experiment2 |   experiment3 |
|:----------------------|--------------:|--------------:|--------------:|
| baseline_inference    |           5.6 |           0   |           0   |
| baseline_inference_v2 |           4.7 |           0   |           0   |
| dataset_download      |          58.3 |           0.1 |           6   |
| evaluation            |          73.5 |         163.4 |          97.8 |
| model_loading         |           3.1 |          16.8 |           0   |
| preprocessing         |           0.1 |           0   |           5.9 |
| training              |         153.2 |           0   |           0   |
| visualization         |           0   |           0   |           0   |

**Total measured wall-clock time across all recorded stages: 588 minutes (~9.8 hours).**

## 7. Evaluation Results

| experiment   | model         | dataset        |   exact_match |   normalized_exact_match |   rouge_l |    n |
|:-------------|:--------------|:---------------|--------------:|-------------------------:|----------:|-----:|
| Exp1         | Base SmolVLM2 | Bangla-Bayanno |        0.3067 |                   0.3067 |    0.3122 |  300 |
| Exp1         | BanglaVLM-v1  | Bangla-Bayanno |        0.3347 |                   0.3351 |    0.3417 | 5004 |
| Exp2         | Base SmolVLM2 | bayanno        |        0      |                   0      |    0.0023 | 5004 |
| Exp2         | Base SmolVLM2 | chitrojera     |        0      |                   0      |    0.0018 | 1224 |
| Exp2         | BanglaVLM-v1  | bayanno        |        0.3347 |                   0.3351 |    0.3417 | 5004 |
| Exp2         | BanglaVLM-v1  | chitrojera     |        0.031  |                   0.031  |    0.0534 | 1224 |
| Exp2         | BanglaVLM-v2  | bayanno        |        0.3233 |                   0.3237 |    0.3311 | 5004 |
| Exp2         | BanglaVLM-v2  | chitrojera     |        0.0629 |                   0.0629 |    0.0995 | 1224 |
| Exp3         | Base SmolVLM2 | Bangla-Bayanno |        0      |                   0      |    0.0006 |  500 |
| Exp3         | Base SmolVLM2 | ChitroJera     |        0      |                   0      |    0.0013 |  500 |
| Exp3         | Base SmolVLM2 | BanglaVerse    |        0      |                   0      |    0.0055 |  500 |
| Exp3         | BanglaVLM-v1  | Bangla-Bayanno |        0.342  |                   0.342  |    0.3473 |  500 |
| Exp3         | BanglaVLM-v1  | ChitroJera     |        0.032  |                   0.032  |    0.0517 |  500 |
| Exp3         | BanglaVLM-v1  | BanglaVerse    |        0      |                   0      |    0.0037 |  500 |
| Exp3         | BanglaVLM-v2  | Bangla-Bayanno |        0.312  |                   0.312  |    0.3193 |  500 |
| Exp3         | BanglaVLM-v2  | ChitroJera     |        0.06   |                   0.06   |    0.0936 |  500 |
| Exp3         | BanglaVLM-v2  | BanglaVerse    |        0.004  |                   0.004  |    0.0156 |  500 |

### Analysis

- Fine-tuning on Bangla-Bayanno alone (BanglaVLM-v1) changed Normalized Exact Match on the in-domain test set by **+2.8 percentage points** versus the base model — a modest, not dramatic, improvement given the constrained training budget.
- On **Bangla-Bayanno**, adding ChitroJera to training (BanglaVLM-v2 vs v1) changed Normalized Exact Match by **-3.0 points**.
- On **ChitroJera**, adding ChitroJera to training (BanglaVLM-v2 vs v1) changed Normalized Exact Match by **+2.8 points**.
- On **BanglaVerse**, adding ChitroJera to training (BanglaVLM-v2 vs v1) changed Normalized Exact Match by **+0.4 points**.

  BanglaVerse was **never used in training** for any model — its results are the cleanest signal of true generalization to unseen Bengali visual content, as opposed to memorizing training-distribution patterns.

## 8. GPU Utilization

| experiment   |   avg_gpu_util_pct |   peak_gpu_util_pct |   avg_vram_mb |   peak_vram_mb |
|:-------------|-------------------:|--------------------:|--------------:|---------------:|
| experiment1  |               49.4 |                  94 |          6303 |           6304 |
| experiment2  |               48.8 |                 100 |          6317 |           6328 |

## 9. Caveats & Honest Limitations

- All numbers in this report are measured directly from saved artifacts on Drive — none are fabricated, estimated, or taken from external papers.
- Training used a resource-constrained subset (see Section 3) on a free-tier GPU; results should not be extrapolated to what a full-dataset, longer-schedule run would achieve.
- Evaluation on Notebook 3's cross-dataset comparison was capped to a fixed sample per test set (documented in that notebook) to keep runtime bounded — not the full test set in every case.
- ROUGE-L uses a whitespace tokenizer (not word-piece/BPE), appropriate for Bengali but not directly comparable to ROUGE-L scores computed with English-oriented tokenizers elsewhere.
