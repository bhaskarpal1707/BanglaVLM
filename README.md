# 🇧🇩 BanglaVLM

## Vision-Language Fine-Tuning & Multi-Dataset Evaluation for Bengali

<p align="center">

**Parameter-Efficient Bengali Vision-Language Learning with Cross-Dataset Evaluation**

</p>

<p align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-FP16-orange.svg)]()
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-HuggingFace-yellow.svg)]()
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-green.svg)]()
[![Model](https://img.shields.io/badge/Base%20Model-Sm olVLM2--2.2B-purple.svg)]()

</p>

---

# 📌 Table of Contents

1. [Overview](#-overview)
2. [Research Motivation](#-research-motivation)
3. [Research Questions](#-research-questions)
4. [Research Objectives](#-research-objectives)
5. [Research Contributions](#-research-contributions)
6. [System Overview](#-system-overview)
7. [Base Vision-Language Model](#-base-vision-language-model)
8. [Datasets](#-datasets)
9. [Dataset Roles](#-dataset-roles)
10. [Experimental Design](#-experimental-design)
11. [Experiment 1](#-experiment-1-baseline-vs-single-dataset-fine-tuning)
12. [Experiment 2](#-experiment-2-multi-dataset-joint-fine-tuning)
13. [Experiment 3](#-experiment-3-cross-dataset-generalization)
14. [Training Methodology](#-training-methodology)
15. [LoRA Configuration](#-lora-configuration)
16. [Evaluation](#-evaluation)
17. [Results](#-results)
18. [Qualitative Analysis](#-qualitative-analysis)
19. [Repository Structure](#-repository-structure)
20. [Predictions](#-predictions)
21. [Results Directory](#-results-directory)
22. [Project Summary](#-project-summary)
23. [Visualizations](#-visualizations)
24. [Quickstart](#-quickstart)
25. [Inference](#-inference)
26. [Hugging Face Models](#-hugging-face-models)
27. [Reproducibility](#-reproducibility)
28. [Research Findings](#-research-findings)
29. [Limitations](#-limitations)
30. [Future Work](#-future-work)
31. [Citation](#-citation)
32. [Author](#-author)

---

# 🔬 Overview

**BanglaVLM** is an end-to-end research project investigating Bengali multimodal Vision-Language Models (VLMs) through **Parameter-Efficient Fine-Tuning (PEFT)** and **multi-dataset evaluation**.

The project uses:

> **HuggingFaceTB/SmolVLM2-2.2B-Instruct**

as the base vision-language model and adapts it to Bengali Visual Question Answering (VQA) using **Low-Rank Adaptation (LoRA)**.

The central goal is not simply to obtain a high score on one Bengali VQA dataset.

Instead, BanglaVLM investigates a broader research question:

> **Can a relatively lightweight Vision-Language Model acquire Bengali multimodal reasoning capabilities through parameter-efficient fine-tuning, while retaining the ability to generalize across different Bengali visual datasets?**

The project therefore evaluates three increasingly challenging settings:

* **Single-dataset adaptation**
* **Multi-dataset joint adaptation**
* **Cross-dataset / zero-shot generalization**

This creates a complete experimental progression from **in-domain learning → multi-task learning → unseen-domain evaluation**.

---

# 🎯 Research Motivation

Bengali is one of the world's major languages, yet multimodal AI systems remain substantially less developed for Bengali compared with English and other high-resource languages.

Modern Vision-Language Models have demonstrated strong capabilities in:

* Image understanding
* Visual question answering
* Image description
* Instruction following
* Visual grounding
* Multimodal reasoning

However, these capabilities do not automatically translate into strong Bengali-language performance.

A model may correctly understand the visual content while still producing:

* English-biased responses
* Incorrect Bengali vocabulary
* Weak Bengali sentence structures
* Generic descriptions
* Hallucinated visual information
* Dataset-specific response patterns

BanglaVLM investigates whether these limitations can be reduced through **parameter-efficient Bengali adaptation** rather than full model fine-tuning.

---

# ❓ Research Questions

BanglaVLM is organized around three primary research questions.

### RQ1 — In-Domain Bengali Adaptation

> Does LoRA fine-tuning on a Bengali VQA dataset significantly improve the Bengali visual-language performance of SmolVLM2-2.2B?

This is evaluated through the transition:

**Base SmolVLM2 → BanglaVLM-v1**

---

### RQ2 — Multi-Dataset Learning

> Does jointly training on structurally different Bengali VQA datasets improve multimodal adaptability compared with single-dataset training?

This is evaluated through:

**Bangla-Bayanno + ChitroJera → BanglaVLM-v2**

---

### RQ3 — Cross-Dataset Generalization

> Can a Bengali VLM trained on existing Bengali visual datasets generalize to an unseen Bengali visual benchmark?

This is evaluated using:

**BanglaVerse**

as an unseen test domain.

---

# 🎯 Research Objectives

The project has four major objectives:

### 1. Bengali Multimodal Adaptation

Adapt a lightweight VLM for Bengali visual-language interaction using parameter-efficient training.

### 2. Multi-Dataset Learning

Investigate whether training on multiple datasets with different QA structures improves general-purpose Bengali VQA capability.

### 3. Cross-Dataset Generalization

Measure whether learned Bengali visual-language representations transfer to an unseen dataset.

### 4. Systematic Evaluation

Maintain a complete experimental record containing:

* Predictions
* Quantitative metrics
* Runtime information
* Visualizations
* Analysis reports
* Cross-dataset comparisons

---

# 💡 Research Contributions

The project provides the following research-oriented contributions:

### Contribution 1 — Bengali VLM Adaptation

A lightweight VLM is adapted specifically for Bengali multimodal interaction.

### Contribution 2 — Parameter-Efficient Training

LoRA is used instead of updating the entire 2.2B-parameter backbone.

### Contribution 3 — Multi-Dataset Training

Bangla-Bayanno and ChitroJera are jointly used to investigate multi-task Bengali visual-language learning.

### Contribution 4 — Cross-Dataset Audit

BanglaVerse is deliberately used as an unseen evaluation dataset to measure generalization.

### Contribution 5 — Reproducible Experiment Organization

Predictions, results, summary reports and visualizations are maintained separately for each experimental stage.

---

# 🧠 System Overview

The overall research pipeline can be represented as:

```text
                         ┌──────────────────────────────┐
                         │ SmolVLM2-2.2B-Instruct       │
                         │ Pretrained Vision-Language    │
                         │ Model                         │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │ Bengali VQA Datasets         │
                         │                              │
                         │ • Bangla-Bayanno             │
                         │ • ChitroJera                 │
                         │ • BanglaVerse                │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │ Parameter-Efficient Fine-     │
                         │ Tuning with LoRA              │
                         └──────────────┬───────────────┘
                                        │
                    ┌───────────────────┴──────────────────┐
                    ▼                                      ▼
        ┌──────────────────────┐                ┌──────────────────────┐
        │ BanglaVLM-v1         │                │ BanglaVLM-v2         │
        │ Bayanno LoRA         │                │ Bayanno + ChitroJera │
        └──────────┬───────────┘                └──────────┬───────────┘
                   │                                       │
                   └────────────────┬──────────────────────┘
                                    ▼
                         ┌──────────────────────────────┐
                         │ Cross-Dataset Evaluation     │
                         │                              │
                         │ Bangla-Bayanno               │
                         │ ChitroJera                   │
                         │ BanglaVerse                  │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │ Quantitative + Qualitative   │
                         │ Evaluation                   │
                         └──────────────────────────────┘
```

---

# 🤖 Base Vision-Language Model

The project uses:

```text
HuggingFaceTB/SmolVLM2-2.2B-Instruct
```

as the base model.

SmolVLM2-2.2B provides the multimodal foundation for:

* Image encoding
* Visual understanding
* Text generation
* Instruction following
* Image-question interaction

Rather than modifying all model parameters, BanglaVLM introduces trainable LoRA adapters.

---

# 📚 Datasets

Three Bengali multimodal datasets are used.

| Dataset            | QA Pairs | Images | Primary Role                           |
| ------------------ | -------: | -----: | -------------------------------------- |
| **Bangla-Bayanno** |   53,817 |  4,673 | Primary training + in-domain benchmark |
| **ChitroJera**     |   12,231 | 12,231 | Multi-task fine-tuning + benchmark     |
| **BanglaVerse**    |    1,143 |  1,143 | Unseen zero-shot evaluation            |

The datasets serve deliberately different experimental purposes.

---

## 🇧🇩 Bangla-Bayanno

**Role:** Primary training dataset.

Bangla-Bayanno contains rich descriptive Bengali visual question-answer pairs.

It is used to investigate:

* Bengali visual grounding
* Descriptive generation
* Object and scene understanding
* Bengali response quality

It forms the training foundation for **BanglaVLM-v1**.

---

## 🖼️ ChitroJera

**Role:** Multi-task training and evaluation.

ChitroJera contains concise visual queries and instruction-following examples.

Its structural difference from Bangla-Bayanno makes it useful for investigating whether joint training improves the model's ability to handle different VQA styles.

---

## 🌏 BanglaVerse

**Role:** Unseen cross-dataset evaluation.

BanglaVerse is not used as the primary training source.

Instead, it acts as a stress test for generalization.

The model must therefore transfer:

```text
Bengali language knowledge
          +
visual understanding
          +
instruction following
          ↓
new visual-question distribution
```

without direct training on the evaluation dataset.

---

# 🧪 Experimental Design

BanglaVLM consists of three major experiments.

```text
Experiment 1
    │
    ├── Base SmolVLM2
    └── BanglaVLM-v1
          │
          ▼
    Single-Dataset Adaptation


Experiment 2
    │
    ├── Bangla-Bayanno
    └── ChitroJera
          │
          ▼
    BanglaVLM-v2
          │
          ▼
    Multi-Dataset Adaptation


Experiment 3
    │
    ├── Base SmolVLM2
    ├── BanglaVLM-v1
    └── BanglaVLM-v2
          │
          ▼
    BanglaVerse
          │
          ▼
    Zero-Shot Generalization Audit
```

---

# 🧪 Experiment 1: Baseline vs Single-Dataset Fine-Tuning

## Objective

Determine whether Bengali-specific LoRA adaptation improves the base SmolVLM2 model on Bangla-Bayanno.

### Models

```text
Model A → Base SmolVLM2-2.2B-Instruct

Model B → BanglaVLM-v1
           └── LoRA trained on Bangla-Bayanno
```

### Evaluation

The models are evaluated on the Bangla-Bayanno test split.

The experiment measures the change in Bengali multimodal performance after adaptation.

### Key Finding

Fine-tuning improves Bengali response quality, including stronger lexical and structural alignment and reduced English bias.

---

# 🧪 Experiment 2: Multi-Dataset Joint Fine-Tuning

## Objective

Investigate whether training on structurally different Bengali VQA datasets improves general-purpose multimodal adaptability.

### Training Data

```text
Bangla-Bayanno
       +
ChitroJera
       ↓
Joint Training
       ↓
BanglaVLM-v2
```

### Hypothesis

Joint training may expose the model to:

* Different question structures
* Different answer lengths
* Different visual concepts
* Different instruction patterns
* Different Bengali linguistic formulations

This potentially creates a more flexible Bengali VLM.

### Key Finding

BanglaVLM-v2 maintains strong descriptive capability while improving performance on concise instruction-following style questions.

---

# 🧪 Experiment 3: Cross-Dataset Generalization

## Objective

Evaluate whether Bengali multimodal knowledge learned from the training datasets transfers to an unseen dataset.

### Evaluation Setup

```text
                  ┌───────────────────┐
                  │ Base SmolVLM2     │
                  └─────────┬─────────┘
                            │
                            ▼
                       BanglaVerse


                  ┌───────────────────┐
                  │ BanglaVLM-v1     │
                  └─────────┬─────────┘
                            │
                            ▼
                       BanglaVerse


                  ┌───────────────────┐
                  │ BanglaVLM-v2     │
                  └─────────┬─────────┘
                            │
                            ▼
                       BanglaVerse
```

BanglaVerse therefore functions as a cross-domain audit.

### Key Finding

BanglaVLM-v2 demonstrates the strongest cross-dataset transfer among the evaluated fine-tuned variants.

---

# ⚙️ Training Methodology

BanglaVLM uses **Parameter-Efficient Fine-Tuning (PEFT)**.

Instead of updating the complete VLM, LoRA introduces a small number of trainable parameters into selected linear layers.

The conceptual transformation is:

```text
Original Layer:

        W


LoRA Adapted Layer:

        W + ΔW

where:

        ΔW = B × A
```

The original pretrained weights remain largely frozen while the LoRA matrices are optimized.

This substantially reduces:

* Trainable parameter count
* GPU memory requirements
* Storage requirements
* Fine-tuning cost

while preserving the pretrained multimodal knowledge of the backbone.

---

# 🔧 LoRA Configuration

The experiments use the following configuration:

| Parameter             | Value                                  |
| --------------------- | -------------------------------------- |
| Base Model            | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` |
| Fine-Tuning           | LoRA / PEFT                            |
| LoRA Rank             | `r = 16`                               |
| LoRA Alpha            | `α = 32`                               |
| LoRA Dropout          | `0.05`                                 |
| Target Modules        | `all-linear`                           |
| Learning Rate         | `2e-4`                                 |
| Scheduler             | Cosine                                 |
| Warmup                | 30 steps                               |
| Precision             | FP16                                   |
| Per-device Batch Size | 1                                      |
| Gradient Accumulation | 8                                      |
| Effective Batch Size  | 8                                      |
| Hardware              | NVIDIA Tesla T4                        |

---

# 📊 Evaluation Methodology

Evaluation is performed from both **quantitative** and **qualitative** perspectives.

## Quantitative Evaluation

The project records model predictions and evaluation metrics for:

* Bangla-Bayanno
* ChitroJera
* BanglaVerse

The experimental repository separates raw predictions from processed evaluation results.

---

## Qualitative Evaluation

Quantitative scores alone cannot fully capture Bengali VQA quality.

Therefore, qualitative inspection is used to examine:

### Visual Grounding

Does the generated answer correspond to the actual image?

### Bengali Language Quality

Is the response naturally expressed in Bengali?

### Answer Relevance

Does the model directly answer the question?

### Hallucination

Does the model introduce objects or facts that are not present?

### Descriptive Granularity

Does the model provide sufficient visual detail when required?

### Instruction Following

Does the model respect the expected answer style?

---

# 📈 Results

The final benchmark compares:

```text
Base SmolVLM2
       │
       ├── Bangla-Bayanno
       ├── ChitroJera
       └── BanglaVerse
       
BanglaVLM-v1
       │
       ├── Bangla-Bayanno
       ├── ChitroJera
       └── BanglaVerse
       
BanglaVLM-v2
       │
       ├── Bangla-Bayanno
       ├── ChitroJera
       └── BanglaVerse
```

### Cross-Dataset Summary

| Model         | Bangla-Bayanno        | ChitroJera            | BanglaVerse                       |
| ------------- | --------------------- | --------------------- | --------------------------------- |
| Base SmolVLM2 | Baseline              | Baseline              | Baseline                          |
| BanglaVLM-v1  | **Highest in-domain** | Moderate              | Evaluated                         |
| BanglaVLM-v2  | High in-domain        | **Highest in-domain** | **Best zero-shot generalization** |

The central result is that joint multi-dataset training produces a model with stronger transfer capability than single-dataset adaptation.

---

# 🔍 Qualitative Analysis

The project also performs visual inspection of generated answers.

The qualitative evaluation focuses on:

```text
Image
  │
  ▼
Question
  │
  ▼
Model Prediction
  │
  ├── Bengali correctness
  ├── Visual grounding
  ├── Relevance
  ├── Completeness
  └── Hallucination
```

Visual comparisons are generated for:

* BanglaVLM-v2 on Bangla-Bayanno
* BanglaVLM-v2 on ChitroJera
* BanglaVLM-v2 on unseen BanglaVerse

---

# 📁 Repository Structure

```text
BanglaVLM/
│
├── 📁 notebooks/
│   └── Experimental Google Colab notebooks
│
├── 📁 models/
│   └── Local model/adapters and related artifacts
│
├── 📁 predictions/
│   ├── experiment1_baseline_predictions.json
│   ├── experiment1_test_predictions.json
│   ├── experiment2_all_predictions.json
│   ├── experiment3_BanglaVLM-v1_BanglaVerse.json
│   ├── experiment3_BanglaVLM-v2_BanglaVerse.json
│   └── experiment3_cross_predictions.json
│
├── 📁 results/
│   ├── experiment1/
│   ├── experiment2/
│   └── experiment3/
│
├── 📁 project_summary/
│   ├── audit_all_metrics.csv
│   ├── comprehensive_analysis_report.md
│   ├── final_results.csv
│   └── total_runtime_summary.csv
│
├── 📁 timing/
│   └── Detailed execution and runtime analytics
│
└── 📁 visualizations/
    └── High-resolution plots, charts and qualitative galleries
```

---

# 📦 Predictions

The `predictions/` directory stores model-generated outputs before or alongside metric computation.

This makes it possible to independently inspect:

* Model responses
* Dataset samples
* Experiment-specific outputs
* Cross-dataset predictions

Current prediction artifacts include:

```text
experiment1_baseline_predictions.json
experiment1_test_predictions.json
experiment2_all_predictions.json

experiment3_BanglaVLM-v1_BanglaVerse.json
experiment3_BanglaVLM-v2_BanglaVerse.json
experiment3_cross_predictions.json
```

Keeping predictions separately is important for reproducibility because metrics can be recomputed from the stored model outputs.

---

# 📊 Results Directory

The `results/` directory is organized by experiment:

```text
results/
│
├── experiment1/
│
├── experiment2/
│
└── experiment3/
```

This organization separates the evaluation artifacts of the three experimental stages.

### Experiment 1

Contains results associated with:

```text
Base Model
      vs
BanglaVLM-v1
```

### Experiment 2

Contains results associated with:

```text
Joint Training
      ↓
BanglaVLM-v2
```

### Experiment 3

Contains cross-dataset evaluation and unseen-domain analysis.

---

# 📝 Project Summary

The `project_summary/` directory contains consolidated project-level artifacts.

```text
project_summary/
│
├── audit_all_metrics.csv
├── comprehensive_analysis_report.md
├── final_results.csv
└── total_runtime_summary.csv
```

### `audit_all_metrics.csv`

Contains consolidated metric-level evaluation information.

### `comprehensive_analysis_report.md`

Contains the broader analysis and interpretation of the experiments.

### `final_results.csv`

Contains the consolidated final benchmark results.

### `total_runtime_summary.csv`

Contains runtime-related information from the experimental pipeline.

---

# 🎨 Visualizations

The `visualizations/` directory contains generated research figures.

Typical visualization categories include:

### Training Curves

```text
Training Step
      ↓
Loss / Evaluation Metric
      ↓
Convergence Analysis
```

### Performance Comparisons

Comparison of:

* Base model
* BanglaVLM-v1
* BanglaVLM-v2

across datasets.

### Qualitative Galleries

Visual grids showing:

```text
Image
Question
Reference Answer
Model Prediction
```

for representative examples.

These visualizations are useful for identifying failure modes that numerical metrics alone may not reveal.

---

# 🚀 Quickstart

## 1. Install Dependencies

```bash
pip install torch
pip install transformers
pip install peft
pip install accelerate
pip install pillow
```

---

## 2. Load the Base Model

```python
import torch

from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq
)

base_model_id = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

processor = AutoProcessor.from_pretrained(
    base_model_id
)

model = AutoModelForVision2Seq.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
```

---

# 🤗 Loading BanglaVLM-v2

The joint Bengali LoRA adapter can be loaded on top of the base model.

```python
from peft import PeftModel

peft_model_id = (
    "bhaskar1707/"
    "smolvlm2-bangla-bayanno-chitrojera-lora"
)

model = PeftModel.from_pretrained(
    model,
    peft_model_id
)
```

---

# 🖼️ Inference

```python
from PIL import Image

image_path = "sample_image.jpg"

image = Image.open(image_path)

prompt = """
<image>
এই ছবিতে কী দেখা যাচ্ছে বিস্তারিত বাংলায় বলুন।
"""

inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt"
).to("cuda")

generated_ids = model.generate(
    **inputs,
    max_new_tokens=128
)

generated_text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True
)

print("BanglaVLM Response:")
print(generated_text[0])
```

---

# 🧠 Model Variants

BanglaVLM currently contains two major LoRA variants.

## BanglaVLM-v1

```text
Base Model
     +
Bangla-Bayanno
     ↓
BanglaVLM-v1
```

Purpose:

> Bengali single-dataset adaptation.

---

## BanglaVLM-v2

```text
Base Model
     +
Bangla-Bayanno
     +
ChitroJera
     ↓
BanglaVLM-v2
```

Purpose:

> Multi-dataset Bengali multimodal adaptation and stronger cross-dataset transfer.

---

# 🤗 Hugging Face Models

The trained LoRA adapters are available through Hugging Face.

### BanglaVLM-v1

```text
bhaskar1707/smolvlm2-bangla-bayanno-lora
```

Training:

```text
Bangla-Bayanno
```

Primary purpose:

> Single-dataset Bengali VQA adaptation.

---

### BanglaVLM-v2

```text
bhaskar1707/smolvlm2-bangla-bayanno-chitrojera-lora
```

Training:

```text
Bangla-Bayanno
+
ChitroJera
```

Primary purpose:

> Joint Bengali multimodal learning and cross-dataset generalization.

---

# 📦 Adapter Artifacts

A LoRA repository typically contains lightweight adapter artifacts such as:

```text
adapter_model.safetensors
adapter_config.json

tokenizer_config.json
preprocessor_config.json
```

The adapter weights contain the learned LoRA parameters rather than a complete copy of the original 2.2B backbone.

This makes adapter distribution substantially lighter than distributing a complete fine-tuned model.

---

# 🔁 Reproducibility

The project is organized to make experiments easier to reproduce.

A reproducible experiment should preserve:

```text
Dataset
   ↓
Preprocessing
   ↓
Training Configuration
   ↓
LoRA Configuration
   ↓
Model Checkpoint
   ↓
Inference
   ↓
Predictions
   ↓
Metrics
   ↓
Visual Analysis
```

The repository therefore separates:

* Models
* Predictions
* Results
* Summary reports
* Timing information
* Visualizations
* Notebooks

This separation reduces ambiguity between raw inference outputs and derived evaluation results.

---

# 📌 Research Findings

The experiments support several important observations.

## Finding 1 — Bengali Fine-Tuning Matters

The base VLM possesses general multimodal capabilities but is not specifically optimized for Bengali VQA.

Bengali-specific LoRA adaptation improves its behavior on Bengali visual-language tasks.

---

## Finding 2 — Single-Dataset Fine-Tuning Works Well In-Domain

BanglaVLM-v1 achieves strong performance on the dataset it was trained on.

This demonstrates effective domain adaptation.

However, strong in-domain performance alone does not guarantee broad generalization.

---

## Finding 3 — Multi-Dataset Training Improves Adaptability

Combining Bangla-Bayanno and ChitroJera exposes the model to different visual-question structures.

BanglaVLM-v2 therefore provides a broader adaptation than a model trained on a single dataset.

---

## Finding 4 — Cross-Dataset Testing Is Essential

A model can perform well on its training distribution while struggling on a different dataset.

The inclusion of BanglaVerse therefore provides a stronger test of whether the model has learned transferable Bengali multimodal capabilities.

---

## Finding 5 — BanglaVLM-v2 Shows Stronger Zero-Shot Transfer

Among the evaluated variants, BanglaVLM-v2 provides the strongest generalization to the unseen BanglaVerse benchmark.

This suggests that exposure to multiple Bengali VQA distributions can improve cross-domain transfer.

---

# ⚠️ Limitations

Several limitations should be considered.

### 1. Dataset Distribution

Performance is dependent on the visual and linguistic distributions represented by the available datasets.

### 2. Limited Bengali VLM Ecosystem

The number of publicly available Bengali multimodal benchmarks remains considerably smaller than comparable English resources.

### 3. Metric Limitations

Automatic lexical metrics cannot completely capture:

* Semantic correctness
* Visual grounding
* Bengali fluency
* Hallucination
* Reasoning quality

### 4. Model Scale

The research focuses on the 2.2B-parameter SmolVLM2 backbone.

Results may differ for substantially larger multimodal architectures.

### 5. Zero-Shot Evaluation Scope

BanglaVerse provides an important unseen-domain test, but it represents only one unseen benchmark.

Broader conclusions would require additional unseen datasets.

---

# 🔮 Future Work

Potential extensions include:

## 1. Larger Bengali Multimodal Models

Evaluate larger VLM architectures under the same experimental protocol.

---

## 2. More Bengali Datasets

Expand training and evaluation to additional Bengali visual-language datasets.

---

## 3. Human Evaluation

Introduce human assessment for:

* Bengali fluency
* Visual grounding
* Relevance
* Hallucination
* Completeness
* Instruction following

---

## 4. Bengali Multimodal Instruction Tuning

Create a larger unified Bengali multimodal instruction dataset covering:

```text
VQA
+
Captioning
+
OCR
+
Visual Reasoning
+
Object Understanding
+
Spatial Reasoning
```

---

## 5. Better Cross-Domain Evaluation

Introduce multiple unseen datasets rather than relying on a single zero-shot benchmark.

---

## 6. Ablation Studies

Future experiments can systematically investigate:

* LoRA rank
* LoRA alpha
* Target modules
* Dataset combinations
* Training size
* Learning rate
* Prompt formulation
* Adapter merging

---

# 🧪 Recommended Ablation Matrix

A future extension can follow:

```text
                         Training Data
                              │
             ┌────────────────┼────────────────┐
             │                │                │
          Bayanno       ChitroJera      Bayanno + ChitroJera
             │                │                │
             └────────────────┼────────────────┘
                              │
                              ▼
                         Model Variants
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
          In-Domain      Cross-Dataset      Zero-Shot
```

This would provide a stronger empirical basis for understanding where the observed gains originate.

---

# 📚 Project Philosophy

BanglaVLM is designed around a simple principle:

> **High in-domain performance is not sufficient to claim robust Bengali multimodal intelligence.**

A stronger Bengali VLM should demonstrate:

```text
Language Adaptation
        +
Visual Understanding
        +
Instruction Following
        +
Multi-Dataset Robustness
        +
Cross-Dataset Generalization
```

Therefore, the project emphasizes **evaluation across distributions**, not only optimization on a single benchmark.

---

# 🏁 Conclusion

BanglaVLM presents a systematic investigation of Bengali Vision-Language Model adaptation using parameter-efficient fine-tuning.

Starting from:

```text
SmolVLM2-2.2B-Instruct
```

the project progresses through:

```text
Single-Dataset Bengali Adaptation
              ↓
        BanglaVLM-v1
              ↓
Multi-Dataset Bengali Adaptation
              ↓
        BanglaVLM-v2
              ↓
Unseen Cross-Dataset Evaluation
              ↓
          BanglaVerse
```

The experimental results indicate that Bengali-specific LoRA adaptation substantially improves task alignment, while joint training on multiple Bengali VQA datasets can further improve transfer to unseen visual-language distributions.

The resulting framework provides a practical foundation for continued research into:

**Low-Resource Bengali Multimodal AI, Bengali VQA, Parameter-Efficient Fine-Tuning, and Cross-Dataset Vision-Language Generalization.**

---

# 📖 Citation

If you use this project, model adapters, experiments, or analysis in academic work, please cite the project as appropriate.

```bibtex
@misc{pal2026banglavlm,
  title        = {BanglaVLM: Vision-Language Fine-Tuning and Multi-Dataset Evaluation for Bengali},
  author       = {Bhaskar Pal},
  year         = {2026},
  note         = {Bengali Vision-Language Model research project}
}
```

---

# 👨‍💻 Author

### Bhaskar Pal

**Researcher & Developer**

* GitHub: `@bhaskarpal1707`
* Hugging Face: `@bhaskar1707`

---

# ⭐ Acknowledgements

This project builds upon the open-source ecosystem surrounding:

* Hugging Face Transformers
* Hugging Face PEFT
* PyTorch
* SmolVLM2
* Bengali VQA datasets
* Open-source multimodal research

The project aims to contribute further resources and empirical findings toward the development of Bengali multimodal AI.

---

<p align="center">

### 🇧🇩 Building Better Multimodal AI for Bengali

**BanglaVLM — Bengali Vision-Language Research**

</p>
